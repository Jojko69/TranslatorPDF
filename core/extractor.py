"""
core/extractor.py
=================
Ekstrakcja struktury dokumentu z plików PDF i TXT.

Strategia dla PDF:
  - pymupdf (fitz)  → tekst z pozycją, czcionki, obrazy
  - pdfplumber      → detekcja i ekstrakcja tabel

Zwraca uporządkowaną listę DocElement — jeden element = jeden
blok treści (akapit, nagłówek, tabela, obraz, separator strony).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Struktura elementu dokumentu
# ---------------------------------------------------------------------------

@dataclass
class DocElement:
    """Jeden blok treści dokumentu."""

    kind: str
    # Możliwe wartości:
    #   "paragraph"  – zwykły akapit tekstu
    #   "heading"    – nagłówek (poziom 1–3)
    #   "list_item"  – element listy punktowanej lub numerowanej
    #   "table"      – tabela z komórkami
    #   "image"      – obraz osadzony w dokumencie
    #   "page_sep"   – separator strony (--- Strona X ---)

    # --- Tekst ---
    text: str = ""
    level: int = 1           # poziom nagłówka (1=H1, 2=H2, 3=H3)

    # --- Tabela ---
    table_data: Optional[List[List[str]]] = None   # wiersze × kolumny

    # --- Obraz ---
    image_data: Optional[bytes] = None
    image_ext: str = "png"

    # --- Metadane pozycji ---
    page: int = 0
    y_pos: float = 0.0       # pozycja pionowa na stronie (do sortowania)

    # --- Wyniki tłumaczenia (wypełniane przez pipeline) ---
    translated_text: str = ""
    translated_cells: Optional[List[List[str]]] = None


# ---------------------------------------------------------------------------
# Narzędzia pomocnicze
# ---------------------------------------------------------------------------

def _zachodzi(bbox1: Tuple, bbox2: Tuple, prog: float = 0.4) -> bool:
    """
    Sprawdza czy bbox1 zachodzi na bbox2 w co najmniej prog * powierzchni bbox1.
    Używane do wykluczenia bloków tekstowych wchodzących w obszar tabeli.
    """
    x0a, y0a, x1a, y1a = bbox1
    x0b, y0b, x1b, y1b = bbox2

    # Część wspólna
    ix0, iy0 = max(x0a, x0b), max(y0a, y0b)
    ix1, iy1 = min(x1a, x1b), min(y1a, y1b)

    if ix0 >= ix1 or iy0 >= iy1:
        return False

    czesc_wspolna = (ix1 - ix0) * (iy1 - iy0)
    pole_bbox1 = max(1.0, (x1a - x0a) * (y1a - y0a))
    return (czesc_wspolna / pole_bbox1) > prog


def _wykryj_rozmiar_tekstu(doc) -> float:
    """
    Wykrywa dominujący rozmiar czcionki tekstu głównego
    (tryb z wagą proporcjonalną do długości tekstu).
    """
    from collections import Counter
    rozmiary: Counter = Counter()

    for strona in doc:
        for blok in strona.get_text("dict")["blocks"]:
            if blok.get("type") != 0:
                continue
            for linia in blok.get("lines", []):
                for span in linia.get("spans", []):
                    r = round(span.get("size", 12), 1)
                    dlugosc = len(span.get("text", "").strip())
                    if dlugosc > 3:
                        rozmiary[r] += dlugosc

    if not rozmiary:
        return 12.0
    return rozmiary.most_common(1)[0][0]


def _czy_element_listy(tekst: str) -> bool:
    """Rozpoznaje element listy punktowanej lub numerowanej."""
    return bool(re.match(r"^\s*[•·▪▸►\-\*]\s+", tekst)) or \
           bool(re.match(r"^\s*\d+[\.\)]\s+", tekst))


# ---------------------------------------------------------------------------
# Ekstrakcja PDF
# ---------------------------------------------------------------------------

def extract_pdf(sciezka: str) -> Tuple[List[DocElement], bool]:
    """
    Ekstrahuje strukturę z pliku PDF.

    Zwraca (elementy, jest_skanem).
    jest_skanem=True gdy PDF nie zawiera warstwy tekstowej — użytkownik
    powinien zostać poinformowany o możliwości użycia OCR.
    """
    import fitz          # pymupdf
    import pdfplumber

    sciezka = str(sciezka)
    elementy: List[DocElement] = []

    fitz_doc = fitz.open(sciezka)
    rozmiar_tekstu = _wykryj_rozmiar_tekstu(fitz_doc)

    # Sprawdź czy PDF jest skanem (brak warstwy tekstowej)
    lacznie_znakow = sum(len(p.get_text("text").strip()) for p in fitz_doc)
    jest_skanem = lacznie_znakow < 80 * max(1, len(fitz_doc))

    with pdfplumber.open(sciezka) as plumber_doc:
        for nr_strony in range(len(fitz_doc)):
            fitz_strona  = fitz_doc[nr_strony]
            plumb_strona = plumber_doc.pages[nr_strony]

            # ----------------------------------------------------------
            # Krok 1: tabele (pdfplumber)
            # ----------------------------------------------------------
            bbox_tabel: List[Tuple] = []
            dane_tabel: List[List[List[str]]] = []

            try:
                for tabela in plumb_strona.find_tables():
                    bbox_tabel.append(tabela.bbox)
                    wiersze = tabela.extract()
                    # Zastąp None pustymi łańcuchami
                    czyste = [
                        [str(k) if k is not None else "" for k in wiersz]
                        for wiersz in (wiersze or [])
                    ]
                    dane_tabel.append(czyste)
            except Exception:
                pass

            # ----------------------------------------------------------
            # Krok 2: obrazy (pymupdf)
            # ----------------------------------------------------------
            elementy_obrazow: List[DocElement] = []
            try:
                for info_img in fitz_strona.get_images(full=True):
                    xref = info_img[0]
                    bazowy = fitz_doc.extract_image(xref)
                    if not bazowy:
                        continue
                    try:
                        bbox_img = fitz_strona.get_image_bbox(info_img)
                        y = float(bbox_img.y0) if bbox_img else 0.0
                    except Exception:
                        y = 0.0

                    elementy_obrazow.append(DocElement(
                        kind="image",
                        image_data=bazowy["image"],
                        image_ext=bazowy.get("ext", "png"),
                        page=nr_strony + 1,
                        y_pos=y,
                    ))
            except Exception:
                pass

            # ----------------------------------------------------------
            # Krok 3: bloki tekstowe (pymupdf)
            # ----------------------------------------------------------
            elementy_tekstowe: List[DocElement] = []
            bloki = fitz_strona.get_text(
                "dict",
                flags=fitz.TEXT_PRESERVE_WHITESPACE
            )["blocks"]

            for blok in bloki:
                if blok.get("type") != 0:
                    continue

                bbox = tuple(blok["bbox"])

                # Pomiń bloki leżące wewnątrz wykrytych tabel
                if any(_zachodzi(bbox, tb) for tb in bbox_tabel):
                    continue

                # Zbierz tekst i właściwości czcionki
                linie_tekstu: List[str] = []
                max_rozmiar = 0.0
                jest_pogrubiony = False

                for linia in blok.get("lines", []):
                    czesci: List[str] = []
                    for span in linia.get("spans", []):
                        t = span.get("text", "")
                        if t.strip():
                            czesci.append(t)
                            r = span.get("size", 12)
                            if r > max_rozmiar:
                                max_rozmiar = r
                            if span.get("flags", 0) & 16:  # flaga bold
                                jest_pogrubiony = True
                    if czesci:
                        linie_tekstu.append("".join(czesci))

                tekst = "\n".join(linie_tekstu).strip()
                if not tekst:
                    continue

                y_pos = float(bbox[1])

                # Klasyfikacja bloku
                jest_naglowkiem = (
                    max_rozmiar > rozmiar_tekstu * 1.15
                    or (jest_pogrubiony and max_rozmiar >= rozmiar_tekstu * 1.05)
                )

                if jest_naglowkiem:
                    # Poziom nagłówka na podstawie stosunku rozmiarów
                    stosunek = max_rozmiar / max(rozmiar_tekstu, 1)
                    if stosunek >= 1.6:
                        poziom = 1
                    elif stosunek >= 1.3:
                        poziom = 2
                    else:
                        poziom = 3
                    elem = DocElement(
                        kind="heading", text=tekst, level=poziom,
                        page=nr_strony + 1, y_pos=y_pos,
                    )
                elif _czy_element_listy(tekst):
                    elem = DocElement(
                        kind="list_item", text=tekst,
                        page=nr_strony + 1, y_pos=y_pos,
                    )
                else:
                    elem = DocElement(
                        kind="paragraph", text=tekst,
                        page=nr_strony + 1, y_pos=y_pos,
                    )

                elementy_tekstowe.append(elem)

            # ----------------------------------------------------------
            # Krok 4: elementy tabel jako DocElement
            # ----------------------------------------------------------
            elementy_tabel: List[DocElement] = []
            for bbox, dane in zip(bbox_tabel, dane_tabel):
                if dane:
                    elementy_tabel.append(DocElement(
                        kind="table",
                        table_data=dane,
                        page=nr_strony + 1,
                        y_pos=float(bbox[1]),
                    ))

            # ----------------------------------------------------------
            # Krok 5: złącz i posortuj wszystkie elementy strony po Y
            # ----------------------------------------------------------
            elementy.append(DocElement(kind="page_sep", page=nr_strony + 1))

            strona_elementy = elementy_tekstowe + elementy_tabel + elementy_obrazow
            strona_elementy.sort(key=lambda e: e.y_pos)
            elementy.extend(strona_elementy)

    fitz_doc.close()
    return elementy, jest_skanem


# ---------------------------------------------------------------------------
# Ekstrakcja TXT
# ---------------------------------------------------------------------------

def extract_txt(sciezka: str) -> List[DocElement]:
    """
    Ekstrahuje strukturę z pliku TXT.

    Heurystyki:
      - Dwa znaki nowej linii = koniec akapitu
      - Linia tylko z CAPS (krótka) = nagłówek
      - Linia zaczynająca się od punktora lub cyfry+kropki = element listy
    """
    with open(sciezka, "r", encoding="utf-8", errors="replace") as f:
        zawartosc = f.read()

    elementy: List[DocElement] = []
    akapity = re.split(r"\n\s*\n", zawartosc)

    for akapit in akapity:
        tekst = akapit.strip()
        if not tekst:
            continue

        linie = tekst.split("\n")

        if (
            len(linie) == 1
            and (tekst.isupper() or (len(tekst) < 80 and tekst.endswith(":")))
        ):
            elementy.append(DocElement(kind="heading", text=tekst, level=2))
        elif _czy_element_listy(tekst):
            elementy.append(DocElement(kind="list_item", text=tekst))
        else:
            elementy.append(DocElement(kind="paragraph", text=tekst))

    return elementy
