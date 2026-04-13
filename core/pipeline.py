"""
core/pipeline.py
================
Orkiestruje cały proces tłumaczenia.

Uruchamiana w wątku tła — komunikuje się z UI przez kolejkę (queue.Queue).

Typy wiadomości wysyłanych do kolejki:
  ("log",      tekst)              – wpis do logu
  ("postep",   przetl, lacznie, eta_s)  – aktualizacja paska postępu
  ("ostrzezenie_ocr",)             – PDF jest skanem, brak tekstu
  ("gotowe",   sciezka_wyjscia)    – zakończono pomyślnie
  ("blad",     komunikat)          – zakończono z błędem
  ("anulowano",)                   – użytkownik anulował
"""

import queue
import threading
import time
from pathlib import Path
from typing import Optional

from .extractor import extract_pdf, extract_txt, DocElement
from .translator import Translator, MODELE, podziel_na_chunki
from .builder import zbuduj_docx, zbuduj_txt


def uruchom_tlumaczenie(
    sciezka_wejscia: str,
    sciezka_wyjscia: str,
    klucz_modelu: str,
    format_wyjscia: str,       # "docx" lub "txt"
    urzadzenie: str,            # "cuda" lub "cpu"
    kolejka: queue.Queue,
    anuluj: threading.Event,
    batch_size: int = 8,
) -> None:
    """
    Główna funkcja pipeline'u — do uruchomienia jako wątek tła.

    Wszystkie wyniki i postęp raportowane przez `kolejka`.
    """

    def log(msg: str):
        kolejka.put(("log", msg))

    def postep(przetl: int, lacznie: int, eta: Optional[float]):
        kolejka.put(("postep", przetl, lacznie, eta))

    translator = Translator()
    translator.urzadzenie = urzadzenie

    try:
        # ------------------------------------------------------------------
        # Krok 1: Pobierz i skonwertuj model (jednorazowo)
        # ------------------------------------------------------------------
        if not translator.model_gotowy(klucz_modelu):
            log(f"Pierwszy start: pobieranie modelu {MODELE[klucz_modelu]['hf_nazwa']}...")
            log("(Jednorazowe pobieranie — może potrwać kilka minut)")
            ok = translator.pobierz_i_konwertuj(klucz_modelu, log=log)
            if not ok:
                kolejka.put(("blad", "Nie udało się pobrać/skonwertować modelu."))
                return

        if anuluj.is_set():
            kolejka.put(("anulowano",))
            return

        # ------------------------------------------------------------------
        # Krok 2: Załaduj model
        # ------------------------------------------------------------------
        translator.zaladuj(klucz_modelu, log=log)

        if anuluj.is_set():
            kolejka.put(("anulowano",))
            return

        # ------------------------------------------------------------------
        # Krok 3: Ekstrahuj dokument
        # ------------------------------------------------------------------
        sufiks = Path(sciezka_wejscia).suffix.lower()

        if sufiks == ".pdf":
            log("Ekstrakcja struktury PDF...")
            elementy, jest_skanem = extract_pdf(sciezka_wejscia)
            if jest_skanem:
                log("OSTRZEŻENIE: PDF wydaje się być skanem (brak warstwy tekstowej).")
                kolejka.put(("ostrzezenie_ocr",))
        else:
            log("Wczytywanie pliku TXT...")
            elementy = extract_txt(sciezka_wejscia)
            jest_skanem = False

        log(f"Wykryto elementów dokumentu: {len(elementy)}")

        if anuluj.is_set():
            kolejka.put(("anulowano",))
            return

        # ------------------------------------------------------------------
        # Krok 4: Zbierz wszystkie teksty do tłumaczenia
        # ------------------------------------------------------------------
        log("Przygotowanie jednostek do tłumaczenia...")

        # Lista krotek: (indeks_elementu, klucz_podmowy, tekst_do_tlumaczenia)
        # klucz_podmowy = ("tekst", nr_chunka) | ("komorka", wiersz, kolumna)
        do_tlumaczenia = []

        for i, elem in enumerate(elementy):
            if elem.kind in ("paragraph", "heading", "list_item"):
                if elem.text.strip():
                    chunki = podziel_na_chunki(
                        elem.text,
                        translator._tokenizer,
                        max_tokenow=400,
                    )
                    for nr, chunk in enumerate(chunki):
                        do_tlumaczenia.append((i, ("tekst", nr), chunk))

            elif elem.kind == "table" and elem.table_data:
                # Zainicjuj przetłumaczone komórki kopią oryginału
                elem.translated_cells = [wiersz[:] for wiersz in elem.table_data]
                for r, wiersz in enumerate(elem.table_data):
                    for c, komorka in enumerate(wiersz):
                        if komorka and str(komorka).strip():
                            do_tlumaczenia.append((i, ("komorka", r, c), str(komorka)))

        lacznie = len(do_tlumaczenia)
        log(f"Jednostek do przetłumaczenia: {lacznie}")

        if lacznie == 0:
            log("Brak tekstu do tłumaczenia — dokument może być pusty lub skanem.")
            _zapisz_wynik(elementy, sciezka_wyjscia, format_wyjscia, log)
            kolejka.put(("gotowe", sciezka_wyjscia))
            return

        # ------------------------------------------------------------------
        # Krok 5: Tłumacz partiami
        # ------------------------------------------------------------------
        log("Tłumaczenie...")

        wszystkie_teksty = [t for _, _, t in do_tlumaczenia]
        przetlumaczone_teksty = []
        start_czasu = time.time()

        for batch_start in range(0, lacznie, batch_size):
            if anuluj.is_set():
                kolejka.put(("anulowano",))
                return

            partia = wszystkie_teksty[batch_start: batch_start + batch_size]
            wyniki = translator.tlumacz_partie(partia, klucz_modelu)
            przetlumaczone_teksty.extend(wyniki)

            # Postęp + ETA
            przetl = min(batch_start + batch_size, lacznie)
            uplynelo = time.time() - start_czasu
            if przetl > 0 and przetl < lacznie:
                eta = (uplynelo / przetl) * (lacznie - przetl)
            else:
                eta = 0.0
            postep(przetl, lacznie, eta)

        # ------------------------------------------------------------------
        # Krok 6: Mapowanie wyników z powrotem na DocElement
        # ------------------------------------------------------------------
        # Bufor dla chunków akapitów: (elem_idx) → lista (nr_chunka, tekst)
        bufor_chunkow: dict = {}

        for (elem_idx, klucz_pod, _), przetl in zip(do_tlumaczenia, przetlumaczone_teksty):
            rodzaj = klucz_pod[0]

            if rodzaj == "tekst":
                nr = klucz_pod[1]
                bufor_chunkow.setdefault(elem_idx, []).append((nr, przetl))

            elif rodzaj == "komorka":
                _, r, c = klucz_pod
                elementy[elem_idx].translated_cells[r][c] = przetl

        # Złącz chunki w całe tłumaczenia akapitów
        for elem_idx, chunki in bufor_chunkow.items():
            chunki.sort(key=lambda x: x[0])
            elementy[elem_idx].translated_text = " ".join(t for _, t in chunki)

        # ------------------------------------------------------------------
        # Krok 7: Zbuduj plik wyjściowy
        # ------------------------------------------------------------------
        _zapisz_wynik(elementy, sciezka_wyjscia, format_wyjscia, log)
        log("Tłumaczenie zakończone pomyślnie!")
        kolejka.put(("gotowe", sciezka_wyjscia))

    except InterruptedError:
        kolejka.put(("anulowano",))
    except Exception as e:
        import traceback
        kolejka.put(("blad", f"{e}\n\n{traceback.format_exc()}"))


def _zapisz_wynik(
    elementy,
    sciezka_wyjscia: str,
    format_wyjscia: str,
    log: callable,
) -> None:
    """Zapisuje przetłumaczony dokument do pliku DOCX lub TXT."""
    log(f"Zapis do pliku: {sciezka_wyjscia} ...")
    if format_wyjscia == "docx":
        zbuduj_docx(elementy, sciezka_wyjscia)
    else:
        zbuduj_txt(elementy, sciezka_wyjscia)
