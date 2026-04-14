"""
core/translator.py
==================
Backend tłumaczenia oparty na CTranslate2.

Obsługuje trzy modele:
  1. MADLAD-3B  – google/madlad400-3b-mt             (~3 GB VRAM)
  2. NLLB-1.3B  – facebook/nllb-200-distilled-1.3B  (~2.6 GB VRAM)
  3. Helsinki   – Helsinki-NLP/opus-mt-en-zlw        (~300 MB)

Przy pierwszym użyciu modelu:
  - Pobiera tokenizer z HuggingFace i zapisuje lokalnie
  - Konwertuje model do formatu CTranslate2 z kwantyzacją int8
  - Zapisuje w katalogu models/

Kolejne uruchomienia ładują z dysku (szybki start).
"""

import re
import time
from pathlib import Path
from typing import Callable, List, Optional

# Katalog modeli — obok katalogu core/
MODEL_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# Konfiguracja modeli
# ---------------------------------------------------------------------------

MODELE = {
    "madlad-3B": {
        # NLLB-3.3B powodował OOM na RTX 3060 Ti (8 GB).
        # MADLAD-400-3B to model T5 od Google trenowany na 400 językach,
        # int8 zajmuje ~3 GB VRAM — mieści się z zapasem na inferencję.
        # Format wejścia: "<2pl> {tekst}" — prefiks wskazuje język docelowy.
        "hf_nazwa":   "google/madlad400-3b-mt",
        "ct2_katalog": "madlad400-3b-ct2",
        "tok_katalog": "madlad400-3b-tok",
        "typ":        "madlad",
        "tgt_prefix": "<2pl>",   # prefiks języka polskiego
        "etykieta":   "MADLAD-3B  (najlepsza jakość,  ~3 GB VRAM)",
    },
    "nllb-1.3B": {
        "hf_nazwa":   "facebook/nllb-200-distilled-1.3B",
        "ct2_katalog": "nllb-1.3B-ct2",
        "tok_katalog": "nllb-1.3B-tok",
        "typ":        "nllb",
        "src_lang":   "eng_Latn",
        "tgt_lang":   "pol_Latn",
        "etykieta":   "NLLB-1.3B  (szybszy,  ~2.6 GB VRAM)",
    },
    "helsinki": {
        # opus-mt-en-pl nie istnieje; opus-mt-en-zlw to model dla języków
        # zachodniosłowiańskich (PL, CS, HSB) — token >>pol<< wymusza polski
        "hf_nazwa":   "Helsinki-NLP/opus-mt-en-zlw",
        "ct2_katalog": "opus-mt-en-zlw-ct2",
        "tok_katalog": "opus-mt-en-zlw-tok",
        "typ":        "marian",
        "tgt_prefix": ">>pol<<",   # wymusza polski jako język docelowy
        "etykieta":   "Helsinki opus-mt  (~300 MB,  CPU / GPU)",
    },
}


# ---------------------------------------------------------------------------
# Chunking tekstu
# ---------------------------------------------------------------------------

def _podziel_na_zdania(tekst: str) -> List[str]:
    """Prosta segmentacja na zdania po znakach końca zdania."""
    zdania = re.split(r'(?<=[.!?…])\s+', tekst.strip())
    return [z.strip() for z in zdania if z.strip()]


def podziel_na_chunki(tekst: str, tokenizer, max_tokenow: int = 400) -> List[str]:
    """
    Dzieli tekst na kawałki nieprzekraczające max_tokenow.
    Granice cięcia są na końcach zdań (nie w środku).
    """
    if not tekst.strip():
        return []

    zdania = _podziel_na_zdania(tekst)
    if not zdania:
        return [tekst]

    chunki: List[str] = []
    biezacy: List[str] = []
    biezaca_dlugosc = 0

    for zdanie in zdania:
        # Szacowanie tokenów: encode() lub aproksymacja słów × 1.3
        try:
            dlugosc = len(tokenizer.encode(zdanie))
        except Exception:
            dlugosc = max(1, int(len(zdanie.split()) * 1.3))

        if biezaca_dlugosc + dlugosc > max_tokenow and biezacy:
            chunki.append(" ".join(biezacy))
            biezacy = [zdanie]
            biezaca_dlugosc = dlugosc
        else:
            biezacy.append(zdanie)
            biezaca_dlugosc += dlugosc

    if biezacy:
        chunki.append(" ".join(biezacy))

    return chunki if chunki else [tekst]


# ---------------------------------------------------------------------------
# Klasa Translator
# ---------------------------------------------------------------------------

class Translator:
    """
    Zarządza cyklem życia modelu CTranslate2:
    pobieranie → konwersja → ładowanie → inferencja.

    Model jest ładowany raz i przechowywany w pamięci między stronami.
    """

    def __init__(self):
        self._ct2_model = None
        self._tokenizer = None
        self._aktywny_model: Optional[str] = None
        self.urzadzenie = "cpu"

    # ------------------------------------------------------------------
    # Status modelu
    # ------------------------------------------------------------------

    def model_gotowy(self, klucz: str) -> bool:
        """Zwraca True jeśli model jest już pobrany i skonwertowany."""
        cfg = MODELE[klucz]
        return (
            (MODEL_DIR / cfg["ct2_katalog"]).exists()
            and (MODEL_DIR / cfg["tok_katalog"]).exists()
        )

    # ------------------------------------------------------------------
    # Pobieranie i konwersja
    # ------------------------------------------------------------------

    def pobierz_i_konwertuj(
        self,
        klucz: str,
        log: Callable[[str], None] = None,
    ) -> bool:
        """
        Pobiera tokenizer z HuggingFace i konwertuje model do CTranslate2 int8.
        Zwraca True przy sukcesie.
        """
        def _log(msg):
            if log:
                log(msg)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cfg = MODELE[klucz]
        tok_sciezka = MODEL_DIR / cfg["tok_katalog"]
        ct2_sciezka = MODEL_DIR / cfg["ct2_katalog"]

        # --- Tokenizer ---
        if not tok_sciezka.exists():
            _log(f"Pobieranie tokenizera: {cfg['hf_nazwa']} ...")
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(cfg["hf_nazwa"])
                tok.save_pretrained(str(tok_sciezka))
                _log("Tokenizer zapisany.")
            except Exception as e:
                _log(f"BŁĄD pobierania tokenizera: {e}")
                return False

        # --- Model CTranslate2 ---
        if not ct2_sciezka.exists():
            _log(f"Konwersja modelu {cfg['hf_nazwa']} -> CTranslate2 int8")
            _log("(Może potrwać kilka minut i wymaga dużo RAM — jednorazowo)")
            try:
                from ctranslate2.converters import TransformersConverter
                konwerter = TransformersConverter(
                    cfg["hf_nazwa"],
                    low_cpu_mem_usage=True,
                )
                konwerter.convert(str(ct2_sciezka), quantization="int8", force=True)
                _log("Model skonwertowany i zapisany.")
            except Exception as e:
                _log(f"BŁĄD konwersji: {e}")
                # Usuń ewentualnie niepełny katalog
                import shutil
                if ct2_sciezka.exists():
                    shutil.rmtree(ct2_sciezka, ignore_errors=True)
                return False

        return True

    # ------------------------------------------------------------------
    # Ładowanie modelu do pamięci
    # ------------------------------------------------------------------

    def zaladuj(self, klucz: str, log: Callable[[str], None] = None) -> None:
        """
        Ładuje model i tokenizer do pamięci GPU/CPU.
        Jeśli ten sam model jest już załadowany — nic nie robi.
        """
        if self._aktywny_model == klucz and self._ct2_model is not None:
            return

        def _log(msg):
            if log:
                log(msg)

        cfg = MODELE[klucz]

        # Zwolnij poprzedni model i wyczyść VRAM przed załadowaniem nowego
        self._ct2_model = None
        self._tokenizer = None
        if self.urzadzenie == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                wolna = torch.cuda.mem_get_info()[0] // (1024 ** 2)
                _log(f"VRAM wolna przed załadowaniem: {wolna} MB")
            except Exception:
                pass

        _log("Ładowanie tokenizera...")
        # MADLAD używa T5Tokenizer — AutoTokenizer też działa, ale T5Tokenizer
        # obsługuje poprawnie tokeny specjalne jak <2pl>
        if cfg["typ"] == "madlad":
            from transformers import T5Tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained(
                str(MODEL_DIR / cfg["tok_katalog"])
            )
        else:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_DIR / cfg["tok_katalog"])
            )

        _log(f"Ładowanie modelu CTranslate2 na {self.urzadzenie.upper()}...")
        import ctranslate2

        watki_miedzy = 4 if self.urzadzenie == "cpu" else 1

        self._ct2_model = ctranslate2.Translator(
            str(MODEL_DIR / cfg["ct2_katalog"]),
            device=self.urzadzenie,
            inter_threads=watki_miedzy,
            intra_threads=4,
        )
        self._aktywny_model = klucz

        if self.urzadzenie == "cuda":
            try:
                import torch
                uzyta = torch.cuda.memory_allocated() // (1024 ** 2)
                _log(f"Model załadowany. VRAM zajęta przez model: {uzyta} MB")
            except Exception:
                _log("Model załadowany.")
        else:
            _log("Model załadowany.")

    # ------------------------------------------------------------------
    # Tłumaczenie
    # ------------------------------------------------------------------

    def tlumacz_partie(
        self,
        teksty: List[str],
        klucz: str,
    ) -> List[str]:
        """
        Tłumaczy jedną partię tekstów (batch).
        Zwraca przetłumaczone teksty w tej samej kolejności.
        """
        if not teksty:
            return []

        cfg = MODELE[klucz]
        tokenizer = self._tokenizer
        model = self._ct2_model
        typ = cfg["typ"]

        # Tokenizacja — przygotuj wejście zależnie od architektury modelu
        if typ == "nllb":
            tokenizer.src_lang = cfg["src_lang"]
            wejscie = teksty
        else:
            # Marian (Helsinki) i MADLAD: prefiks języka w tekście źródłowym
            prefiks = cfg.get("tgt_prefix", "")
            wejscie = [f"{prefiks} {t}" if prefiks else t for t in teksty]

        zakodowane = tokenizer(
            wejscie,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=512,
        )

        zrodlowe_tokeny = [
            tokenizer.convert_ids_to_tokens(ids)
            for ids in zakodowane["input_ids"]
        ]

        # Inferencja
        if typ == "nllb":
            prefiks_ct2 = [[cfg["tgt_lang"]]] * len(teksty)
            wyniki = model.translate_batch(
                zrodlowe_tokeny,
                target_prefix=prefiks_ct2,
                max_decoding_length=512,
                beam_size=4,
            )
            przetlumaczone = []
            for wynik in wyniki:
                tokeny = wynik.hypotheses[0][1:]  # pomiń token języka docelowego
                tekst = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(tokeny),
                    skip_special_tokens=True,
                )
                przetlumaczone.append(tekst)
        else:
            # Helsinki/Marian i MADLAD — encoder-decoder bez target_prefix
            wyniki = model.translate_batch(
                zrodlowe_tokeny,
                max_decoding_length=512,
                beam_size=4,
            )
            przetlumaczone = []
            for wynik in wyniki:
                tokeny = wynik.hypotheses[0]
                tekst = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(tokeny),
                    skip_special_tokens=True,
                )
                przetlumaczone.append(tekst)

        return przetlumaczone

    # ------------------------------------------------------------------
    # Pełne tłumaczenie z postępem
    # ------------------------------------------------------------------

    def _dobierz_batch_size(self, klucz: str, zadany: int) -> int:
        """
        Dobiera bezpieczny batch_size na podstawie wolnej VRAM.
        Chroni przed OOM przy dużych modelach (NLLB-3.3B zostawia mało wolnego).
        """
        if self.urzadzenie != "cuda":
            return zadany
        try:
            import torch
            wolna_mb = torch.cuda.mem_get_info()[0] // (1024 ** 2)
            if wolna_mb < 500:
                return 1
            elif wolna_mb < 1500:
                return 2
            elif wolna_mb < 3000:
                return 4
            return zadany
        except Exception:
            return zadany

    def tlumacz_wszystko(
        self,
        teksty: List[str],
        klucz: str,
        batch_size: int = 8,
        postep: Callable[[int, int, Optional[float]], None] = None,
        anuluj=None,
    ) -> List[str]:
        """
        Tłumaczy całą listę tekstów partiami.

        postep(przetlumaczono, lacznie, eta_sekundy) — callback postępu
        anuluj — threading.Event; ustawienie przerywa tłumaczenie
        """
        if not teksty:
            return []

        # Dobierz bezpieczny batch_size na podstawie wolnej VRAM
        batch_size = self._dobierz_batch_size(klucz, batch_size)

        wyniki: List[str] = []
        lacznie = len(teksty)
        start = time.time()

        for i in range(0, lacznie, batch_size):
            if anuluj and anuluj.is_set():
                raise InterruptedError("Tłumaczenie anulowane.")

            partia = teksty[i: i + batch_size]
            wyniki.extend(self.tlumacz_partie(partia, klucz))

            przetlumaczono = min(i + batch_size, lacznie)
            uplynelo = time.time() - start
            if przetlumaczono > 0 and przetlumaczono < lacznie:
                eta = (uplynelo / przetlumaczono) * (lacznie - przetlumaczono)
            else:
                eta = 0.0

            if postep:
                postep(przetlumaczono, lacznie, eta)

        return wyniki

    def zwolnij(self) -> None:
        """Zwalnia model z pamięci GPU/CPU."""
        self._ct2_model = None
        self._tokenizer = None
        self._aktywny_model = None
