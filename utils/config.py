"""
utils/config.py
===============
Zapis i odczyt ustawień użytkownika z pliku config.json.
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

DOMYSLNE = {
    "model":           "nllb-1.3B",
    "format_wyjscia":  "docx",
    "ostatni_katalog_wejscia":  "",
    "ostatni_katalog_wyjscia":  "",
    "batch_size":      8,
}


def wczytaj() -> dict:
    """Wczytuje konfigurację. Jeśli plik nie istnieje — zwraca domyślną."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                zapisane = json.load(f)
            # Scalamy z domyślnymi — nowe klucze są automatycznie dodawane
            return {**DOMYSLNE, **zapisane}
        except Exception:
            pass
    return DOMYSLNE.copy()


def zapisz(cfg: dict) -> None:
    """Zapisuje konfigurację do pliku JSON."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # błąd zapisu nie powinien przerywać pracy aplikacji
