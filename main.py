"""
main.py
=======
Punkt wejścia aplikacji TranslatorPDF.
Uruchom: python main.py
"""

import sys
from pathlib import Path

# Dodaj katalog projektu do ścieżki Pythona
sys.path.insert(0, str(Path(__file__).parent))

from ui.app import TranslatorApp

if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()
