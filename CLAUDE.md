# TranslatorPDF – kontekst projektu dla Claude Code

## Co to jest
Desktopowa aplikacja do tłumaczenia PDF/TXT z angielskiego na polski.
Pipeline: ekstrakcja (PyMuPDF/pdfplumber) → tłumaczenie (CTranslate2 / transformers) → zapis DOCX.
Stos: Python 3.12 + CustomTkinter + PyTorch/CUDA.

## Architektura
- `main.py` – punkt wejścia
- `core/extractor.py` – ekstrakcja PDF/TXT do listy `DocElement` (typ: akapit, nagłówek, tabela, obraz)
- `core/translator.py` – ładowanie modelu + tłumaczenie (Allegro BiDi EN↔PL przez marian_hf)
- `core/builder.py` – budowanie wyjściowego DOCX z python-docx
- `core/pipeline.py` – orkiestracja: extractor → translator → builder, raportowanie postępu do UI
- `ui/app.py` – interfejs CustomTkinter (wybór pliku, pasek postępu, log)
- `utils/config.py` – wczytywanie/zapis `config.json`
- `utils/cuda_check.py` – detekcja GPU
- `models/` – lokalny cache modeli (pobierane z Hugging Face przy pierwszym użyciu)

## Środowisko
- Wirtualne środowisko: `.venv/` (Python 3.12)
- Uruchomienie: `uruchom.bat` lub `.venv\Scripts\activate && python main.py`

## Krytyczne pułapki
- **Python 3.12 obowiązkowy** — 3.13+ nie ma kół CUDA dla PyTorch
- **PyTorch z CUDA**: `pip install torch --index-url https://download.pytorch.org/whl/cu121` — NIE z PyPI
- **Model EN→PL**: Historia iteracji — Helsinki → OPUS-MT → NLLB → **Allegro BiDi (marian_hf)** (aktualny)
- **`opus-mt-en-pl` nie istnieje** — jeśli wrócisz do OPUS-MT, używaj `opus-mt-en-sla` z prefiksem `>>pl<<`
- **CTranslate2**: Wymaga konwersji modelu (`ct2-opus-mt-train-model`); aktualnie tłumaczenie natywnie przez marian_hf
- **Tabele w PDF jako obrazy**: pdfplumber nie wykryje tabel zeskanowanych/obrazkowych

## Model tłumaczenia
Aktualny: Allegro BiDi EN↔PL (marian_hf), obsługiwany natywnie przez transformers.
Konwersja CTranslate2 ominięta — pipeline działa bezpośrednio przez `AutoModelForSeq2SeqLM`.

## Stan
Gotowy, aktywnie rozwijany. Ostatnie prace: obsługa BiDi natywnie, iteracje na modelach.
