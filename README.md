# TranslatorPDF

Desktopowa aplikacja do tłumaczenia dokumentów PDF i TXT z angielskiego na polski.
Działa w pełni lokalnie — po jednorazowym pobraniu modeli nie wymaga połączenia z internetem.

---

## Co robi

1. **Ekstrakcja** — odczytuje tekst z pliku PDF lub TXT z zachowaniem struktury (akapity, nagłówki, tabele)
2. **Tłumaczenie** — tłumaczy tekst na język polski przez model Allegro BiDi EN↔PL
3. **Eksport** — zapisuje przetłumaczony dokument jako `.docx`

---

## Wymagania systemowe

| Składnik | Minimum | Zalecane |
|---|---|---|
| System operacyjny | Windows 10/11 | Windows 11 |
| Python | 3.12 | **3.12** (3.13+ nie obsługuje CUDA) |
| RAM | 8 GB | 16 GB |
| VRAM (GPU NVIDIA) | opcjonalne | 4 GB+ (znacznie szybsze) |
| Dysk | 10 GB | 15 GB |

> GPU nie jest wymagane — aplikacja działa na CPU, ale tłumaczenie będzie wolniejsze.

---

## Instalacja

### Krok 1 — Python 3.12

Pobierz Python 3.12 ze strony [python.org](https://www.python.org/downloads/).
Podczas instalacji zaznacz **"Add Python to PATH"**.

Sprawdź:
```
py -3.12 --version
```

### Krok 2 — Środowisko wirtualne i zależności

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Krok 3 — PyTorch z CUDA (opcjonalne, dla GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Bez tego kroku aplikacja użyje CPU.

---

## Uruchomienie

### Windows

```bash
uruchom.bat
```

### Bezpośrednio (z aktywnym środowiskiem)

```bash
.venv\Scripts\activate
python main.py
```

---

## Pierwsze użycie

1. Uruchom aplikację
2. Kliknij **"Wybierz plik"** i wskaż plik `.pdf` lub `.txt`
3. Kliknij **"Tłumacz"**
4. Plik wynikowy zostanie zapisany obok oryginału z sufiksem `_PL.docx`

---

## Struktura projektu

```
TranslatorPDF/
├── main.py                  # Punkt wejścia aplikacji
├── requirements.txt         # Zależności Python
├── uruchom.bat              # Skrypt uruchomienia
├── config.json              # Ustawienia (ścieżki modeli, GPU)
├── models/                  # Lokalne modele tłumaczenia (po pobraniu)
├── core/
│   ├── extractor.py         # Ekstrakcja PDF/TXT → lista elementów (DocElement)
│   ├── builder.py           # Budowanie wyjściowego DOCX
│   ├── translator.py        # Ładowanie i obsługa modelu
│   └── pipeline.py          # Orkiestracja całego procesu
├── ui/
│   └── app.py               # Interfejs CustomTkinter
└── utils/
    ├── config.py            # Wczytywanie/zapisywanie config.json
    └── cuda_check.py        # Detekcja dostępności GPU
```

---

## Technologie

| Biblioteka | Zastosowanie |
|---|---|
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | Szybkie wnioskowanie modeli tłumaczenia na GPU/CPU |
| [transformers](https://huggingface.co/docs/transformers) | Ładowanie modeli Hugging Face |
| [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) | Ekstrakcja tekstu z PDF z pozycją |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | Detekcja i ekstrakcja tabel z PDF |
| [python-docx](https://python-docx.readthedocs.io) | Generowanie dokumentów DOCX |
| [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) | Nowoczesny interfejs graficzny |

---

## Rozwiązywanie problemów

**Aplikacja nie wykrywa GPU:**
- Sprawdź: `nvidia-smi` w terminalu — powinno pokazać kartę graficzną
- Upewnij się że PyTorch zainstalowano z indeksem CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**Błąd przy ekstrakcji tabel z PDF:**
- Niektóre PDF-y mają tabele jako obrazy — ekstrakcja tekstu nie zadziała w takim przypadku

**Model nie jest dostępny / błąd pobierania:**
- Sprawdź połączenie internetowe przy pierwszym uruchomieniu (model pobierany z Hugging Face)
- Modele zapisywane są lokalnie w `models/` po pierwszym pobraniu

---

## Licencja

MIT
