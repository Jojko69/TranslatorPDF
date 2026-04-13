"""
ui/app.py
=========
Główne okno aplikacji TranslatorPDF zbudowane w CustomTkinter.

Układ:
  ┌──────────────────────────────────────────────┐
  │  Pliki wejścia / wyjścia                     │
  │  Model | Format wyjścia                       │
  │  [ TŁUMACZ ]  [ ANULUJ ]                     │
  │  Pasek postępu + ETA                          │
  │  Log zdarzeń                                  │
  └──────────────────────────────────────────────┘
"""

import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core.translator import MODELE
from core.pipeline import uruchom_tlumaczenie
from utils.config import wczytaj as wczytaj_cfg, zapisz as zapisz_cfg
from utils.cuda_check import check_cuda


# Stałe kolorów (krotki ciemny/jasny dla CTk)
TEXT_PRIMARY   = ("#4c4f69", "#cdd6f4")
TEXT_SECONDARY = ("#6c6f85", "#a6adc8")
TEXT_MUTED     = ("#9ca0b0", "#6c7086")
TEXT_VALUE     = ("#1e66f5", "#89b4fa")
SUCCESS_COLOR  = ("#40a02b", "#a6e3a1")
WARNING_COLOR  = ("#df8e1d", "#f9e2af")
ERROR_COLOR    = ("#d20f39", "#f38ba8")
CARD_BG        = ("#e6e9f0", "#313244")
ACCENT         = ("#1e66f5", "#89b4fa")


class TranslatorApp(ctk.CTk):
    """Główne okno aplikacji."""

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._cfg = wczytaj_cfg()
        self._cuda_dostepna, self._urzadzenie_opis = check_cuda()
        self._urzadzenie = "cuda" if self._cuda_dostepna else "cpu"

        self._kolejka: queue.Queue = queue.Queue()
        self._watek_tlumaczenia: threading.Thread | None = None
        self._anuluj = threading.Event()
        self._czas_startu: float = 0.0
        self._w_trakcie = False

        self._buduj_okno()
        self._buduj_ui()
        self._wczytaj_ustawienia()

        # Wyświetl status CUDA po uruchomieniu
        if self._cuda_dostepna:
            self._log(f"GPU wykryte: {self._urzadzenie_opis}")
        else:
            self._log(f"OSTRZEŻENIE: {self._urzadzenie_opis} — tłumaczenie będzie wolniejsze.")

    # ------------------------------------------------------------------
    # Konfiguracja okna
    # ------------------------------------------------------------------

    def _buduj_okno(self):
        self.title("TranslatorPDF  |  EN → PL")
        self.geometry("820x680")
        self.minsize(700, 560)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.configure(fg_color=("#eff1f5", "#1e1e2e"))

        # Wyśrodkuj
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = (sw - 820) // 2
        y = max(0, (sh - 680) // 2 - 30)
        self.geometry(f"820x680+{x}+{y}")

    # ------------------------------------------------------------------
    # Budowanie UI
    # ------------------------------------------------------------------

    def _buduj_ui(self):
        glowna = ctk.CTkScrollableFrame(
            self,
            fg_color=("#eff1f5", "#1e1e2e"),
            scrollbar_button_color=CARD_BG,
        )
        glowna.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        glowna.grid_columnconfigure(0, weight=1)

        # Logo
        self._buduj_logo(glowna)
        # Sekcja plików
        self._buduj_sekcje_plikow(glowna)
        # Opcje (model + format)
        self._buduj_sekcje_opcji(glowna)
        # Przyciski akcji
        self._buduj_przyciski(glowna)
        # Postęp
        self._buduj_pasek_postepu(glowna)
        # Log
        self._buduj_log(glowna)

    def _buduj_logo(self, rodzic):
        ramka = ctk.CTkFrame(rodzic, fg_color="transparent", height=56)
        ramka.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 4))
        ramka.grid_columnconfigure(1, weight=1)
        ramka.grid_propagate(False)

        ctk.CTkLabel(
            ramka,
            text="TranslatorPDF",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=ACCENT,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            ramka,
            text="  EN → PL  |  v1.0",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
        ).grid(row=0, column=1, sticky="sw", pady=(0, 4))

        # Status urządzenia po prawej
        self._etykieta_urzadzenia = ctk.CTkLabel(
            ramka,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=SUCCESS_COLOR if self._cuda_dostepna else WARNING_COLOR,
            anchor="e",
        )
        self._etykieta_urzadzenia.grid(row=0, column=2, sticky="e")
        self._etykieta_urzadzenia.configure(
            text=f"{'GPU ✓' if self._cuda_dostepna else 'CPU tylko'}"
        )

    def _buduj_sekcje_plikow(self, rodzic):
        karta = ctk.CTkFrame(rodzic, fg_color=CARD_BG, corner_radius=12)
        karta.grid(row=1, column=0, sticky="ew", padx=24, pady=(8, 4))
        karta.grid_columnconfigure(1, weight=1)

        # --- Plik wejściowy ---
        ctk.CTkLabel(
            karta,
            text="Plik wejściowy:",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
            width=130,
            anchor="e",
        ).grid(row=0, column=0, padx=(16, 8), pady=(14, 4), sticky="e")

        self._wejscie_var = ctk.StringVar()
        self._pole_wejscia = ctk.CTkEntry(
            karta,
            textvariable=self._wejscie_var,
            placeholder_text="Wybierz plik PDF lub TXT...",
            font=ctk.CTkFont(size=12),
            height=36,
        )
        self._pole_wejscia.grid(row=0, column=1, padx=(0, 8), pady=(14, 4), sticky="ew")

        ctk.CTkButton(
            karta,
            text="Wybierz",
            width=90,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            command=self._wybierz_plik_wejscia,
        ).grid(row=0, column=2, padx=(0, 16), pady=(14, 4))

        # --- Plik wyjściowy ---
        ctk.CTkLabel(
            karta,
            text="Zapisz wynik jako:",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
            width=130,
            anchor="e",
        ).grid(row=1, column=0, padx=(16, 8), pady=(4, 14), sticky="e")

        self._wyjscie_var = ctk.StringVar()
        self._pole_wyjscia = ctk.CTkEntry(
            karta,
            textvariable=self._wyjscie_var,
            placeholder_text="Wskaż lokalizację pliku wyjściowego...",
            font=ctk.CTkFont(size=12),
            height=36,
        )
        self._pole_wyjscia.grid(row=1, column=1, padx=(0, 8), pady=(4, 14), sticky="ew")

        ctk.CTkButton(
            karta,
            text="Wybierz",
            width=90,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            command=self._wybierz_plik_wyjscia,
        ).grid(row=1, column=2, padx=(0, 16), pady=(4, 14))

    def _buduj_sekcje_opcji(self, rodzic):
        karta = ctk.CTkFrame(rodzic, fg_color=CARD_BG, corner_radius=12)
        karta.grid(row=2, column=0, sticky="ew", padx=24, pady=4)
        karta.grid_columnconfigure(1, weight=1)

        # --- Wybór modelu ---
        ctk.CTkLabel(
            karta,
            text="Model:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_SECONDARY,
        ).grid(row=0, column=0, padx=(16, 12), pady=(14, 4), sticky="w")

        self._model_var = ctk.StringVar(value=self._cfg.get("model", "nllb-1.3B"))
        ramka_modeli = ctk.CTkFrame(karta, fg_color="transparent")
        ramka_modeli.grid(row=0, column=1, columnspan=2, sticky="w", padx=(0, 16), pady=(14, 4))

        for klucz, dane in MODELE.items():
            ctk.CTkRadioButton(
                ramka_modeli,
                text=dane["etykieta"],
                variable=self._model_var,
                value=klucz,
                font=ctk.CTkFont(size=12),
                text_color=TEXT_PRIMARY,
            ).pack(side="left", padx=(0, 20))

        # --- Format wyjścia ---
        ctk.CTkLabel(
            karta,
            text="Format:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, padx=(16, 12), pady=(4, 14), sticky="w")

        self._format_var = ctk.StringVar(value=self._cfg.get("format_wyjscia", "docx"))
        ramka_formatow = ctk.CTkFrame(karta, fg_color="transparent")
        ramka_formatow.grid(row=1, column=1, columnspan=2, sticky="w", padx=(0, 16), pady=(4, 14))

        ctk.CTkRadioButton(
            ramka_formatow,
            text="DOCX  (zachowuje obrazy i tabele)",
            variable=self._format_var,
            value="docx",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_PRIMARY,
            command=self._na_zmiane_formatu,
        ).pack(side="left", padx=(0, 20))

        ctk.CTkRadioButton(
            ramka_formatow,
            text="TXT  (tylko tekst)",
            variable=self._format_var,
            value="txt",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_PRIMARY,
            command=self._na_zmiane_formatu,
        ).pack(side="left")

    def _buduj_przyciski(self, rodzic):
        ramka = ctk.CTkFrame(rodzic, fg_color="transparent")
        ramka.grid(row=3, column=0, sticky="ew", padx=24, pady=(8, 4))

        self._btn_tlumacz = ctk.CTkButton(
            ramka,
            text="  TŁUMACZ",
            width=160,
            height=44,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_tlumaczenia,
        )
        self._btn_tlumacz.pack(side="left")

        self._btn_anuluj = ctk.CTkButton(
            ramka,
            text="  Anuluj",
            width=120,
            height=44,
            corner_radius=10,
            font=ctk.CTkFont(size=13),
            fg_color=("#ccd0da", "#45475a"),
            text_color=TEXT_PRIMARY,
            hover_color=("#bcc0ce", "#585b70"),
            state="disabled",
            command=self._anuluj_tlumaczenie,
        )
        self._btn_anuluj.pack(side="left", padx=(12, 0))

    def _buduj_pasek_postepu(self, rodzic):
        ramka = ctk.CTkFrame(rodzic, fg_color="transparent")
        ramka.grid(row=4, column=0, sticky="ew", padx=24, pady=(8, 0))
        ramka.grid_columnconfigure(0, weight=1)

        self._pasek = ctk.CTkProgressBar(ramka, height=12, corner_radius=6)
        self._pasek.set(0)
        self._pasek.grid(row=0, column=0, sticky="ew")

        self._etykieta_postepu = ctk.CTkLabel(
            ramka,
            text="Gotowy.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self._etykieta_postepu.grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _buduj_log(self, rodzic):
        ctk.CTkLabel(
            rodzic,
            text="  Log:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
            anchor="w",
        ).grid(row=5, column=0, sticky="w", padx=24, pady=(12, 2))

        self._obszar_logu = ctk.CTkTextbox(
            rodzic,
            height=200,
            font=ctk.CTkFont(family="Courier New", size=11),
            fg_color=("#dce0e8", "#181825"),
            text_color=TEXT_SECONDARY,
            corner_radius=8,
            wrap="word",
            state="disabled",
        )
        self._obszar_logu.grid(row=6, column=0, sticky="ew", padx=24, pady=(0, 20))

    # ------------------------------------------------------------------
    # Dialogi wyboru pliku
    # ------------------------------------------------------------------

    def _wybierz_plik_wejscia(self):
        ostatni_katalog = self._cfg.get("ostatni_katalog_wejscia", "")
        sciezka = filedialog.askopenfilename(
            title="Wybierz plik do tłumaczenia",
            initialdir=ostatni_katalog or None,
            filetypes=[
                ("Obsługiwane formaty", "*.pdf *.txt"),
                ("PDF", "*.pdf"),
                ("Tekst", "*.txt"),
            ],
        )
        if sciezka:
            self._wejscie_var.set(sciezka)
            self._cfg["ostatni_katalog_wejscia"] = str(Path(sciezka).parent)
            # Zaproponuj automatyczną nazwę pliku wyjściowego
            self._zaproponuj_plik_wyjscia(sciezka)

    def _zaproponuj_plik_wyjscia(self, sciezka_wejscia: str):
        """Uzupełnia pole wyjściowe na podstawie wybranego pliku."""
        if self._wyjscie_var.get():
            return  # użytkownik już wybrał — nie nadpisuj
        p = Path(sciezka_wejscia)
        ext = "." + self._format_var.get()
        propozycja = str(p.parent / (p.stem + "_PL" + ext))
        self._wyjscie_var.set(propozycja)

    def _wybierz_plik_wyjscia(self):
        fmt = self._format_var.get()
        ostatni_katalog = self._cfg.get("ostatni_katalog_wyjscia", "")

        if fmt == "docx":
            typy = [("Word DOCX", "*.docx")]
            domyslne_ext = ".docx"
        else:
            typy = [("Plik tekstowy", "*.txt")]
            domyslne_ext = ".txt"

        sciezka = filedialog.asksaveasfilename(
            title="Zapisz przetłumaczony plik",
            initialdir=ostatni_katalog or None,
            defaultextension=domyslne_ext,
            filetypes=typy,
        )
        if sciezka:
            self._wyjscie_var.set(sciezka)
            self._cfg["ostatni_katalog_wyjscia"] = str(Path(sciezka).parent)

    def _na_zmiane_formatu(self):
        """Aktualizuje rozszerzenie pliku wyjściowego po zmianie formatu."""
        wyjscie = self._wyjscie_var.get()
        if not wyjscie:
            return
        p = Path(wyjscie)
        nowe_ext = "." + self._format_var.get()
        self._wyjscie_var.set(str(p.with_suffix(nowe_ext)))

    # ------------------------------------------------------------------
    # Wczytywanie ustawień
    # ------------------------------------------------------------------

    def _wczytaj_ustawienia(self):
        self._model_var.set(self._cfg.get("model", "nllb-1.3B"))
        self._format_var.set(self._cfg.get("format_wyjscia", "docx"))

    # ------------------------------------------------------------------
    # Logowanie
    # ------------------------------------------------------------------

    def _log(self, tekst: str):
        """Dopisuje linię do obszaru logu z znacznikiem czasu."""
        from datetime import datetime
        znacznik = datetime.now().strftime("%H:%M:%S")
        linia = f"[{znacznik}] {tekst}\n"

        self._obszar_logu.configure(state="normal")
        self._obszar_logu.insert("end", linia)
        self._obszar_logu.see("end")
        self._obszar_logu.configure(state="disabled")

    # ------------------------------------------------------------------
    # Tłumaczenie — start i anulowanie
    # ------------------------------------------------------------------

    def _start_tlumaczenia(self):
        wejscie = self._wejscie_var.get().strip()
        wyjscie = self._wyjscie_var.get().strip()

        if not wejscie:
            messagebox.showwarning("Brak pliku", "Wybierz plik wejściowy (PDF lub TXT).")
            return
        if not wyjscie:
            messagebox.showwarning("Brak ścieżki", "Wskaż lokalizację pliku wyjściowego.")
            return
        if not Path(wejscie).exists():
            messagebox.showerror("Błąd", f"Plik nie istnieje:\n{wejscie}")
            return

        # Zapisz ustawienia
        self._cfg["model"]           = self._model_var.get()
        self._cfg["format_wyjscia"]  = self._format_var.get()
        zapisz_cfg(self._cfg)

        # Resetuj UI
        self._pasek.set(0)
        self._etykieta_postepu.configure(text="Uruchamianie...", text_color=TEXT_MUTED)
        self._btn_tlumacz.configure(state="disabled")
        self._btn_anuluj.configure(state="normal")
        self._w_trakcie = True
        self._anuluj.clear()
        self._czas_startu = time.time()

        # Uruchom wątek tła
        self._watek_tlumaczenia = threading.Thread(
            target=uruchom_tlumaczenie,
            kwargs=dict(
                sciezka_wejscia=wejscie,
                sciezka_wyjscia=wyjscie,
                klucz_modelu=self._model_var.get(),
                format_wyjscia=self._format_var.get(),
                urzadzenie=self._urzadzenie,
                kolejka=self._kolejka,
                anuluj=self._anuluj,
                batch_size=self._cfg.get("batch_size", 8),
            ),
            daemon=True,
        )
        self._watek_tlumaczenia.start()
        self._sprawdzaj_kolejke()

    def _anuluj_tlumaczenie(self):
        self._anuluj.set()
        self._btn_anuluj.configure(state="disabled")
        self._log("Wysłano sygnał anulowania — czekam na zakończenie bieżącej partii...")

    # ------------------------------------------------------------------
    # Polling kolejki (wątek UI)
    # ------------------------------------------------------------------

    def _sprawdzaj_kolejke(self):
        """Sprawdza kolejkę co 100 ms i aktualizuje UI."""
        try:
            while True:
                wiad = self._kolejka.get_nowait()
                self._obsłuz_wiadomosc(wiad)
        except queue.Empty:
            pass

        if self._w_trakcie:
            self.after(100, self._sprawdzaj_kolejke)

    def _obsłuz_wiadomosc(self, wiad: tuple):
        rodzaj = wiad[0]

        if rodzaj == "log":
            self._log(wiad[1])

        elif rodzaj == "postep":
            _, przetl, lacznie, eta = wiad
            if lacznie > 0:
                pct = przetl / lacznie
                self._pasek.set(pct)
                pct_txt = f"{int(pct * 100)}%"

                if eta and eta > 0:
                    if eta < 60:
                        eta_txt = f"~{int(eta)} s pozostało"
                    elif eta < 3600:
                        eta_txt = f"~{int(eta / 60)} min pozostało"
                    else:
                        eta_txt = f"~{eta / 3600:.1f} h pozostało"
                else:
                    eta_txt = "prawie gotowe..."

                self._etykieta_postepu.configure(
                    text=f"{pct_txt}  |  {przetl}/{lacznie} jednostek  |  {eta_txt}",
                    text_color=TEXT_MUTED,
                )

        elif rodzaj == "ostrzezenie_ocr":
            messagebox.showinfo(
                "Skan PDF",
                "Wykryto PDF bez warstwy tekstowej (skan).\n\n"
                "Tłumaczenie tekstu nie będzie możliwe.\n"
                "Rozważ użycie OCR (np. pytesseract) przed tłumaczeniem.",
            )

        elif rodzaj == "gotowe":
            self._zakoncz_tlumaczenie(sukces=True, sciezka=wiad[1])

        elif rodzaj == "blad":
            self._log(f"BŁĄD: {wiad[1]}")
            self._zakoncz_tlumaczenie(sukces=False)
            messagebox.showerror("Błąd tłumaczenia", f"Wystąpił błąd:\n\n{wiad[1][:400]}")

        elif rodzaj == "anulowano":
            self._log("Tłumaczenie anulowane.")
            self._zakoncz_tlumaczenie(sukces=False)

    def _zakoncz_tlumaczenie(self, sukces: bool, sciezka: str = ""):
        self._w_trakcie = False
        self._btn_tlumacz.configure(state="normal")
        self._btn_anuluj.configure(state="disabled")

        if sukces:
            self._pasek.set(1.0)
            czas = time.time() - self._czas_startu
            if czas < 60:
                czas_txt = f"{czas:.1f} s"
            else:
                czas_txt = f"{czas / 60:.1f} min"
            self._etykieta_postepu.configure(
                text=f"Gotowe!  Czas: {czas_txt}  |  {sciezka}",
                text_color=SUCCESS_COLOR,
            )
            self._log(f"Zapisano: {sciezka}")
        else:
            self._etykieta_postepu.configure(
                text="Przerwano.",
                text_color=TEXT_MUTED,
            )
