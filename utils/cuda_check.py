"""
utils/cuda_check.py
===================
Wykrywanie dostępności CUDA przy starcie aplikacji.
"""


def check_cuda() -> tuple[bool, str]:
    """
    Sprawdza czy CUDA jest dostępna.
    Zwraca (dostępna: bool, opis_urządzenia: str).
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            return True, f"{name} ({vram} MB VRAM)"
        return False, "CPU (brak obsługi CUDA)"
    except ImportError:
        return False, "CPU (torch niezainstalowany)"
    except Exception as e:
        return False, f"CPU (błąd CUDA: {e})"
