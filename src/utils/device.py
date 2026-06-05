"""Cihaz secimi yardimci fonksiyonu."""

import torch

ALLOWED_DEVICES = {"auto", "cuda", "mps", "cpu"}


def get_device(config_device: str = "auto") -> torch.device:
    """Uygun hesaplama cihazini secer."""
    if config_device not in ALLOWED_DEVICES:
        raise ValueError(
            f"Gecersiz cihaz: '{config_device}'. "
            f"Gecerli degerler: {ALLOWED_DEVICES}"
        )

    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)
