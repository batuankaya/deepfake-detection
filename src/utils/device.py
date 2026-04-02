"""Cihaz secimi yardimci fonksiyonu."""

import torch


def get_device(config_device: str = "auto") -> torch.device:
    """Uygun hesaplama cihazini secer."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)
