"""Frekans analizi gorsellestirme araclari.

Bu modul egitim gerektirmez - saf sinyal isleme ile
FFT/DCT haritalarini gorsellestir.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_fft_analysis(image: np.ndarray, save_path: str = None) -> plt.Figure:
    """Bir goruntunun FFT analizini 4 panelli grafik olarak gorsellestir.

    Panel 1: Orijinal goruntu
    Panel 2: FFT guc spektrumu
    Panel 3: Yuksek frekans maskesi
    Panel 4: Azimutal ortalama profili
    """
    from src.models.frequency.frequency_model import FrequencyAnalyzer

    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    # FFT hesapla
    spectrum = FrequencyAnalyzer.compute_fft_spectrum(gray)
    azimuthal = FrequencyAnalyzer.compute_azimuthal_average(spectrum)

    # Yuksek frekans maskesi
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(cy, cx) // 4
    y, x = np.ogrid[:h, :w]
    mask = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) > radius
    high_freq = spectrum * mask

    # Grafik
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Fourier Frekans Analizi", fontsize=16, fontweight="bold")

    # Panel 1: Orijinal
    if len(image.shape) == 3:
        axes[0, 0].imshow(image)
    else:
        axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Orijinal Goruntu")
    axes[0, 0].axis("off")

    # Panel 2: FFT spektrum
    im = axes[0, 1].imshow(spectrum, cmap="hot")
    axes[0, 1].set_title("FFT Guc Spektrumu")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Panel 3: Yuksek frekans
    im2 = axes[1, 0].imshow(high_freq, cmap="inferno")
    axes[1, 0].set_title("Yuksek Frekans Bilesenleri\n(GAN artifact bolgeleri)")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # Panel 4: Azimutal profil
    axes[1, 1].plot(azimuthal, color="#e74c3c", linewidth=1.5)
    axes[1, 1].fill_between(range(len(azimuthal)), azimuthal,
                             alpha=0.3, color="#e74c3c")
    axes[1, 1].set_xlabel("Frekans (piksel)")
    axes[1, 1].set_ylabel("Ortalama Genlik")
    axes[1, 1].set_title("Azimutal Ortalama\n(Tepeler = yapay iz)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_dct_analysis(image: np.ndarray, save_path: str = None) -> plt.Figure:
    """Block-wise DCT analizini gorsellestir."""
    from src.models.frequency.frequency_model import FrequencyAnalyzer

    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()

    dct_map = FrequencyAnalyzer.compute_dct_spectrum(gray, block_size=8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Block-wise DCT Analizi", fontsize=16, fontweight="bold")

    # Orijinal
    if len(image.shape) == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Orijinal Goruntu")
    axes[0].axis("off")

    # DCT haritasi
    im = axes[1].imshow(dct_map, cmap="viridis")
    axes[1].set_title("DCT Katsayi Haritasi")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # DCT enerji dagilimi
    energy = np.mean(dct_map.reshape(-1, 8, 8), axis=0)
    im2 = axes[2].imshow(energy, cmap="YlOrRd")
    axes[2].set_title("Ortalama DCT Enerji Dagilimi\n(8x8 blok)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for i in range(8):
        for j in range(8):
            axes[2].text(j, i, f"{energy[i, j]:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if energy[i, j] < energy.max() * 0.7 else "white")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison(real_image: np.ndarray, fake_image: np.ndarray,
                    save_path: str = None) -> plt.Figure:
    """Gercek ve sahte goruntu frekans karsilastirmasi."""
    from src.models.frequency.frequency_model import FrequencyAnalyzer

    def to_gray(img):
        return np.mean(img, axis=2) if len(img.shape) == 3 else img

    real_spectrum = FrequencyAnalyzer.compute_fft_spectrum(to_gray(real_image))
    fake_spectrum = FrequencyAnalyzer.compute_fft_spectrum(to_gray(fake_image))
    real_azimuthal = FrequencyAnalyzer.compute_azimuthal_average(real_spectrum)
    fake_azimuthal = FrequencyAnalyzer.compute_azimuthal_average(fake_spectrum)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Gercek vs Sahte - Frekans Karsilastirmasi",
                 fontsize=16, fontweight="bold")

    # Ust satir: Gercek
    axes[0, 0].imshow(real_image if len(real_image.shape) == 3 else real_image, cmap="gray")
    axes[0, 0].set_title("Gercek Goruntu")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(real_spectrum, cmap="hot")
    axes[0, 1].set_title("Gercek - FFT Spektrum")
    axes[0, 1].axis("off")

    axes[0, 2].plot(real_azimuthal, color="#27ae60", linewidth=1.5, label="Gercek")
    axes[0, 2].fill_between(range(len(real_azimuthal)), real_azimuthal,
                             alpha=0.3, color="#27ae60")
    axes[0, 2].set_title("Gercek - Frekans Profili")
    axes[0, 2].grid(True, alpha=0.3)

    # Alt satir: Sahte
    axes[1, 0].imshow(fake_image if len(fake_image.shape) == 3 else fake_image, cmap="gray")
    axes[1, 0].set_title("Sahte Goruntu")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(fake_spectrum, cmap="hot")
    axes[1, 1].set_title("Sahte - FFT Spektrum")
    axes[1, 1].axis("off")

    axes[1, 2].plot(fake_azimuthal, color="#e74c3c", linewidth=1.5, label="Sahte")
    axes[1, 2].fill_between(range(len(fake_azimuthal)), fake_azimuthal,
                             alpha=0.3, color="#e74c3c")
    axes[1, 2].set_title("Sahte - Frekans Profili")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
