"""Modul 3: Frekans Analiz - Spatial-Frequency Collaborative Learning.

Referans: "Towards Generalizable Deepfake Detection with Spatial-Frequency
Collaborative Learning and Hierarchical Cross-Modal Fusion" (2025)
https://arxiv.org/abs/2504.17223

Teknikler:
- Block-wise DCT ile lokal spektral ozellik cikarma
- Multi-scale convolution ile cascaded analiz
- Scale-invariant differential accumulation ile global desen tespiti
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDCT(nn.Module):
    """Block-wise Ayrik Kosinus Donusumu (DCT).

    Goruntuleri bloklara bolup her blokta DCT uygular.
    GAN/Difuzyon modellerinin biraktigi lokal frekans izlerini yakalar.
    """

    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        # DCT basis matrisini onceden hesapla
        self.register_buffer("dct_matrix", self._create_dct_matrix(block_size))

    @staticmethod
    def _create_dct_matrix(n: int) -> torch.Tensor:
        """NxN DCT-II basis matrisi olusturur."""
        matrix = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    matrix[k, i] = 1.0 / np.sqrt(n)
                else:
                    matrix[k, i] = np.sqrt(2.0 / n) * np.cos(
                        np.pi * (2 * i + 1) * k / (2 * n)
                    )
        return matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W) - girdi goruntusu
        Returns:
            (batch, C, H, W) - DCT katsayilari (blok bazinda)
        """
        b, c, h, w = x.shape
        bs = self.block_size

        # Bloklara bol
        x = x.unfold(2, bs, bs).unfold(3, bs, bs)  # (b, c, h//bs, w//bs, bs, bs)
        shape = x.shape

        # 2D DCT: D * X * D^T
        x = torch.matmul(self.dct_matrix, x)
        x = torch.matmul(x, self.dct_matrix.t())

        # Tekrar birlestir
        x = x.contiguous().view(b, c, shape[2], shape[3], bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, c, shape[2] * bs, shape[3] * bs)

        return x


class MultiScaleSpectralExtractor(nn.Module):
    """Cascaded Multi-scale Convolution ile lokal spektral ozellik cikarma.

    Farkli olceklerdeki konvolusyonlar, farkli frekans bantlarindaki
    artifact'leri yakalamak icin kullanilir.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        # Coklu olcekte konvolusyon (3x3, 5x5, 7x7)
        self.scale_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        self.scale_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        self.scale_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        # 1x1 kanal donusumu
        self.scale_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.scale_3x3(x)
        s2 = self.scale_5x5(x)
        s3 = self.scale_7x7(x)
        s4 = self.scale_1x1(x)
        return torch.cat([s1, s2, s3, s4], dim=1)


class DifferentialAccumulator(nn.Module):
    """Scale-invariant Differential Accumulation.

    Ardisik kareler veya farkli olcekler arasindaki frekans farklarini
    birikimli olarak hesaplayarak global spektral desenleri tespit eder.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        diff = current - previous
        diff = self.norm(diff)
        # Kanal bazinda onem agirlandirma
        gate = self.gate(diff).unsqueeze(-1).unsqueeze(-1)
        return diff * gate + current


class FrequencyAnalyzer:
    """Goruntuleri frekans uzayina tasiyarak GAN artifact'lerini tespit eder."""

    @staticmethod
    def compute_fft_spectrum(image: np.ndarray) -> np.ndarray:
        """2D FFT uygulayip guc spektrumunu dondurur."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        spectrum = np.log1p(magnitude)
        return spectrum

    @staticmethod
    def compute_dct_spectrum(image: np.ndarray, block_size=8) -> np.ndarray:
        """Block-wise DCT uygulayip frekans haritasi dondurur."""
        from scipy.fft import dctn

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        h, w = gray.shape
        h_blocks = h // block_size * block_size
        w_blocks = w // block_size * block_size
        gray = gray[:h_blocks, :w_blocks]

        dct_map = np.zeros_like(gray)
        for i in range(0, h_blocks, block_size):
            for j in range(0, w_blocks, block_size):
                block = gray[i:i + block_size, j:j + block_size]
                dct_map[i:i + block_size, j:j + block_size] = np.log1p(
                    np.abs(dctn(block, norm="ortho"))
                )
        return dct_map

    @staticmethod
    def compute_azimuthal_average(spectrum: np.ndarray) -> np.ndarray:
        """Frekans spektrumunun azimutal ortalamasini hesaplar."""
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)
        radial_profile = np.zeros(max_radius)

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        for radius in range(max_radius):
            mask = r == radius
            radial_profile[radius] = spectrum[mask].mean() if mask.any() else 0

        return radial_profile


class FrequencyModel(nn.Module):
    """Spatial-Frequency Collaborative Learning modeli.

    Block-wise DCT + Multi-scale CNN + Differential Accumulation
    bilesenleriyle frekans tabanli deepfake tespiti.
    """

    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()

        self.block_dct = BlockDCT(block_size=8)

        # Lokal spektral ozellik cikarici (DCT sonrasi)
        self.local_extractor = MultiScaleSpectralExtractor(in_channels, 64)

        # Global spektral ozellik cikarici (FFT tabanli)
        self.global_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Diferansiyel birikimci
        self.diff_accumulator = DifferentialAccumulator(64)

        # Birlesik ozellik isleme
        self.feature_processor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Siniflandirici oncesi ozellik vektoru dondurur (fuzyon icin)."""
        dct_x = self.block_dct(x)
        local_feat = self.local_extractor(dct_x)
        global_feat = self.global_extractor(x)
        global_feat = self.diff_accumulator(global_feat, global_feat)
        combined = torch.cat([local_feat, global_feat], dim=1)
        processed = self.feature_processor(combined)
        flat = processed.flatten(1)
        return flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) - girdi goruntusu (224x224)
        Returns:
            (batch, 2) - gercek/sahte logits
        """
        features = self.extract_features(x)
        return self.classifier(features)
