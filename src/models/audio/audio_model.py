"""Modul 2: Isitsel (Ses) Analiz - Mel-spektrogram tabanli CNN."""

import torch
import torch.nn as nn


class AudioModel(nn.Module):
    """Ses spektrogrami uzerinde CNN siniflandirici."""

    def __init__(self, n_mels=128):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Blok 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Blok 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, time) - mel-spektrogram
        Returns:
            (batch, 2) - gercek/sahte logits
        """
        features = self.features(x)
        return self.classifier(features)
