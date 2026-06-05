"""Modul 1: Uzamsal (Goruntu) Analiz - CNN + LSTM modeli."""

import torch
import torch.nn as nn
from torchvision import models


class SpatialModel(nn.Module):
    """EfficientNet omurga + LSTM temporal analiz."""

    def __init__(self, lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()

        # EfficientNet-B4 omurga (onceden egitilmis)
        efficientnet = models.efficientnet_b4(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])
        self.feature_dim = 1792  # EfficientNet-B4 cikis boyutu

        # LSTM - temporal tutarsizlik tespiti
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # Siniflandirici
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, C, H, W) - video kare dizisi
        Returns:
            (batch, 2) - gercek/sahte logits
        """
        batch, seq_len, c, h, w = x.shape

        # Her kareyi CNN'den gecir
        x = x.view(batch * seq_len, c, h, w)
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        features = features.view(batch, seq_len, self.feature_dim)

        # LSTM ile temporal analiz
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]

        return self.classifier(last_hidden)
