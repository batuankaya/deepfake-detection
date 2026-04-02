"""Modul 4: Hierarchical Cross-Modal Fusion.

Referanslar:
- "Towards Generalizable Deepfake Detection with Spatial-Frequency
  Collaborative Learning and Hierarchical Cross-Modal Fusion" (2025)
  https://arxiv.org/abs/2504.17223
- Oz Denetimli Ogrenme Yaklasimlari ile Derin Sahte Ses ve Goruntu
  Manipulasyonunun Tespiti (Merve Yildirim, 2025) - YOK Tez No: 957056

Teknikler:
- Shallow-layer attention enhancement (sig katman dikkat guclendirme)
- Deep-layer dynamic modulation (derin katman dinamik modulasyon)
- Cok modlu (goruntu + ses + frekans) hiyerarsik fuzyon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Capraz modlu dikkat mekanizmasi.

    Bir modalitenin ozelliklerini, diger modalitenin bilgisiyle
    agirlandirarak onemli bolgelere odaklanmayi saglar.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, dim) - hedef modalite ozellikleri
            context: (batch, dim) - kaynak modalite ozellikleri
        Returns:
            (batch, dim) - dikkat ile zenginlestirilmis ozellikler
        """
        b = query.shape[0]

        q = self.q_proj(query).view(b, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(b, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(b, self.num_heads, self.head_dim)

        attn = torch.bmm(
            q.view(b * self.num_heads, 1, self.head_dim),
            k.view(b * self.num_heads, self.head_dim, 1),
        ).view(b, self.num_heads) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn.unsqueeze(-1) * v).view(b, -1)
        out = self.out_proj(out)

        return self.norm(query + out)


class DynamicModulation(nn.Module):
    """Derin katman dinamik modulasyon.

    Modalitenin guvenilirligine gore agirlik ayarlama yapar.
    Dusuk kaliteli bir modalite (ornegin sessiz video) otomatik olarak
    daha az etkili hale getirilir.
    """

    def __init__(self, dim: int, num_modalities: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Sigmoid(),
        )
        self.num_modalities = num_modalities

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [spatial (b, d), audio (b, d), frequency (b, d)]
        Returns:
            (batch, dim) - dinamik olarak agirlandirilmis birlesik ozellik
        """
        concat = torch.cat(features, dim=1)
        gates = self.gate(concat)  # (batch, num_modalities)

        modulated = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            modulated = modulated + gates[:, i:i + 1] * feat

        return modulated


class HierarchicalFusionModel(nn.Module):
    """Hiyerarsik Capraz Modlu Fuzyon Modeli.

    Iki asamali fuzyon:
    1. Sig katman: Cross-modal attention ile modalitenin birbirini zenginlestirmesi
    2. Derin katman: Dynamic modulation ile guvenilirlik tabanli agirlandirma
    """

    def __init__(self, spatial_dim=256, audio_dim=256, freq_dim=256, hidden_dim=256):
        super().__init__()

        # Modalite boyutlarini ortak boyuta projekte et
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)

        # Aşama 1: Sig katman - Capraz dikkat (Shallow-layer attention)
        # Spatial <-> Frequency isbirligi (makaledeki ana tema)
        self.spatial_freq_attn = CrossModalAttention(hidden_dim)
        self.freq_spatial_attn = CrossModalAttention(hidden_dim)
        # Audio <-> Spatial isbirligi
        self.audio_spatial_attn = CrossModalAttention(hidden_dim)

        # Aşama 2: Derin katman - Dinamik modulasyon
        self.dynamic_mod = DynamicModulation(hidden_dim, num_modalities=3)

        # Nihai siniflandirici
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_feat: (batch, spatial_dim) - goruntu ozellik vektoru
            audio_feat: (batch, audio_dim) - ses ozellik vektoru
            freq_feat: (batch, freq_dim) - frekans ozellik vektoru
        Returns:
            (batch, 2) - gercek/sahte logits
        """
        # Ortak uzaya projekte et
        s = self.spatial_proj(spatial_feat)
        a = self.audio_proj(audio_feat)
        f = self.freq_proj(freq_feat)

        # Asama 1: Capraz dikkat ile zenginlestirme
        s_enhanced = self.spatial_freq_attn(query=s, context=f)
        f_enhanced = self.freq_spatial_attn(query=f, context=s)
        a_enhanced = self.audio_spatial_attn(query=a, context=s)

        # Asama 2: Dinamik modulasyon
        fused = self.dynamic_mod([s_enhanced, a_enhanced, f_enhanced])

        return self.classifier(fused)

    def get_confidence_score(
        self,
        spatial_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> tuple[float, str]:
        """Guvenilirlik skoru ve etiket dondurur."""
        with torch.no_grad():
            logits = self.forward(spatial_feat, audio_feat, freq_feat)
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item() * 100
            label = "GERCEK" if probs.argmax().item() == 0 else "SAHTE"
        return confidence, label
