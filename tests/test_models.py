"""Tum modullerin sekil/boyut uyumunu dummy veri ile dogrulamak icin test scripti."""

import torch
import sys
sys.path.insert(0, ".")


def test_spatial_model():
    """Modul 1: Uzamsal Analiz - EfficientNet + LSTM"""
    from src.models.spatial.spatial_model import SpatialModel

    model = SpatialModel(lstm_hidden=256, lstm_layers=2, dropout=0.3)
    model.eval()

    # (batch=2, seq_len=10, C=3, H=224, W=224)
    dummy = torch.randn(2, 10, 3, 224, 224)

    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, 2), f"Beklenen (2, 2), alinan {out.shape}"
    print(f"[OK] Spatial Model: girdi {dummy.shape} -> cikti {out.shape}")


def test_audio_model():
    """Modul 2: Isitsel Analiz - Mel-Spektrogram CNN"""
    from src.models.audio.audio_model import AudioModel

    model = AudioModel(n_mels=128)
    model.eval()

    # (batch=2, 1, n_mels=128, time=300)
    dummy = torch.randn(2, 1, 128, 300)

    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, 2), f"Beklenen (2, 2), alinan {out.shape}"
    print(f"[OK] Audio Model: girdi {dummy.shape} -> cikti {out.shape}")


def test_frequency_model():
    """Modul 3: Frekans Analizi - Block-wise DCT + Multi-scale"""
    from src.models.frequency.frequency_model import FrequencyModel

    model = FrequencyModel(in_channels=3, feature_dim=256)
    model.eval()

    # (batch=2, C=3, H=224, W=224) - 8'e bolunebilir boyut
    dummy = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out = model(dummy)
        feat = model.extract_features(dummy)

    assert out.shape == (2, 2), f"Beklenen (2, 2), alinan {out.shape}"
    print(f"[OK] Frequency Model: girdi {dummy.shape} -> cikti {out.shape}")
    print(f"     Feature vektoru: {feat.shape}")


def test_fusion_model():
    """Modul 4: Hierarchical Cross-Modal Fusion"""
    from src.models.fusion.fusion_model import HierarchicalFusionModel

    model = HierarchicalFusionModel(
        spatial_dim=256, audio_dim=256, freq_dim=256, hidden_dim=256
    )
    model.eval()

    # Her modülden 256 boyutlu ozellik vektoru
    spatial_feat = torch.randn(2, 256)
    audio_feat = torch.randn(2, 256)
    freq_feat = torch.randn(2, 256)

    with torch.no_grad():
        out = model(spatial_feat, audio_feat, freq_feat)
        confidence, label = model.get_confidence_score(
            spatial_feat, audio_feat, freq_feat
        )

    assert out.shape == (2, 2), f"Beklenen (2, 2), alinan {out.shape}"
    print(f"[OK] Fusion Model: girdi 3x(2, 256) -> cikti {out.shape}")
    print(f"     Guvenilirlik: %{confidence:.1f} - Etiket: {label}")


def test_end_to_end():
    """Uctan uca pipeline testi (dummy veri ile)"""
    from src.models.spatial.spatial_model import SpatialModel
    from src.models.audio.audio_model import AudioModel
    from src.models.frequency.frequency_model import FrequencyModel
    from src.models.fusion.fusion_model import HierarchicalFusionModel

    batch = 2

    # --- Modulleri olustur ---
    spatial_model = SpatialModel(lstm_hidden=256, lstm_layers=2, dropout=0.3)
    audio_model = AudioModel(n_mels=128)
    freq_model = FrequencyModel(in_channels=3, feature_dim=256)
    fusion_model = HierarchicalFusionModel(
        spatial_dim=256, audio_dim=256, freq_dim=256, hidden_dim=256
    )

    for m in [spatial_model, audio_model, freq_model, fusion_model]:
        m.eval()

    # --- Dummy girdiler ---
    video_frames = torch.randn(batch, 10, 3, 224, 224)
    audio_spec = torch.randn(batch, 1, 128, 300)
    freq_input = torch.randn(batch, 3, 224, 224)

    with torch.no_grad():
        # Spatial -> ozellik vektoru almak icin classifier oncesini kullan
        s_out = spatial_model(video_frames)
        # Ozellik boyutunu 256'ya projekte et (basit linear)
        spatial_proj = torch.nn.Linear(2, 256)
        spatial_feat = spatial_proj(s_out)

        # Audio -> benzer sekilde
        a_out = audio_model(audio_spec)
        audio_proj = torch.nn.Linear(2, 256)
        audio_feat = audio_proj(a_out)

        # Frequency -> extract_features ile direkt 256 boyut
        # extract_features 256*4*4 = 4096 boyutlu vektor donduruyor
        freq_feat_raw = freq_model.extract_features(freq_input)
        freq_proj = torch.nn.Linear(freq_feat_raw.shape[1], 256)
        freq_feat = freq_proj(freq_feat_raw)

        # Fusion
        final_out = fusion_model(spatial_feat, audio_feat, freq_feat)
        confidence, label = fusion_model.get_confidence_score(
            spatial_feat, audio_feat, freq_feat
        )

    print(f"\n[OK] Uctan Uca Pipeline Testi Basarili!")
    print(f"     Video  -> Spatial  -> {s_out.shape}")
    print(f"     Ses    -> Audio    -> {a_out.shape}")
    print(f"     Kare   -> Frekans  -> {freq_feat.shape}")
    print(f"     Fusion -> Sonuc    -> {final_out.shape}")
    print(f"     Guvenilirlik: %{confidence:.1f} | Etiket: {label}")
    print(f"     (Egitilmemis model - rastgele sonuc beklenir)")


if __name__ == "__main__":
    print("=" * 60)
    print("DEEPFAKE TESPIT SISTEMI - MODEL DOGRULAMA TESTLERI")
    print("=" * 60)

    print("\n--- Modul 1: Uzamsal Analiz ---")
    test_spatial_model()

    print("\n--- Modul 2: Isitsel Analiz ---")
    test_audio_model()

    print("\n--- Modul 3: Frekans Analizi ---")
    test_frequency_model()

    print("\n--- Modul 4: Hierarchical Fusion ---")
    test_fusion_model()

    print("\n--- Uctan Uca Pipeline ---")
    test_end_to_end()

    print("\n" + "=" * 60)
    print("TUM TESTLER BASARILI!")
    print("=" * 60)
