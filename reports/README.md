# reports/ — Değerlendirme Çıktıları

Tüm sayısal sonuçlar, ROC/CM grafikleri ve derin analiz çıktıları
buradan üretilir. Üst seviye anlatı [README](../README.md) ve
[proje önerisi](../docs/proje_onerisi.md) içindedir; bu dizin ham
artefakttır.

## Modül başına test ve cross-dataset

| Klasör | Modül | Set | Kaynak |
|---|---|---|---|
| `best_audio_test/` | Audio (Mel-CNN) | ASVspoof 2019 LA eval | `metrics.json` + `test_{roc,cm}.png` |
| `best_frequency_test/` | Frequency (Block-DCT) | FF++ c23 test | aynı |
| `best_spatial_test/` | Spatial (EffNet-B4 + BiLSTM) | FF++ c23 test | aynı |
| `best_frequency_cross_celebdf/` | Frequency | Celeb-DF v2 (cross) | `metrics.json` + `cross_celebdf_{roc,cm}.png` |
| `best_spatial_cross_celebdf/` | Spatial | Celeb-DF v2 (cross) | aynı |

## Füzyon

| Dosya | Açıklama |
|---|---|
| `fusion_eval/metrics.json` | 2-modüllü (spatial + frequency) FF++ test füzyon sonucu — production karar fonksiyonu |

## Derin analiz (`analysis/`)

Tek geçişte üretilen ablation, hata, eşik, kalibrasyon, latency,
vahşi-doğa testleri. Yeniden üretmek için kök dizinde:

```bash
python scripts/test_analysis.py       # ablation + error + threshold (predictions.json cache'li)
python scripts/calibration_ci.py      # reliability + bootstrap CI
python scripts/benchmark_inference.py # CPU/GPU latency
python scripts/wild_test.py           # 16 eğitim-dışı video
```

| Dosya | Üreten script | İçerik |
|---|---|---|
| `predictions.json` | test_analysis.py | 751 FF++ test videosu için (label, type, p_spatial, p_frequency) — diğer analizlerin kaynağı |
| `ablation.{json,png}` | test_analysis.py | 5 fusion stratejisi karşılaştırması |
| `error_by_type.{json,png}` | test_analysis.py | FF++ 4 manipülasyon tipi recall/FN |
| `threshold_sweep.{json,png}` | test_analysis.py | Spatial+freq için Acc/F1/recall eğrileri |
| `calibration.{json,png}` | calibration_ci.py | Reliability + Brier + ECE |
| `bootstrap_ci.json` | calibration_ci.py | %95 güven aralıkları (Acc/F1/AUC) |
| `inference_time.{json,png}` | benchmark_inference.py | CPU/GPU bileşen-bileşen latency |
| `wild_test.{json,md}` | wild_test.py | 100 eğitim-dışı video (5 kaynak × 20), per-video verdict tablosu |
