# scripts/ — Yardımcı Scriptler

CLI tahmin aracı, web başlatıcılar ve analiz/rapor scriptleri. Hepsi
proje kökünden çalıştırılır.

## Kullanım Araçları

| Script | Açıklama |
|---|---|
| `predict_cli.py` | Tek video üzerinde komut satırı tahmin (insan-okunur veya `--json`) |
| `run_streamlit.ps1` | Streamlit web arayüzünü stabil ortamda başlatır (Windows) |
| `overnight_train.ps1` | Tüm modülleri sırayla gece eğitimi + değerlendirmeye sokar |

## Analiz/Rapor Scriptleri

| Script | Üreten | Açıklama |
|---|---|---|
| `test_analysis.py` | `reports/analysis/{predictions,ablation,error_by_type,threshold_sweep}.{json,png}` | FF++ test setinde tek geçişte ablation + tip-bazlı hata + eşik süpürmesi (predictions.json cache'li — ikinci çalıştırma anında) |
| `calibration_ci.py` | `reports/analysis/{calibration,bootstrap_ci}.{json,png}` | Reliability diagram + Brier/ECE + 1000-resample %95 CI. Girdi: `predictions.json` |
| `benchmark_inference.py` | `reports/analysis/inference_time.{json,png}` | 8 ham video üstünde CPU+CUDA bileşen-bileşen latency (extract / face / spatial / frequency / audio / fusion) |
| `wild_test.py` | `reports/analysis/wild_test.{json,md}` | 5 eğitim-dışı kaynaktan rastgele 100 video, uçtan-uca tahmin, per-video tablo |
| `build_pdf.py` | `docs/232923023_BatuhanKaya.pdf` | `docs/proje_onerisi.md` → tek PDF (gömülü grafikler dahil) |

## Bağımlılık zinciri

```
test_analysis.py  ──>  predictions.json  ──>  calibration_ci.py
                                              (predictions.json'u re-use eder)
benchmark_inference.py                   (bagimsiz, ham video gerek)
wild_test.py                             (bagimsiz, ham video gerek)
build_pdf.py                             (docs/proje_onerisi.md)
```

`predictions.json` mevcutsa `test_analysis.py` ve `calibration_ci.py`
saniyeler içinde tamamlanır; aksi halde tüm test setinde forward
pass çalışır (~5 dakika CUDA üstünde).
