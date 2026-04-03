# Deepfake Tespit Sistemi

**Spatial-Frequency Collaborative Learning ve Hierarchical Cross-Modal Fusion Tabanlı Deepfake Tespiti**

> Batuhan Kaya — 232923023

---

## Proje Hakkında

Bu proje, manipüle edilmiş sahte içerikleri (Deepfake) sadece görüntüye veya sadece sese bakarak değil; farklı veri tiplerini eşzamanlı inceleyerek tespit etmeyi amaçlamaktadır.

Bir videodaki:
- **Yüz hareketlerinin tutarsızlıkları** (CNN + LSTM ile)
- **Görüntü piksellerine gizlenmiş yapay zeka izleri** (Block-wise DCT + Fourier analizi ile)
- **Ses frekanslarındaki yapaylıklar** (Mel-spektrogram analizi ile)

eşzamanlı olarak incelenir ve **Hierarchical Cross-Modal Fusion** ile nihai "Gerçek/Sahte" kararı verilir.

---

## Sistem Mimarisi

```
                    ┌─────────────────┐
                    │   Video Girdi   │
                    │     (.mp4)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  Modül 1   │  │  Modül 2   │  │  Modül 3   │
     │  Uzamsal   │  │  İşitsel   │  │  Frekans   │
     │ Analiz     │  │  Analiz    │  │  Analiz    │
     │            │  │            │  │            │
     │ EfficientNet  │ Mel-Spekt. │  │ Block DCT  │
     │ + LSTM     │  │ CNN        │  │ Multi-scale│
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
              ┌─────────────────────┐
              │      Modül 4        │
              │  Hierarchical       │
              │  Cross-Modal Fusion │
              │                     │
              │ 1. Cross-Modal      │
              │    Attention        │
              │ 2. Dynamic          │
              │    Modulation       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Gerçek / Sahte     │
              │  Güvenilirlik Skoru │
              └─────────────────────┘
```

---

## Modüller

| Modül | Açıklama | Teknik |
|-------|----------|--------|
| **Uzamsal Analiz** | Yüz hareketlerindeki mimik, ışık ve doku tutarsızlıkları | EfficientNet-B4 + Bidirectional LSTM |
| **İşitsel Analiz** | Ses frekanslarındaki robotik/sentetik tonlamalar | Mel-Spektrogram + CNN |
| **Frekans Analizi** | GAN/Difüzyon modellerinin bıraktığı frekans izleri | Block-wise DCT + Multi-scale CNN + Differential Accumulation |
| **Füzyon** | Tüm modüllerin hiyerarşik birleştirilmesi | Cross-Modal Attention + Dynamic Modulation |

---

## Teknoloji Yığını

| Bileşen | Teknoloji |
|---------|-----------|
| Dil | Python 3.10+ |
| Derin Öğrenme | PyTorch |
| Görüntü İşleme | OpenCV, MTCNN |
| Ses İşleme | Librosa |
| Frekans Analizi | NumPy FFT, SciPy DCT |
| Web Arayüzü | Streamlit |
| GPU | NVIDIA RTX 4070 (12 GB) |

---

## Proje Yapısı

```
deepfake/
├── app/
│   └── streamlit_app.py              # Streamlit web arayüzü
├── configs/
│   └── config.yaml                   # Model ve eğitim parametreleri
├── data/
│   ├── raw/                          # Ham veri setleri
│   └── processed/                    # İşlenmiş veriler
├── src/
│   ├── inference.py                  # Uçtan uca inference pipeline
│   ├── train.py                      # Eğitim scripti
│   ├── preprocessing/
│   │   ├── video_processor.py        # Video → kare + ses ayrıştırma
│   │   ├── face_detector.py          # MTCNN yüz tespiti
│   │   └── dataset.py                # DataLoader ve veri seti sınıfları
│   ├── models/
│   │   ├── spatial/
│   │   │   └── spatial_model.py      # EfficientNet + LSTM
│   │   ├── audio/
│   │   │   └── audio_model.py        # Mel-Spektrogram CNN
│   │   ├── frequency/
│   │   │   └── frequency_model.py    # Block-wise DCT + Multi-scale
│   │   └── fusion/
│   │       └── fusion_model.py       # Hierarchical Cross-Modal Fusion
│   └── utils/
│       ├── device.py                 # GPU/CPU cihaz seçimi
│       ├── metrics.py                # Değerlendirme metrikleri (Acc, F1, AUC)
│       └── visualize.py              # FFT/DCT görselleştirme araçları
├── tests/
│   └── test_models.py                # Model doğrulama testleri
├── notebooks/                        # Deneysel Jupyter notebook'lar
├── checkpoints/                      # Eğitilmiş model ağırlıkları
└── requirements.txt
```

---

## Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/batuankaya/deepfake-detection.git
cd deepfake-detection

# Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## Kullanım

### Web Arayüzü

```bash
streamlit run app/streamlit_app.py
```

Tarayıcıdan `http://localhost:8501` adresine gidip `.mp4` video yükleyerek analiz yapılabilir. Frekans analizi modülü şu an aktif olarak çalışmaktadır; diğer modüller eğitim sonrası aktif olacaktır.

### Eğitim

```bash
# Frekans modülünü eğit
python -m src.train --config configs/config.yaml --module frequency

# Uzamsal modülü eğit
python -m src.train --config configs/config.yaml --module spatial

# Ses modülünü eğit
python -m src.train --config configs/config.yaml --module audio
```

Eğitim için verilerin `data/processed/train/real/` ve `data/processed/train/fake/` klasörlerinde bulunması gerekmektedir.

### Testler

```bash
python tests/test_models.py
```

Tüm modüllerin şekil/boyut uyumunu dummy veri ile doğrular.

---

## Güvenlik

Uygulama aşağıdaki güvenlik önlemlerini içermektedir:

- **XSS koruması**: Kullanıcı girdisi doğrudan HTML olarak render edilmez
- **Dosya doğrulama**: Yüklenen dosyalarda uzantı whitelist ve boyut limiti (200MB) kontrolü
- **Path traversal koruması**: Veri seti yükleme sırasında `is_relative_to()` ile yol doğrulama
- **Deserialization güvenliği**: `np.load(allow_pickle=False)`, `yaml.safe_load()`
- **Geçici dosya yönetimi**: `try/finally` bloğu ile garanti temizlik
- **Girdi doğrulama**: Cihaz seçiminde whitelist, video yolunda dosya varlık kontrolü
- **Hata yönetimi**: Kullanıcıya iç sistem bilgisi sızdırılmaz

---

## Referanslar

1. **Towards Generalizable Deepfake Detection with Spatial-Frequency Collaborative Learning and Hierarchical Cross-Modal Fusion** (2025)
   — [arxiv.org/abs/2504.17223](https://arxiv.org/abs/2504.17223)

2. **Öz Denetimli Öğrenme Yaklaşımları ile Derin Sahte Ses ve Görüntü Manipülasyonunun Tespiti**
   — Merve Yıldırım, 2025 — YÖK Tez Merkezi (Tez No: 957056)

---

## Lisans

Bu proje akademik amaçlarla geliştirilmektedir.
