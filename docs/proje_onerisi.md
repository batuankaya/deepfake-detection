# Proje Önerisi

> **Proje Adı:** Çok-Modlu Deepfake Tespit Sistemi (Görüntü + Ses + Frekans Füzyonu)
> **Hazırlayan:** Batuhan Kaya — 232923023
> **Tarih:** 2026

Bir videoyu (.mp4) yükleyip **gerçek mi sahte mi** olduğunu söyleyen bir
sistem. Karar tek bir bakış açısıyla değil; videodaki **yüz hareketleri**,
**ses** ve **frekans izleri** üç ayrı modelle paralel analiz edilip
oylanarak verilir.

---

## 1) Problem Tanımı

**Deepfake** nedir? Üretici yapay zekâ modelleri (GAN, VAE, difüzyon) ile
gerçek bir kişinin yüzünü veya sesini taklit ederek üretilen sentetik
içeriklerdir. Bu içerikler artık çıplak gözle ayırt edilemeyecek kaliteye
ulaştığı için tespit görevi otomatik sistemlere kalmaktadır.

**Mevcut tespit yöntemlerinin üç temel zaafı:**

1. **Tek bakış açısına bağımlılık.** Yöntemlerin büyük çoğunluğu yalnızca
   görüntü piksellerini inceler; ses ve frekans alanı izleri göz ardı
   edilir.
2. **Zayıf cross-dataset genellemesi.** Bir veri setinde %97+ doğruluk
   veren modeller, farklı manipülasyon teknikleriyle üretilmiş veri
   setlerinde belirgin şekilde düşer.
3. **Görülmemiş manipülasyona karşı kırılganlık.** Eğitimde olmayan bir
   teknikle üretilmiş sahteler karşısında başarı çöker.

**Bu çalışmanın araştırma sorusu:** Bir videodaki **görüntü (yüz)**,
**ses** ve **frekans (DCT)** modalitelerini eşzamanlı analiz eden
karar-seviyesi füzyon mimarisi, tek modlu yöntemlere kıyasla in-dataset
ve cross-dataset performansta ne kadar iyileştirme sağlar?

---

## 2) Makale/Tez ve Literatür Taraması

Bu proje **iki çekirdek kaynağa** dayanır. Her ikisinin de tam metni
elimizde olup tüm sayısal alıntılar bu metinlerden doğrulanmıştır.

### 2.1. Qiao et al. (2025)

| Alan | İçerik |
|---|---|
| **Yayının Adı** | Towards Generalizable Deepfake Detection with Spatial-Frequency Collaborative Learning and Hierarchical Cross-Modal Fusion |
| **Yazarlar** | Mengyu Qiao, Runze Tian, Yang Wang |
| **Tarih** | Nisan 2025 |
| **Yayınlandığı Yer** | arXiv:2504.17223 |
| **Kullanılan Dataset** | FaceForensics++ (c23, c40), Celeb-DF v2, DFDC |
| **Uygulanan Yöntem** | Block-wise DCT + Cascaded Multi-scale Convolution + Hierarchical Cross-Modal Fusion |
| **Başarı Metrikleri** | FF++ c23: %97.43 Acc / %99.58 AUC · Celeb-DF v2 cross-dataset: **0.7468 AUC** |

**Bu projeye katkısı:** Frekans modülünün (Block-wise DCT + multi-scale
convolution) tasarımı bu makaleden alınmıştır. Celeb-DF v2 cross-dataset
AUC karşılaştırma noktası (0.7468) bu çalışmadan referans alınmıştır.

### 2.2. Yıldırım (2025)

| Alan | İçerik |
|---|---|
| **Yayının Adı** | Öz Denetimli Öğrenme Yaklaşımları ile Derin Sahte Ses ve Görüntü Manipülasyonunun Tespiti |
| **Yazar** | Merve Yıldırım |
| **Tarih** | 2025 |
| **Yayınlandığı Yer** | YÖK Tez Merkezi, Tez No: 957056 (Fırat Üniversitesi, Yüksek Lisans) |
| **Kullanılan Dataset** | DFDC (alt küme), Celeb-DF v2, deepfake-and-real-images |
| **Uygulanan Yöntem** | RotationNet/ResNet18 SSL + CNN-LSTM (görüntü) + Mel-CNN (ses) + öğrenilebilir-ağırlık ensemble |
| **Başarı Metrikleri** | Çok modlu (DFDC): %92.17 Acc / %92.02 F1 · Video (Celeb-DF v2): %89.17 Acc |

**Bu projeye katkısı:** Çok modlu (görüntü + ses) ensemble fikri ve
Celeb-DF v2 üzerinde Acc karşılaştırma noktası bu tezden alınmıştır.

### 2.3. Bu Projenin Konumu

İki referans modern deepfake tespitinin iki farklı çizgisini temsil eder:

- **Qiao et al.** — frekans-uzamsal birleşik öğrenme (tek video
  modalitesi içinde).
- **Yıldırım** — görüntü + ses çok-modlu ensemble.

Bu proje her iki çizgiyi **birleştirir**: 3 modalitenin (görüntü +
frekans + ses) karar-seviyesi füzyonu. Tasarım modalite-eksik veri
(yalnız ses, yalnız video) durumunda da çalışır.

---

## 3) Dataset

### 3.1. Kullanılan Veri Setleri

| Veri Seti | Modalite | İçerik | Kullanım |
|---|---|---|---|
| **FaceForensics++ c23** | Video | 1000 gerçek + 4000 sahte (Deepfakes, Face2Face, FaceSwap, NeuralTextures) | Birincil eğitim + in-dataset test |
| **ASVspoof 2019 LA** | Ses | 121k klip (bona-fide + spoofed) | Ses modülü eğitim/test |
| **Celeb-DF v2** | Video | 590 gerçek + 5639 sahte | Cross-dataset test (eğitime KATILMAZ) |
| **FF++ FaceShifter** | Video | 1000 sahte (5. manipülasyon) | Yalnız vahşi-doğa testi — eğitime/teste girmedi |
| **FF++ DeepFakeDetection** (Google DFD) | Video | ~3000 sahte | Yalnız vahşi-doğa testi — eğitime/teste girmedi |

FF++ c23'teki 4 sahte alt yöntem orijinal FaceForensics++ benchmark'ıdır.
FaceShifter sonradan eklenen 5. yöntemdir ve eğitim setinden bilinçli
olarak dışarıda bırakılmıştır — böylece görülmemiş manipülasyon karşısında
modelin nasıl davrandığını ölçebilen kontrollü bir test seti elimizde
kalmıştır.

### 3.2. Veri Bölünmesi

- **Eğitim:** %70 (yalnız FF++ c23 + ASVspoof 2019 LA)
- **Doğrulama:** %15
- **Test (in-dataset):** %15 → 751 video
- **Cross-dataset test:** Celeb-DF v2 (tamamı, eğitime hiç dahil değil)

Tüm bölünmeler `configs/config.yaml` ve
`src/preprocessing/build_dataset.py` ile sabittir; sabit seed ile
yeniden üretilebilir.

### 3.3. Ön İşleme Adımları

1. Video → kare ayrıştırma: OpenCV, 5 fps
2. Yüz tespiti ve hizalama: MTCNN, 224×224 kırpma
3. Ses ayrıştırma: FFmpeg, 16 kHz mono PCM
4. Mel-spektrogram: 128-bant, hop=512, n_fft=2048
5. Frekans dönüşümü: Block-wise DCT (8×8 blok)
6. Maksimum 150 kare/video; uzun videolarda zamansal alt-örnekleme

Pipeline kodu: `src/preprocessing/`.

### 3.4. Veri Etiği

Tüm veri setleri akademik kullanım için resmi kaynaklarından
edinilmiştir. Proje ders ödevi kapsamında olup yayınlanmayacaktır.
Üretilen modeller yalnız savunma amaçlı (tespit) kullanılır; herhangi
bir deepfake **üretimi** yapılmamıştır.

---

## 4) Problem Hangi Sorunu Çözüyor

### 4.1. Yaklaşım

Sistem deepfake tespit problemini **3 bağımsız bakış açısıyla** çözer
ve bu üç kararı **karar-seviyesi (decision-level) füzyon** ile
birleştirir:

| Modül | Neye Bakar? | Hangi Modelle? |
|---|---|---|
| **Görüntü** | Yüzdeki mimik, ışık, doku tutarsızlıkları | EfficientNet-B4 + Bi-LSTM |
| **Ses** | Robotik / sentetik ses izleri | Mel-Spektrogram + CNN |
| **Frekans** | GAN / difüzyon modellerinin yüksek frekansta bıraktığı izler | Block-wise DCT + Multi-scale CNN |

**Füzyon katmanı üç kuralla karar verir:**

1. **EER kalibrasyonu** — her modülün ham olasılığı ortak bir eşiğe
   çekilir ki "p=0.8" her modülde aynı şeyi ifade etsin.
2. **AUC²-ağırlıklı oylama** — daha güvenilir modülün (yüksek AUC) oyu
   daha fazla sayılır.
3. **STRONG_FAKE override** — bir modül çok yüksek güvenle (p ≥ 0.85)
   "sahte" derse, diğerleri "gerçek" dese bile karar SAHTE olur. Çünkü
   **kaçırılmış bir deepfake**, yanlış alarmdan çok daha pahalı bir
   hatadır.

Uygulama: `src/inference.py → fuse()`.

> **Neden eğitilebilir hiyerarşik füzyon değil?** Referans makaledeki
> (Qiao et al. 2025) o yöntem her örnekte görüntü + ses + frekans'ın
> **birlikte** olmasını ister. Bizim veri setlerimiz heterojen (FF++ ve
> Celeb-DF sessiz, ASVspoof yalnız ses); o yüzden modalite-eksik
> durumda da çalışan karar-seviyesi füzyon tercih edildi.

### 4.2. Başarı Metrikleri

Aşağıdaki tüm sayılar `reports/` altındaki JSON çıktılarından gelir;
her satırda kaynak dosya verilmiştir.

#### Modül Başarıları

| Modül | Test AUC | Test Acc | Test F1 | Cross AUC (Celeb-DF v2) | Kaynak |
|---|:---:|:---:|:---:|:---:|---|
| **Ses** (Mel-CNN, ASVspoof) | **0.9806** | %92.94 | **0.9594** | — | `reports/best_audio_test/metrics.json` (segment) |
| **Frekans** (Block-DCT + Multi-scale) | 0.8927 | %78.56 | 0.8522 | 0.7353 | `reports/analysis/ablation.json` (per-video) |
| **Görüntü** (EfficientNet-B4 + BiLSTM) | **0.9812** | %93.61 | **0.9588** | **0.8393** | `reports/analysis/ablation.json` (per-video) |

> Ses modülü FF++ test setinde değil ASVspoof eval'de ölçüldü (FF++
> videoları sessiz); o yüzden segment-seviyesi metrik. Frekans ve görüntü
> modülleri ise FF++ test setinde **video bazında** (751 video)
> değerlendirildi.

#### Karar-Seviyesi Füzyon (FF++ c23 Test, 751 video)

| Strateji | Accuracy | Recall | Precision | F1 |
|---|:---:|:---:|:---:|:---:|
| Yalnız görüntü (en güçlü tek modül) | %93.61 | — | — | 0.9599 |
| Eşit oylama | %93.48 | %94.68 | %97.10 | 0.9587 |
| AUC-ağırlıklı | %93.61 | %93.68 | %98.25 | 0.9591 |
| **AUC-ağırlıklı + STRONG_FAKE override** | **%94.14** | **%94.68** | %97.93 | **0.9628** |

Füzyon, en güçlü tek modülü accuracy'de **+0.53 puan**, F1'de **+0.003
puan** geçen tek konfigürasyondur (kaynak:
`reports/fusion_eval/metrics.json` ve `reports/analysis/ablation.json`).

#### Literatürle Karşılaştırma (Celeb-DF v2)

| Çalışma | Yaklaşım | Acc | F1 | AUC |
|---|---|:---:|:---:|:---:|
| Qiao et al. (2025) | Spatial-Freq + Hier. Fusion | — | — | 0.7468 |
| Yıldırım (2025) | ResNet18 video (SSL) | %89.17 | %89.29 | — |
| **Bizim — görüntü** | EfficientNet-B4 + BiLSTM | %77.91 | **0.8608** | **0.8393** |

**AUC ekseninde** görüntü modülümüz çekirdek referans Qiao et al.'ı
**+9.25 puan** aşmaktadır (0.8393 > 0.7468). Her ikisi de aynı protokol
altında çalışır: FF++ ile eğit → Celeb-DF v2'de cross-dataset test.

**Accuracy ekseninde** Yıldırım'ın %89.17 değeri kafa karıştırıcı
görünebilir; fakat tezinde (Bölüm 5.3) açıkça belirtildiği üzere
**"Celeb-DF v2 alt kümesi üzerinde sıfırdan eğitim yapılmış"** —
yani bu sayı **in-dataset** sonucudur, cross-dataset değil. Bizim
%77.91 ise FF++ eşiğiyle Celeb-DF v2 üstündeki saf cross-dataset
performansı. İki sayı doğrudan karşılaştırılamaz. AUC ekseni (0.8393)
ise eşik bağımsız olduğu için bu kalibrasyon farkından da etkilenmez
ve karşılaştırma için daha adil bir eksendir.

#### Hata Metrikleri — Modül Bazında EER

**EER** (Equal Error Rate), FPR = FNR olduğu noktadaki ortak hata
oranıdır; eşik bağımsız, sınıf-dengesizliğinden bağımsız bir model
kalitesi göstergesidir.

| Modül | EER (in-dataset) | EER (Celeb-DF v2 cross) | Kaynak |
|---|:---:|:---:|---|
| **Ses** | **0.0706** | — | `reports/best_audio_test/metrics.json` |
| **Frekans** | 0.2208 | 0.3295 | `reports/best_frequency_*/metrics.json` |
| **Görüntü** | **0.0616** | 0.2461 | `reports/best_spatial_*/metrics.json` |

#### Hata Metrikleri — Füzyon (FF++ c23 Test, Üretim Eşiği)

Karışıklık matrisi (kaynak: `reports/fusion_eval/metrics.json`):

|  | Tahmin: GERÇEK | Tahmin: SAHTE |
|---|:---:|:---:|
| **Gerçek: GERÇEK** | 138 (TN) | 12 (FP) |
| **Gerçek: SAHTE** | 32 (FN) | 569 (TP) |

| Hata Türü | Değer | Açıklama |
|---|:---:|---|
| **FPR** (False Positive Rate) | 12/150 = **%8.00** | Gerçek videoyu sahte sayma |
| **FNR** (False Negative Rate) | 32/601 = **%5.32** | Kaçırılmış deepfake (en kritik hata) |
| **Toplam hata** | 44/751 = **%5.86** | Genel yanlış sınıflandırma |

Tasarım kararı (§4.1 — STRONG_FAKE override) **FNR'yi minimize etmeye**
yöneliktir; kaçırılmış bir deepfake, yanlış alarmdan çok daha maliyetli
bir hata olarak değerlendirilmiştir.

#### Görülmemiş Manipülasyon Testi (Vahşi-Doğa)

Eğitim ve test setine HİÇ girmemiş 5 kaynaktan rastgele seçilmiş 100
video (seed=7, kaynak: `reports/analysis/wild_test.json`):

| Kaynak | Doğru | Oran |
|---|:---:|:---:|
| FF++ original (gerçek, eğitim-dışı kimlikler) | 20/20 | %100 |
| Celeb-DF Celeb-real (gerçek) | 17/20 | %85 |
| FF++ DeepFakeDetection (sahte, Google DFD) | 14/20 | %70 |
| Celeb-DF Celeb-synthesis (sahte) | 12/20 | %60 |
| **FF++ FaceShifter (görülmemiş manipülasyon)** | **1/20** | **%5** |
| **Genel** | **64/100** | **%64.0** |

**Bulgu:** Gerçek-tanıma sağlam (%92.5 = 37/40); fakat eğitimde
görülmeyen bir manipülasyon tekniği (FaceShifter) karşısında model
neredeyse kör (%5). Bu sonuç §1)'de 3. zaaf olarak tanımlanan
"görülmemiş manipülasyona karşı kırılganlık" problemini sayısal olarak
doğrular — model deepfake değil, eğitimde gördüğü 4 tekniğin parmak
izini tespit etmektedir.

---

## 5) Kodların Bulunduğu GitHub Linki

**https://github.com/batuankaya/deepfake-detection**

Yeniden üretim için repodaki ana dosyalar:

- `configs/config.yaml` — model ve eğitim parametreleri
- `scripts/` — CLI tahmin, başlatıcılar ve analiz scriptleri
- `reports/analysis/` — Bölüm 4.2'deki tüm sayıların ham JSON kaynakları
- `app/streamlit_app.py` — son kullanıcı web arayüzü (.mp4 yükle → gerçek/sahte tahmini)
- `README.md` — kurulum ve çalıştırma rehberi
