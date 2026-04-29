# Proje Önerisi

> **Proje Adı:** Spatial-Frequency Collaborative Learning ve Hierarchical Cross-Modal Fusion Tabanlı Deepfake Tespiti
> **Hazırlayan:** Batuhan Kaya — 232923023
> **Danışman:** Kerem Hoca
> **Tarih:** 2026

---

## İçindekiler

1. [Problem Tanımı](#1-problem-tanımı)
2. [Makale/Tez ve Literatür Taraması](#2-makaletez-ve-literatür-taraması)
3. [Dataset](#3-dataset)
4. [Problem Hangi Sorunu Çözüyor](#4-problem-hangi-sorunu-çözüyor)

---

## 1. Problem Tanımı

### 1.1. Genel Tanım

**Deepfake**, üretici yapay zekâ modelleri (GAN, VAE, Difüzyon) ile gerçek bir kişinin yüzünü, sesini veya bedenini taklit ederek üretilen sentetik içeriklerdir. Bu içerikler artık çıplak gözle ayırt edilemeyecek kaliteye ulaşmıştır ve kötüye kullanım potansiyeli yüksektir:

- **Politik dezenformasyon:** Liderlerin söylemediği sözleri "söylüyor" göstermek
- **Finansal dolandırıcılık:** Üst yönetim taklidiyle banka/şirket talimatı oluşturmak (CEO fraud)
- **Kimlik hırsızlığı:** Biometrik doğrulama sistemlerinin (KYC) atlatılması
- **Bireysel mağduriyet:** Rıza dışı pornografik içerik (non-consensual deepfake)

### 1.2. Teknik Problem

Mevcut deepfake tespit yöntemlerinin üç temel zaafı vardır:

1. **Tek modaliteye bağımlılık:** Yöntemlerin büyük çoğunluğu yalnızca uzamsal görüntü piksellerini inceler; ses ve frekans alanı izleri göz ardı edilir.
2. **Zayıf genelleme (generalization):** Bir veri setinde (örn. FaceForensics++) %97+ doğruluk veren modeller, farklı bir manipülasyon tekniği ile üretilmiş veri setinde (örn. Celeb-DF v2, DFDC) %65-75 AUC seviyesine düşmektedir.
3. **Sıkıştırma/dönüşüme dayanıksızlık:** JPEG sıkıştırma, yeniden boyutlandırma, sosyal medya yeniden kodlaması gibi gerçek dünya bozulmaları altında doğruluk ciddi şekilde düşmektedir.

### 1.3. Araştırma Sorusu

> *Bir videodaki **uzamsal (yüz)**, **işitsel (ses)** ve **frekans (DCT/FFT)** modalitelerini eşzamanlı analiz ederek **hiyerarşik çapraz modlu füzyon** uygulayan bir sistem; tek modlu yöntemlere kıyasla **cross-dataset genelleme** ve **sıkıştırmaya dayanıklılık** açısından ne kadar iyileştirme sağlar?*

### 1.4. Hipotez

GAN ve difüzyon modelleri yüksek frekans bandında karakteristik izler bırakır; yüz hareketleri zamansal tutarsızlıklar üretir; sentetik ses ise mel-spektrogram düzleminde robotik harmonikler içerir. Bu **üç bağımsız delil** hiyerarşik bir füzyon mekanizmasıyla birleştirildiğinde, tek modaliteye dayanan tespit edicilerden istatistiksel olarak anlamlı şekilde daha yüksek başarı elde edilecektir.

---

## 2. Makale/Tez ve Literatür Taraması

Toplam **30 yayın**, 2020-2025 döneminde, **7 kategoride** taranmıştır. İstenen kolonlar (Yayının Adı, Yazarlar, Tarih, Yayınlandığı Yer, Kullanılan Dataset, Uygulanan Yöntem, Başarı Metrikleri) tüm tablolarda yer almaktadır.

**Tam tablo için:** [`docs/literature_review.md`](literature_review.md)
**CSV/Excel formatı:** [`docs/literature_review.csv`](literature_review.csv)
**LaTeX formatı (tez için):** [`docs/literature_review.tex`](literature_review.tex)

### 2.1. Çekirdek Referanslar (Özet)

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 1 | Towards Generalizable Deepfake Detection with Spatial-Frequency Collaborative Learning and Hierarchical Cross-Modal Fusion | Mengyu Qiao, Runze Tian | Nisan 2025 | arXiv:2504.17223 | FF++ (c23, c40), Celeb-DF v2, DFDC | Block-wise DCT + Cascaded Multi-scale Conv + Hierarchical Cross-Modal Fusion | FF++ c23: %97.43 Acc / %99.58 AUC; Celeb-DF v2: %74.68 AUC |
| 2 | Öz Denetimli Öğrenme Yaklaşımları ile Derin Sahte Ses ve Görüntü Manipülasyonunun Tespiti | Merve Yıldırım | 2025 | YÖK Tez No: 957056 | FaceForensics++, ASVspoof | Wav2Vec/HuBERT (ses) + ViT (görüntü) + SSL | Tez içinde raporlu |

### 2.2. Kategorilere Göre Yayın Dağılımı

| Kategori | Yayın Sayısı | Tablo |
|----------|:---:|:---:|
| Çekirdek (Ana Referans) Çalışmalar | 2 | Tablo 1 |
| Frekans Tabanlı (DCT/FFT) Tespit Yöntemleri | 5 | Tablo 2 |
| Çok Modlu (Multimodal) ve Hiyerarşik Füzyon | 5 | Tablo 3 |
| Uzamsal-Zamansal Tek Modlu Yöntemler | 4 | Tablo 4 |
| Genelleme (Generalization) Odaklı Yöntemler | 4 | Tablo 5 |
| Ses (Audio) Deepfake Tespit Yöntemleri | 5 | Tablo 6 |
| Veri Setleri ve Benchmark Çalışmaları | 5 | Tablo 7 |
| **Toplam** | **30** | — |

### 2.3. Literatürdeki Eksiklikler (Gap Analysis)

Tarama sonucunda projenin doldurmayı hedeflediği boşluklar:

- **Üç modalitenin (uzamsal + işitsel + frekans) eşzamanlı işlenmesi** literatürde çok azdır; mevcut multimodal çalışmalar genellikle yalnızca ses+görüntü ikilisini kullanmaktadır.
- **Hiyerarşik (shallow-attention + deep-modulation) füzyon** stratejisinin ses modalitesi ile birlikte değerlendirilmesi henüz yeterince araştırılmamıştır.
- **Block-wise DCT + Differential Accumulation** yaklaşımının Türkçe akademik çalışmalardaki uygulaması sınırlıdır.

---

## 3. Dataset

### 3.1. Kullanılacak Veri Setleri

| Veri Seti | Modalite | İçerik | Kullanım Amacı | Erişim |
|-----------|----------|--------|----------------|--------|
| **FaceForensics++** | Video (görüntü+ses) | 1000 gerçek + 4000 manipüle (Deepfakes, Face2Face, FaceSwap, NeuralTextures) — c23/c40 sıkıştırma | Birincil eğitim ve in-dataset değerlendirme | Akademik başvuru ([github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)) |
| **Celeb-DF v2** | Video | 590 gerçek + 5639 sahte ünlü videosu | Cross-dataset genelleme testi | Akademik başvuru |
| **DFDC (Deepfake Detection Challenge)** | Video (görüntü+ses) | 100k+ klip, 3426 paid actor | Cross-dataset genelleme + büyük ölçekli test | Kaggle |
| **ASVspoof 2019 LA** | Ses | 121k bona-fide + spoofed konuşma kaydı | Ses modülü eğitimi | [asvspoof.org](https://www.asvspoof.org/) |
| **FakeAVCeleb** | Audio-Visual | 500 gerçek + 19500 sahte (gerçek-ses-sahte-görüntü kombinasyonları) | Çok modlu füzyon doğrulama | Akademik başvuru |

### 3.2. Veri Bölümleme

- **Eğitim:** %70
- **Doğrulama (validation):** %15
- **Test (in-dataset):** %15
- **Cross-dataset test:** Ayrı veri setleri (Celeb-DF v2, DFDC) eğitime hiç dahil edilmez; yalnızca son değerlendirmede kullanılır.

### 3.3. Ön İşleme Pipeline'ı

1. **Video → Kare ayrıştırma:** OpenCV ile saniyede 5 kare (`frame_rate: 5`)
2. **Yüz tespiti ve hizalama:** MTCNN ile 224×224 kırpma (`face_size: 224`)
3. **Ses ayrıştırma:** FFmpeg ile 16 kHz mono PCM (`audio_sample_rate: 16000`)
4. **Mel-spektrogram:** 128-bant, hop=512, n_fft=2048
5. **Frekans dönüşümü:** Block-wise DCT (8×8 blok) + 2D FFT (genlik + faz)
6. **Maksimum kare sayısı:** 150 (`max_frames: 150`) — uzun videolar için temporal sub-sampling

### 3.4. Veri Kullanımı ve Etik

- Tüm veri setleri akademik kullanım anlaşması altında temin edilmiştir.
- Hiçbir kişisel/biyometrik veri yerel olarak yedeklenmez veya 3. tarafla paylaşılmaz.
- Üretilen modeller saldırı amaçlı (yeni deepfake üretimi) değil, **savunma amaçlı (tespit)** kullanım için tasarlanmıştır.

---

## 4. Problem Hangi Sorunu Çözüyor

### 4.1. Toplumsal Sorun

Deepfake içeriklerin yarattığı dezenformasyon, dolandırıcılık ve bireysel mağduriyet vakaları her yıl katlanarak artmaktadır. 2024 itibarıyla:

- Kimlik doğrulama (KYC) sistemlerini hedef alan deepfake saldırılarında **yıllık %3000+ artış** raporlanmıştır.
- Sosyal medyada dolaşan sahte siyasi içerikler seçim süreçlerini etkileme potansiyeli taşımaktadır.
- Türkiye'de KVKK kapsamında, kişinin görüntü/ses verisinin rıza dışı sentezlenmesi suç teşkil etmektedir; ancak otomatik tespit altyapısı henüz yeterli değildir.

### 4.2. Teknik Sorun ve Projenin Katkısı

| Mevcut Sorun | Projenin Çözümü |
|--------------|-----------------|
| Tek modlu modeller, manipülasyon yöntemi değişince başarısız oluyor | **Üç modaliteyi (uzamsal + işitsel + frekans) eşzamanlı** kullanarak farklı manipülasyon tekniklerine karşı dayanıklılık |
| GAN/Difüzyon izleri görsel domainde gizlenebiliyor | **Block-wise DCT + Multi-scale Conv** ile yüksek frekans izlerinin doğrudan tespiti |
| Cross-dataset performans %20-30 düşüyor | **Hierarchical Cross-Modal Fusion** ile genelleme yeteneği artırılır |
| Ağır modeller (>500MB) gerçek zamanlı kullanıma uygun değil | EfficientNet-B4 + hafif fusion katmanı ile **<200MB toplam model boyutu** hedeflenir |

### 4.3. Beklenen Çıktılar

1. **Akademik:** Yüksek lisans tezi + en az 1 ulusal/uluslararası konferans bildirisi
2. **Pratik:** Streamlit tabanlı, son kullanıcının `.mp4` yükleyip "Gerçek/Sahte + Güvenilirlik Skoru" alabildiği web arayüzü
3. **Hedef Metrikler:**
   - FF++ in-dataset: **≥%97 Acc / ≥%99 AUC**
   - Celeb-DF v2 cross-dataset: **≥%80 AUC** (referans çalışmadan +5 puan iyileştirme hedefi)
   - DFDC cross-dataset: **≥%78 AUC**
   - Inference süresi: **≤3 sn / 10 saniyelik video** (RTX 4070 üzerinde)

### 4.4. Kim Faydalanır?

- **Kamu kurumları:** Adli bilişim, Cumhuriyet Savcılıkları, MASAK
- **Medya kuruluşları:** Doğrulama platformları (Teyit.org gibi)
- **Sosyal medya platformları:** Otomatik içerik moderasyonu
- **Bankacılık/Finans:** KYC ve uzaktan kimlik doğrulama
- **Bireysel kullanıcı:** Şüpheli video/ses içeriklerini doğrulama

---

## Ek Belgeler

- [`docs/literature_review.md`](literature_review.md) — Tam literatür tablosu (30 yayın, 7 kategori)
- [`docs/literature_review.csv`](literature_review.csv) — Excel uyumlu CSV
- [`docs/literature_review.tex`](literature_review.tex) — Tez için LaTeX longtable
- [`README.md`](../README.md) — Sistem mimarisi ve kurulum
- [`configs/config.yaml`](../configs/config.yaml) — Model ve eğitim parametreleri
