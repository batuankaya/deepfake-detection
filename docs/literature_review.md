# Literatür Taraması

> **Proje:** Spatial-Frequency Collaborative Learning ve Hierarchical Cross-Modal Fusion Tabanlı Deepfake Tespiti
> **Hazırlayan:** Batuhan Kaya — 232923023
> **Toplam Yayın:** 30 (7 kategori, 2020-2025 arası)

---

## İçindekiler

1. [Çekirdek (Ana Referans) Çalışmalar](#tablo-1-çekirdek-ana-referans-çalışmalar)
2. [Frekans Tabanlı (DCT/FFT) Tespit Yöntemleri](#tablo-2-frekans-tabanlı-dctfft-tespit-yöntemleri)
3. [Çok Modlu (Multimodal) ve Hiyerarşik Füzyon Yöntemleri](#tablo-3-çok-modlu-multimodal-ve-hiyerarşik-füzyon-yöntemleri)
4. [Uzamsal-Zamansal (Spatial-Temporal) Tek Modlu Yöntemler](#tablo-4-uzamsal-zamansal-spatial-temporal-tek-modlu-yöntemler)
5. [Genelleme (Generalization) Odaklı Yöntemler](#tablo-5-genelleme-generalization-odaklı-yöntemler)
6. [Ses (Audio) Deepfake Tespit Yöntemleri](#tablo-6-ses-audio-deepfake-tespit-yöntemleri)
7. [Veri Setleri ve Benchmark Çalışmaları](#tablo-7-veri-setleri-ve-benchmark-çalışmaları)

---

## Tablo 1. Çekirdek (Ana Referans) Çalışmalar

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 1 | Towards Generalizable Deepfake Detection with Spatial-Frequency Collaborative Learning and Hierarchical Cross-Modal Fusion | Mengyu Qiao, Runze Tian | Nisan 2025 | arXiv:2504.17223 (North China University of Technology) | FaceForensics++ (c23, c40), Celeb-DF v2, DFDC | Block-wise DCT + Cascaded Multi-scale Conv (3×3, 5×5, 7×7) + Scale-invariant Differential Accumulation + Hierarchical Cross-Modal Fusion (Shallow Attention + Deep Dynamic Modulation) | FF++ c23: %97.43 Acc / %99.58 AUC; FF++ c40: %91.77 Acc / %94.21 AUC; Cross-data Celeb-DF v2: %74.68 AUC; DFDC: %73.71 AUC |
| 2 | Öz Denetimli Öğrenme Yaklaşımları ile Derin Sahte Ses ve Görüntü Manipülasyonunun Tespiti | Merve Yıldırım | 2025 | YÖK Tez Merkezi (Tez No: 957056), Yüksek Lisans Tezi | FaceForensics++, ASVspoof | Self-Supervised Learning (SSL) + Wav2Vec/HuBERT tabanlı ses analizi + ViT tabanlı görüntü analizi | Tez içerisinde belirtilmiş (Türkçe akademik referans) |

---

## Tablo 2. Frekans Tabanlı (DCT/FFT) Tespit Yöntemleri

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 3 | Leveraging Frequency Analysis for Deep Fake Image Recognition | J. Frank, T. Eisenhofer, L. Schönherr, A. Fischer, D. Kolossa, T. Holz | 2020 | ICML 2020 | CelebA, FFHQ, LSUN-Bedroom + StyleGAN/GANFingerprints | DCT spektrumu üzerinde basit CNN + Logistic Regression + KNN | %100'e yakın doğruluk (StyleGAN/CelebA); rastgele indirme/sıkıştırmaya dayanıklı |
| 4 | Fighting Deepfakes by Detecting GAN DCT Anomalies | O. Giudice, L. Guarnera, S. Battiato | 2021 | Journal of Imaging (MDPI) — arXiv:2101.09781 | CelebA + StarGAN, GDWCT, AttGAN, StyleGAN, StyleGAN2 | β-AC katsayıları ile GAN-Specific Frequencies (GSF) çıkarımı + SVM/k-NN | Sınıflandırma doğruluğu >%99; JPEG/döndürme/ölçekleme robustluğu state-of-the-art üstü |
| 5 | FreqNet: Frequency-Aware Deepfake Detection — Improving Generalizability through Frequency Space Learning | C. Tan, Y. Zhao, S. Wei, G. Gu, P. Liu, Y. Wei | Mart 2024 | AAAI 2024 — arXiv:2403.07240 | 17 farklı GAN (ProGAN, StyleGAN, BigGAN, vb.) + ForenSynths | FFT/iFFT arasında phase + amplitude spectra üzerinde conv (high-frequency vurgu) | Baseline'a göre +%9.8 ortalama performans; daha az parametre |
| 6 | CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features | Ahmed K. vd. | Şubat 2025 | arXiv:2502.10682 | FaceForensics++, Celeb-DF, DFDC + diffusion görüntüleri | Block-wise DCT + cross-band/cross-block conv + Spatial CNN + Attention | GAN: %97.7 Acc; Diffusion: %73 Acc |
| 7 | SFIAD: Deepfake detection through spatial-frequency feature integration and dynamic margin optimization | Çoklu yazar | 2025 | Artificial Intelligence Review (Springer) | FaceForensics++, Celeb-DF | Spatial + Frequency feature integration + Dynamic Margin Loss | Cross-dataset generalization'da rakiplerini aşıyor |

---

## Tablo 3. Çok Modlu (Multimodal) ve Hiyerarşik Füzyon Yöntemleri

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 8 | HOLA: Enhancing Audio-Visual Deepfake Detection via Hierarchical Contextual Aggregations and Efficient Pre-training | X. Wu, D. Huang, H. Sun, X. Yin, Y. Wang, H. Wang, J. Zhang, F. Wang vd. | Temmuz 2025 | arXiv:2507.22781 | 1.81M örneklik self-built dataset + 1M-Deepfakes Detection Challenge | Iterative-Aware Cross-Modal Learning + Hierarchical Contextual Modeling + Pyramid Refiner | 1M-Deepfakes Challenge Video-Level 1.lik; TestA'da +0.0476 AUC |
| 9 | HFMF: Hierarchical Fusion Meets Multi-Stream Models for Deepfake Detection | A. Mehta, B. McArthur, N. Kolloju, Z. Tu | Ocak 2025 | WACV 2025 Workshops (AI4MFDD) | FaceForensics++, Celeb-DF, DFDC | Multi-stream CNN + ViT + Hiyerarşik Füzyon | Single-stream baseline'ları ortalama %3-7 AUC ile geçiyor |
| 10 | Enhancing multimodal deepfake detection with local-global feature integration and diffusion models | Çoklu yazar | 2025 | Signal, Image and Video Processing (Springer) | FakeAVCeleb, DFDC | CNN (lokal) + Vision Transformer (global) + Audio modülü + Diffusion-aware features | FakeAVCeleb >%95 doğruluk |
| 11 | Modality-Agnostic Deepfakes Detection | Çoklu yazar | 2025 | ACM Workshop on Information Hiding and Multimedia Security 2025 | FakeAVCeleb, DFDC | Audio-Visual Speech Recognition (ön-görev) + missing modality handling | Tek modalite eksik durumlarda dahi >%90 AUC |
| 12 | A Comprehensive Survey of DeepFake Generation and Detection Techniques in Audio-Visual Media | Çoklu yazar | 2025 | ICCK Journal of AI Patterns | — (survey) | Audio + Visual + Text üçlü inceleme | Survey/karşılaştırma tablosu |

---

## Tablo 4. Uzamsal-Zamansal (Spatial-Temporal) Tek Modlu Yöntemler

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 13 | Video deepfake detection using a hybrid CNN-LSTM-Transformer model for identity verification | Çoklu yazar | 2024 | Multimedia Tools and Applications (Springer) | FaceForensics++, DFDC | EfficientNet + LSTM + Transformer + 3DMM yüz özellikleri | Image-based: %95 Acc; Video-based: %87 Acc |
| 14 | Video Deepfake Detection Using EfficientNet and LSTM | Çoklu yazar | Mart 2025 | IRJIET Vol-9 Issue-3 | FaceForensics++ alt-küme | EfficientNet-B0 + LSTM | ~%93 doğruluk |
| 15 | LightFakeDetect: A Lightweight Model for Deepfake Detection in Videos That Focuses on Facial Regions | Çoklu yazar | 2025 | Mathematics (MDPI) | Celeb-DF, FF++ | MobileNet + bölgesel attention | ~%91 Acc, düşük FLOPS |
| 16 | An Investigation into the Utilisation of CNN with LSTM for Video Deepfake Detection | Çoklu yazar | 2024 | Applied Sciences (MDPI) Vol.14 | DFDC, FF++ | ResNet/Xception + LSTM | %88-%92 Acc aralığı |

---

## Tablo 5. Genelleme (Generalization) Odaklı Yöntemler

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 17 | Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection (LSA) | Z. Yan, Y. Luo, S. Lyu, Q. Liu, B. Wu | Haziran 2024 | CVPR 2024 | FF++ (eğitim), Celeb-DF, DFDC, WildDeepfake (cross-eval) | Latent space augmentation + forgery feature enlargement | Cross-dataset AUC'da baseline'lara göre %3-7 iyileşme |
| 18 | DiffusionFake: Enhancing Generalization in Deepfake Detection via Guided Stable Diffusion | Çoklu yazar | Aralık 2024 | NeurIPS 2024 | FF++ (eğitim), Celeb-DF, DFDC, DFD (test) | Stable Diffusion ile reverse generation + hybrid feature extraction | Cross-dataset Celeb-DF AUC: ~%85; DFDC: ~%78 |
| 19 | Deepfake Detection that Generalizes Across Benchmarks | Çoklu yazar | Ağustos 2025 | arXiv:2508.06248 | FF++, Celeb-DF, DFDC, DFDCP, WildDeepfake | Benchmark-invariant representation learning | Cross-benchmark ortalama AUC %80+ |
| 20 | Towards Generalizable Deepfake Detection by Primary Region Regularization | Çoklu yazar | 2024 | ACM TOMM | FF++, Celeb-DF | Primary face region regularization + attention | Cross-data generalization'da %4-6 AUC kazancı |

---

## Tablo 6. Ses (Audio) Deepfake Tespit Yöntemleri

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 21 | Deepfake Audio Detection Using Spectrogram-based Feature and Ensemble of Deep Learning Models | Çoklu yazar | Temmuz 2024 | arXiv:2407.01777 | ASVspoof 2019 LA | Mel-spectrogram + CNN ensemble (ResNet, EfficientNet) | EER: 0.03; AUC: 0.994 (Top-3 ASVspoof 2019) |
| 22 | Deepfake audio detection with spectral features and ResNeXt-based architecture | Çoklu yazar | 2025 | Knowledge-Based Systems (Elsevier) | ASVspoof 2019, ASVspoof 2021 | MFCC + Mel-spectrogram + ResNeXt | ASVspoof 2019: EER %2.1; ASVspoof 2021: EER %5.4 |
| 23 | Hybrid CNN-LSTM Architectures for Deepfake Audio Detection Using MFCC and Spectrogram Analysis | Çoklu yazar | 2025 | American Journal of Math. and Comp. Modelling | ASVspoof 2019, FoR | CNN + LSTM (MFCC + Mel) | %93+ doğruluk |
| 24 | Quantum Vision Theory Applied to Audio Classification for Deepfake Speech Detection | Çoklu yazar | 2025 | arXiv:2604.08104 | ASVspoof 2019 | Quantum-inspired CNN + Mel-spectrogram | %94.57 Acc; EER %9.04 |
| 25 | Audio Deepfake Detection via a Fuzzy Dual-Path Time-Frequency Attention Network | Çoklu yazar | 2025 | PMC (open-access) | ASVspoof 2019, In-the-wild | Fuzzy attention + dual-path time-frequency | EER %1.85 (ASVspoof 2019) |

---

## Tablo 7. Veri Setleri ve Benchmark Çalışmaları

| # | Yayının Adı | Yazarlar | Tarih | Yayınlandığı Yer | Kullanılan Dataset | Uygulanan Yöntem | Başarı Metrikleri |
|---|---|---|---|---|---|---|---|
| 26 | Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics | Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu | Haziran 2020 | CVPR 2020 | Celeb-DF (590 gerçek + 5639 sahte video) | Yeni veri seti; Xception, MesoNet baseline'ları | Tespit edicilerin Celeb-DF AUC'leri ~%65-75 (FF++'da %95+) |
| 27 | Celeb-DF++: A Large-scale Challenging Video DeepFake Benchmark for Generalizable Forensics | Çoklu yazar | Temmuz 2025 | arXiv:2507.18015 | Celeb-DF v2 + yeni manipülasyonlar | Benchmark + cross-method test | SOTA AUC'leri %60-78 aralığına düşüyor |
| 28 | DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection | Z. Yan, Y. Zhang, X. Yuan, S. Lyu, B. Wu | Aralık 2023 | NeurIPS 2023 (Datasets & Benchmarks) | FF++, DFD, FaceShifter, DFDC, DFDCP, Celeb-DF v1/v2, UADFV | 15+ tespit modelinin standart kıyaslaması | Cross-dataset AUC karşılaştırma tablosu |
| 29 | Inclusion 2024: Global Multimedia Deepfake Detection Challenge | Challenge Organizatörleri | Aralık 2024 | arXiv:2412.20833 | Multi-dimensional Face Forgery Dataset | Yarışma; çok boyutlu sahtecilik | Top-3 takım Acc >%95 |
| 30 | A Contemporary Survey on Deepfake Detection: Datasets, Algorithms, and Challenges | Çoklu yazar | 2024 | Electronics (MDPI) Vol.13 | — (survey) | Tüm yaklaşımların taksonomisi | ViT cross-dataset düşüş %11.33; CNN düşüş >%15 |

---

## Notlar

- "Çoklu yazar" olarak işaretlenenler için orijinal yayın sayfasından tam yazar listesi teyit edilmelidir.
- Başarı metrikleri orijinal makalelerin raporlarından alınmıştır.
- Toplam 30 yayın 7 kategoride sınıflandırılmıştır (2020-2025 dönemi).
