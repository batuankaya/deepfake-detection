## Vahşi-Doğa Testi (Eğitim-Dışı Kaynaklar)

Rastgele seçilmiş **100 video / 5 kaynak** (seed=7, n_per_source=20). Bu videoların hiçbiri eğitim veya test setinde değildi.

**Genel doğruluk: %64.0**  (Gerçek 37/40, Sahte 27/60)

### Kaynak başına özet

| Kaynak | n | Doğru | Doğruluk | Ort. spatial | Ort. freq |
|---|:---:|:---:|:---:|:---:|:---:|
| FF++ original (raw, eğitim-dışı ID) | 20 | 20/20 | **%100.0** | 0.003 | 0.246 |
| FF++ DeepFakeDetection (Google DFD — train'de yok) | 20 | 14/20 | **%70.0** | 0.758 | 0.776 |
| FF++ FaceShifter (eğitim-dışı manipülasyon) | 20 | 1/20 | **%5.0** | 0.131 | 0.479 |
| Celeb-DF v2 Celeb-real (farklı dağılım) | 20 | 17/20 | **%85.0** | 0.342 | 0.377 |
| Celeb-DF v2 Celeb-synthesis (farklı dağılım) | 20 | 12/20 | **%60.0** | 0.760 | 0.673 |

### Per-video sonuçlar

| # | Kaynak | Video | Gerçek | Tahmin | Fake olas. | Spatial | Freq | s |
|---|--------|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | FF++ original (raw, eğitim-dışı ID) | `331.mp4` | GERÇEK | **REAL** ✓ | 0.010 | 0.001 | 0.034 | 6.28 |
| 2 | FF++ original (raw, eğitim-dışı ID) | `970.mp4` | GERÇEK | **REAL** ✓ | 0.019 | 0.000 | 0.067 | 2.88 |
| 3 | FF++ original (raw, eğitim-dışı ID) | `154.mp4` | GERÇEK | **REAL** ✓ | 0.212 | 0.001 | 0.721 | 6.14 |
| 4 | FF++ original (raw, eğitim-dışı ID) | `404.mp4` | GERÇEK | **REAL** ✓ | 0.135 | 0.001 | 0.474 | 6.07 |
| 5 | FF++ original (raw, eğitim-dışı ID) | `666.mp4` | GERÇEK | **REAL** ✓ | 0.021 | 0.001 | 0.072 | 2.93 |
| 6 | FF++ original (raw, eğitim-dışı ID) | `049.mp4` | GERÇEK | **REAL** ✓ | 0.006 | 0.001 | 0.020 | 2.7 |
| 7 | FF++ original (raw, eğitim-dışı ID) | `074.mp4` | GERÇEK | **REAL** ✓ | 0.065 | 0.001 | 0.227 | 9.26 |
| 8 | FF++ original (raw, eğitim-dışı ID) | `840.mp4` | GERÇEK | **REAL** ✓ | 0.032 | 0.001 | 0.112 | 2.86 |
| 9 | FF++ original (raw, eğitim-dışı ID) | `548.mp4` | GERÇEK | **REAL** ✓ | 0.144 | 0.027 | 0.471 | 5.03 |
| 10 | FF++ original (raw, eğitim-dışı ID) | `096.mp4` | GERÇEK | **REAL** ✓ | 0.030 | 0.001 | 0.106 | 11.07 |
| 11 | FF++ original (raw, eğitim-dışı ID) | `374.mp4` | GERÇEK | **REAL** ✓ | 0.076 | 0.001 | 0.268 | 4.75 |
| 12 | FF++ original (raw, eğitim-dışı ID) | `596.mp4` | GERÇEK | **REAL** ✓ | 0.013 | 0.001 | 0.046 | 2.8 |
| 13 | FF++ original (raw, eğitim-dışı ID) | `059.mp4` | GERÇEK | **REAL** ✓ | 0.136 | 0.001 | 0.477 | 2.55 |
| 14 | FF++ original (raw, eğitim-dışı ID) | `931.mp4` | GERÇEK | **REAL** ✓ | 0.056 | 0.001 | 0.195 | 2.8 |
| 15 | FF++ original (raw, eğitim-dışı ID) | `519.mp4` | GERÇEK | **REAL** ✓ | 0.040 | 0.001 | 0.140 | 2.71 |
| 16 | FF++ original (raw, eğitim-dışı ID) | `219.mp4` | GERÇEK | **REAL** ✓ | 0.069 | 0.001 | 0.242 | 5.71 |
| 17 | FF++ original (raw, eğitim-dışı ID) | `038.mp4` | GERÇEK | **REAL** ✓ | 0.130 | 0.015 | 0.438 | 2.57 |
| 18 | FF++ original (raw, eğitim-dışı ID) | `088.mp4` | GERÇEK | **REAL** ✓ | 0.044 | 0.001 | 0.154 | 3.04 |
| 19 | FF++ original (raw, eğitim-dışı ID) | `444.mp4` | GERÇEK | **REAL** ✓ | 0.103 | 0.001 | 0.362 | 4.81 |
| 20 | FF++ original (raw, eğitim-dışı ID) | `428.mp4` | GERÇEK | **REAL** ✓ | 0.082 | 0.001 | 0.286 | 5.2 |
| 21 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `02_09__talking_angry_couch__6KUOFMZW.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.993 | 6.45 |
| 22 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `06_14__walking_and_outside_surprised__8U9ULZDT.mp4` | SAHTE | **FAKE** ✓ | 0.905 | 0.959 | 0.906 | 8.01 |
| 23 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `02_15__kitchen_still__XNJIS4S3.mp4` | SAHTE | **FAKE** ✓ | 0.998 | 0.999 | 0.784 | 7.13 |
| 24 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `14_26__podium_speech_happy__ILLP29ZH.mp4` | SAHTE | **FAKE** ✓ | 0.999 | 1.000 | 0.904 | 6.42 |
| 25 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `12_15__kitchen_pan__N0SRODQD.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.973 | 7.6 |
| 26 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `02_07__meeting_serious__1JCLEEBQ.mp4` | SAHTE | **REAL** ✗ | 0.252 | 0.143 | 0.694 | 7.01 |
| 27 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `23_19__walking_and_outside_surprised__H4SUVFTL.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.931 | 7.93 |
| 28 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `15_03__exit_phone_room__GZNLO4X9.mp4` | SAHTE | **FAKE** ✓ | 0.910 | 0.094 | 0.947 | 6.42 |
| 29 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `03_06__hugging_happy__2XWC5JED.mp4` | SAHTE | **REAL** ✗ | 0.382 | 0.612 | 0.522 | 6.43 |
| 30 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `27_03__walking_and_outside_surprised__KWGI1LSZ.mp4` | SAHTE | **REAL** ✗ | 0.406 | 0.763 | 0.404 | 7.67 |
| 31 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `06_03__talking_against_wall__4I8LRXWF.mp4` | SAHTE | **REAL** ✗ | 0.289 | 0.632 | 0.165 | 6.73 |
| 32 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `18_02__exit_phone_room__B95S4G6F.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.958 | 6.37 |
| 33 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `17_16__walk_down_hall_angry__EWJJTZHY.mp4` | SAHTE | **FAKE** ✓ | 0.894 | 0.599 | 0.937 | 2.77 |
| 34 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `15_06__walking_down_street_outside_angry__YDX3GHFS.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.982 | 13.27 |
| 35 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `02_07__walk_down_hall_angry__U7DEOZNV.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.961 | 4.06 |
| 36 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `15_04__walking_and_outside_surprised__46TJ9IOJ.mp4` | SAHTE | **REAL** ✗ | 0.350 | 0.358 | 0.723 | 8.72 |
| 37 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `15_07__podium_speech_happy__KM6M2TBT.mp4` | SAHTE | **FAKE** ✓ | 0.999 | 1.000 | 0.982 | 7.21 |
| 38 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `11_13__outside_talking_still_laughing__61T622EK.mp4` | SAHTE | **REAL** ✗ | 0.033 | 0.002 | 0.113 | 7.02 |
| 39 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `02_06__podium_speech_happy__N8OSN8P6.mp4` | SAHTE | **FAKE** ✓ | 0.999 | 1.000 | 0.772 | 6.38 |
| 40 | FF++ DeepFakeDetection (Google DFD — train'de yok) | `28_16__walking_down_street_outside_angry__U757192N.mp4` | SAHTE | **FAKE** ✓ | 0.998 | 0.999 | 0.875 | 8.02 |
| 41 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `226_491.mp4` | SAHTE | **REAL** ✗ | 0.285 | 0.206 | 0.712 | 4.33 |
| 42 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `047_862.mp4` | SAHTE | **FAKE** ✓ | 0.864 | 0.872 | 0.919 | 3.37 |
| 43 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `570_624.mp4` | SAHTE | **REAL** ✗ | 0.335 | 0.370 | 0.682 | 5.22 |
| 44 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `879_963.mp4` | SAHTE | **REAL** ✗ | 0.171 | 0.004 | 0.596 | 4.02 |
| 45 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `136_285.mp4` | SAHTE | **REAL** ✗ | 0.115 | 0.001 | 0.403 | 5.24 |
| 46 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `296_293.mp4` | SAHTE | **REAL** ✗ | 0.215 | 0.050 | 0.690 | 3.12 |
| 47 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `429_404.mp4` | SAHTE | **REAL** ✗ | 0.111 | 0.002 | 0.388 | 2.6 |
| 48 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `147_055.mp4` | SAHTE | **REAL** ✗ | 0.191 | 0.001 | 0.669 | 2.64 |
| 49 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `553_545.mp4` | SAHTE | **REAL** ✗ | 0.080 | 0.006 | 0.272 | 2.52 |
| 50 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `120_118.mp4` | SAHTE | **REAL** ✗ | 0.177 | 0.001 | 0.622 | 2.62 |
| 51 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `584_823.mp4` | SAHTE | **REAL** ✗ | 0.023 | 0.001 | 0.082 | 5.39 |
| 52 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `315_322.mp4` | SAHTE | **REAL** ✗ | 0.431 | 0.425 | 0.805 | 3.02 |
| 53 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `573_757.mp4` | SAHTE | **REAL** ✗ | 0.230 | 0.192 | 0.551 | 5.21 |
| 54 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `835_651.mp4` | SAHTE | **REAL** ✗ | 0.039 | 0.002 | 0.136 | 10.67 |
| 55 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `698_693.mp4` | SAHTE | **REAL** ✗ | 0.170 | 0.388 | 0.077 | 2.71 |
| 56 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `185_276.mp4` | SAHTE | **REAL** ✗ | 0.051 | 0.001 | 0.177 | 4.87 |
| 57 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `105_180.mp4` | SAHTE | **REAL** ✗ | 0.292 | 0.038 | 0.818 | 2.35 |
| 58 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `595_597.mp4` | SAHTE | **REAL** ✗ | 0.116 | 0.030 | 0.368 | 5.21 |
| 59 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `654_648.mp4` | SAHTE | **REAL** ✗ | 0.144 | 0.001 | 0.505 | 2.59 |
| 60 | FF++ FaceShifter (eğitim-dışı manipülasyon) | `192_134.mp4` | SAHTE | **REAL** ✗ | 0.049 | 0.040 | 0.117 | 3.74 |
| 61 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id47_0001.mp4` | GERÇEK | **REAL** ✓ | 0.052 | 0.044 | 0.124 | 2.26 |
| 62 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id20_0001.mp4` | GERÇEK | **REAL** ✓ | 0.098 | 0.196 | 0.082 | 2.6 |
| 63 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id7_0000.mp4` | GERÇEK | **REAL** ✓ | 0.266 | 0.009 | 0.796 | 2.22 |
| 64 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id16_0010.mp4` | GERÇEK | **REAL** ✓ | 0.123 | 0.073 | 0.335 | 2.39 |
| 65 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id8_0007.mp4` | GERÇEK | **FAKE** ✗ | 0.972 | 0.988 | 0.652 | 3.2 |
| 66 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id16_0007.mp4` | GERÇEK | **REAL** ✓ | 0.005 | 0.003 | 0.013 | 2.68 |
| 67 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id30_0001.mp4` | GERÇEK | **REAL** ✓ | 0.178 | 0.103 | 0.488 | 4.31 |
| 68 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id58_0008.mp4` | GERÇEK | **REAL** ✓ | 0.251 | 0.105 | 0.720 | 3.2 |
| 69 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id61_0004.mp4` | GERÇEK | **REAL** ✓ | 0.124 | 0.127 | 0.263 | 2.72 |
| 70 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id51_0007.mp4` | GERÇEK | **REAL** ✓ | 0.404 | 0.739 | 0.428 | 2.98 |
| 71 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id40_0002.mp4` | GERÇEK | **REAL** ✓ | 0.010 | 0.003 | 0.031 | 2.45 |
| 72 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id55_0006.mp4` | GERÇEK | **REAL** ✓ | 0.490 | 0.562 | 0.815 | 2.38 |
| 73 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id54_0004.mp4` | GERÇEK | **REAL** ✓ | 0.350 | 0.671 | 0.331 | 2.85 |
| 74 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id46_0000.mp4` | GERÇEK | **FAKE** ✗ | 0.806 | 0.917 | 0.760 | 2.19 |
| 75 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id39_0007.mp4` | GERÇEK | **FAKE** ✗ | 0.857 | 0.938 | 0.523 | 2.92 |
| 76 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id34_0005.mp4` | GERÇEK | **REAL** ✓ | 0.095 | 0.103 | 0.196 | 3.51 |
| 77 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id28_0005.mp4` | GERÇEK | **REAL** ✓ | 0.190 | 0.417 | 0.108 | 2.37 |
| 78 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id34_0000.mp4` | GERÇEK | **REAL** ✓ | 0.292 | 0.515 | 0.335 | 3.0 |
| 79 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id19_0005.mp4` | GERÇEK | **REAL** ✓ | 0.180 | 0.317 | 0.207 | 2.58 |
| 80 | Celeb-DF v2 Celeb-real (farklı dağılım) | `id9_0008.mp4` | GERÇEK | **REAL** ✓ | 0.092 | 0.001 | 0.324 | 3.03 |
| 81 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id30_id32_0008.mp4` | SAHTE | **REAL** ✗ | 0.215 | 0.102 | 0.621 | 2.8 |
| 82 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id48_id41_0002.mp4` | SAHTE | **REAL** ✗ | 0.374 | 0.504 | 0.637 | 2.77 |
| 83 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id45_id41_0006.mp4` | SAHTE | **FAKE** ✓ | 0.979 | 0.961 | 0.988 | 2.16 |
| 84 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id32_id35_0003.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.790 | 3.04 |
| 85 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id3_id35_0008.mp4` | SAHTE | **FAKE** ✓ | 0.952 | 0.979 | 0.831 | 2.45 |
| 86 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id30_id17_0003.mp4` | SAHTE | **FAKE** ✓ | 0.801 | 0.914 | 0.681 | 3.58 |
| 87 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id55_id51_0000.mp4` | SAHTE | **FAKE** ✓ | 0.819 | 0.922 | 0.786 | 2.8 |
| 88 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id19_id27_0008.mp4` | SAHTE | **FAKE** ✓ | 0.948 | 0.978 | 0.959 | 1.99 |
| 89 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id20_id9_0002.mp4` | SAHTE | **REAL** ✗ | 0.252 | 0.366 | 0.392 | 2.03 |
| 90 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id47_id39_0001.mp4` | SAHTE | **FAKE** ✓ | 0.822 | 0.923 | 0.325 | 2.23 |
| 91 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id38_id30_0003.mp4` | SAHTE | **REAL** ✗ | 0.221 | 0.044 | 0.710 | 3.33 |
| 92 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id23_id30_0004.mp4` | SAHTE | **REAL** ✗ | 0.344 | 0.715 | 0.250 | 2.13 |
| 93 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id32_id34_0000.mp4` | SAHTE | **REAL** ✗ | 0.326 | 0.536 | 0.427 | 2.91 |
| 94 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id23_id16_0004.mp4` | SAHTE | **REAL** ✗ | 0.340 | 0.799 | 0.073 | 2.07 |
| 95 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id44_id43_0005.mp4` | SAHTE | **FAKE** ✓ | 0.978 | 0.991 | 0.844 | 2.32 |
| 96 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id38_id33_0008.mp4` | SAHTE | **FAKE** ✓ | 0.979 | 0.991 | 0.797 | 2.8 |
| 97 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id16_id31_0001.mp4` | SAHTE | **REAL** ✗ | 0.459 | 0.750 | 0.605 | 2.57 |
| 98 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id7_id13_0000.mp4` | SAHTE | **FAKE** ✓ | 0.996 | 0.998 | 0.993 | 2.24 |
| 99 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id1_id17_0007.mp4` | SAHTE | **FAKE** ✓ | 1.000 | 1.000 | 0.905 | 2.3 |
| 100 | Celeb-DF v2 Celeb-synthesis (farklı dağılım) | `id4_id6_0001.mp4` | SAHTE | **FAKE** ✓ | 0.728 | 0.735 | 0.839 | 2.76 |

> Ses kanalı bu kaynaklarda yok — aktif modül = 2 (spatial + frequency). 3-modüllü test multimodal bir set (DFDC/FakeAVCeleb) gerektirir; bkz. README *Sınırlamalar*.