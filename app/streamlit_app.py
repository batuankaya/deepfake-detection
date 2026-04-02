"""Deepfake Tespit Sistemi - Streamlit Arayuzu.

Spatial-Frequency Collaborative Learning ve
Hierarchical Cross-Modal Fusion tabanli deepfake tespiti.
"""

import streamlit as st
import tempfile
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Deepfake Tespit Sistemi",
    page_icon="🔍",
    layout="wide",
)

st.title("Deepfake Tespit Sistemi")
st.markdown(
    "Spatial-Frequency Collaborative Learning ve "
    "Hierarchical Cross-Modal Fusion Tabanli Deepfake Tespiti"
)
st.markdown("---")

# Video yukleme
uploaded_file = st.file_uploader(
    "Analiz etmek istediginiz video dosyasini yukleyin (.mp4)",
    type=["mp4"],
)

if uploaded_file is not None:
    # Gecici dosyaya kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Video onizleme
    st.video(uploaded_file)

    if st.button("Analizi Baslat", type="primary"):
        st.markdown("---")

        from src.inference import DeepfakeDetector
        from src.utils.visualize import plot_fft_analysis, plot_dct_analysis

        progress = st.progress(0, text="Video isleniyor...")

        # Dedektoru olustur
        detector = DeepfakeDetector()

        # ===== ADIM 1: Video karelerini cikar =====
        progress.progress(10, text="Videodan kareler cikariliyor...")
        results = detector.analyze_video(tmp_path)

        if "error" in results:
            st.error(results["error"])
        else:
            progress.progress(30, text="Kareler cikarildi!")

            # Ornek kareleri goster
            st.subheader(f"Video Analizi ({results['frames_analyzed']} kare cikarildi)")
            if results["sample_frames"]:
                fcols = st.columns(len(results["sample_frames"]))
                labels = ["Baslangic", "Orta", "Son"]
                for i, (col, frame) in enumerate(zip(fcols, results["sample_frames"])):
                    with col:
                        st.image(frame, caption=f"{labels[i]} kare",
                                 use_container_width=True)

            st.markdown("---")

            # ===== ADIM 2: Modul bazli analiz =====
            col1, col2, col3 = st.columns(3)

            # Modul 1: Goruntu Analizi
            with col1:
                st.subheader("Uzamsal Analiz")
                st.caption("EfficientNet-B4 + LSTM")
                progress.progress(40, text="Goruntu analizi...")
                st.info("Model henuz egitilmedi.\n\n"
                        "Egitim sonrasi yuz hareketlerindeki "
                        "tutarsizliklar burada gosterilecektir.")

            # Modul 2: Ses Analizi
            with col2:
                st.subheader("Isitsel Analiz")
                st.caption("Mel-Spektrogram CNN")
                progress.progress(50, text="Ses analizi...")
                st.info("Model henuz egitilmedi.\n\n"
                        "Egitim sonrasi ses frekanslarindaki "
                        "yapayliklar burada gosterilecektir.")

            # Modul 3: Frekans Analizi (CALISIYOR!)
            with col3:
                st.subheader("Frekans Analizi")
                st.caption("Block-wise DCT + Multi-scale CNN")
                progress.progress(60, text="Frekans analizi yapiliyor...")
                freq = results["frequency_analysis"]
                st.metric("Yuksek Frekans Enerjisi",
                          f"{freq['high_freq_energy']:.4f}")
                st.metric("Dusuk Frekans Enerjisi",
                          f"{freq['low_freq_energy']:.4f}")
                st.metric("Frekans Orani (Y/D)",
                          f"{freq['freq_ratio']:.4f}")

            st.markdown("---")

            # ===== ADIM 3: FFT Gorsellestirme =====
            progress.progress(70, text="FFT haritasi olusturuluyor...")
            st.subheader("Fourier Frekans Haritasi")

            mid_frame = results["sample_frames"][1] if len(results["sample_frames"]) > 1 else results["sample_frames"][0]

            tab1, tab2 = st.tabs(["FFT Analizi", "DCT Analizi"])

            with tab1:
                fig_fft = plot_fft_analysis(mid_frame)
                st.pyplot(fig_fft)

            with tab2:
                try:
                    fig_dct = plot_dct_analysis(mid_frame)
                    st.pyplot(fig_dct)
                except Exception as e:
                    st.warning(f"DCT analizi icin scipy gerekli: {e}")

            st.markdown("---")

            # ===== ADIM 4: Istatistiksel Ozet =====
            progress.progress(85, text="Istatistikler hesaplaniyor...")
            st.subheader("Frekans Istatistikleri (Tum Kareler)")

            scol1, scol2 = st.columns(2)
            with scol1:
                st.metric("Ortalama Frekans Orani",
                          f"{results['avg_freq_ratio']:.4f}")
            with scol2:
                st.metric("Standart Sapma",
                          f"{results['std_freq_ratio']:.4f}")

            st.caption("Not: Yuksek frekans orani ve dusuk standart sapma, "
                       "yapay icerik gostergesi olabilir. "
                       "Kesin sonuc icin modellerin egitilmesi gerekmektedir.")

            st.markdown("---")

            # ===== ADIM 5: Fuzyon =====
            progress.progress(95, text="Hierarchical Fusion...")
            st.subheader("Hierarchical Cross-Modal Fusion")

            fcol1, fcol2 = st.columns(2)
            with fcol1:
                st.markdown("**Asama 1:** Shallow-layer Attention Enhancement")
                st.caption("Spatial ↔ Frequency capraz dikkat")
            with fcol2:
                st.markdown("**Asama 2:** Deep-layer Dynamic Modulation")
                st.caption("Modalite guvenilirlik agirlandirma")

            st.warning("Nihai fuzyon skoru icin tum modullerin "
                       "egitilmis olmasi gerekmektedir.")

            progress.progress(100, text="Analiz tamamlandi!")

    # Temizlik
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

else:
    st.info("Lutfen bir .mp4 video dosyasi yukleyin.")

# Sidebar bilgi
with st.sidebar:
    st.header("Hakkinda")
    st.markdown("""
    **Proje:** Deepfake Tespit Sistemi
    **Gelistirici:** Batuhan Kaya (232923023)

    **Analiz Modulleri:**
    1. Uzamsal Analiz (EfficientNet + LSTM)
    2. Isitsel Analiz (Mel-Spektrogram CNN)
    3. Frekans Analizi (Block-wise DCT + Multi-scale)
    4. Hierarchical Cross-Modal Fusion
    """)

    st.markdown("---")
    st.header("Durum")
    st.markdown("""
    | Modul | Durum |
    |-------|-------|
    | Frekans Analizi | ✅ Aktif |
    | Uzamsal Analiz | ⏳ Egitim bekleniyor |
    | Isitsel Analiz | ⏳ Egitim bekleniyor |
    | Fuzyon | ⏳ Egitim bekleniyor |
    """)

    st.markdown("---")
    st.header("Referanslar")
    st.markdown("""
    **[1]** Towards Generalizable Deepfake Detection with
    Spatial-Frequency Collaborative Learning and
    Hierarchical Cross-Modal Fusion (2025)
    [arxiv.org/abs/2504.17223](https://arxiv.org/abs/2504.17223)

    **[2]** Oz Denetimli Ogrenme Yaklasimlari ile Derin
    Sahte Ses ve Goruntu Manipulasyonunun Tespiti
    (Merve Yildirim, 2025) - YOK Tez No: 957056
    """)
