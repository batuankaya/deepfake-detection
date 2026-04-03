"""Deepfake Tespit Sistemi - Analiz Arayuzu."""

import streamlit as st
import tempfile
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALLOWED_EXTENSIONS = {".mp4"}
MAX_FILE_SIZE_MB = 200

st.set_page_config(page_title="Deepfake Tespit", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("Deepfake Tespit")
    st.caption("Batuhan Kaya - 232923023")
    st.divider()

    st.subheader("Modul Durumu")
    st.progress(25, text="1/4 modul aktif")

    with st.container(border=True):
        st.markdown(":green[**Frekans Analizi**] — Aktif")
        st.markdown(":orange[**Uzamsal Analiz**] — Egitim bekleniyor")
        st.markdown(":orange[**Isitsel Analiz**] — Egitim bekleniyor")
        st.markdown(":orange[**Cross-Modal Fuzyon**] — Egitim bekleniyor")

    st.divider()
    with st.expander("Teknik Detaylar"):
        st.markdown("""
        **Goruntu:** EfficientNet-B4 + BiLSTM

        **Ses:** Mel-Spektrogram + CNN

        **Frekans:** Block-wise DCT, 2D-FFT,
        Multi-scale Convolution

        **Fuzyon:** Hierarchical Cross-Modal
        Attention + Dynamic Modulation
        """)

    with st.expander("Kaynaklar"):
        st.caption(
            "[1] Spatial-Frequency Collaborative Learning "
            "and Hierarchical Cross-Modal Fusion, 2025 — "
            "[arxiv](https://arxiv.org/abs/2504.17223)"
        )
        st.caption(
            "[2] Oz Denetimli Ogrenme ile Derin Sahte "
            "Ses ve Goruntu Tespiti, Yildirim 2025 — "
            "YOK Tez No: 957056"
        )


# --- Ana sayfa ---
st.markdown("### Video Analizi")
st.markdown("Bir video yukleyin, sistem frekans uzayinda analiz yaparak "
            "manipulasyon izlerini arayacaktir.")

uploaded_file = st.file_uploader(
    "Video sec",
    type=["mp4"],
    help="Maksimum 200MB, .mp4",
    label_visibility="collapsed",
)

if uploaded_file is None:
    # Bos durum
    with st.container(border=True):
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.markdown("")
            st.markdown(
                "**Yukleme bekleniyor**  \n"
                "Analiz icin .mp4 formatinda bir video secin."
            )
            st.markdown("")

else:
    file_ext = Path(uploaded_file.name).suffix.lower()
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

    if file_ext not in ALLOWED_EXTENSIONS:
        st.error("Sadece .mp4 dosyalari desteklenmektedir.")
    elif file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"Dosya boyutu {file_size_mb:.1f}MB. Maksimum {MAX_FILE_SIZE_MB}MB.")
    else:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Video onizleme
            with st.container(border=True):
                vid_col, info_col = st.columns([2, 1])
                with vid_col:
                    st.video(uploaded_file)
                with info_col:
                    st.markdown(f"**Dosya:** {uploaded_file.name}")
                    st.markdown(f"**Boyut:** {file_size_mb:.1f} MB")
                    st.markdown("")
                    st.button("Analizi Baslat", type="primary",
                              key="analyze_btn", use_container_width=True)

            if st.session_state.get("analyze_btn"):
                from src.inference import DeepfakeDetector
                from src.utils.visualize import plot_fft_analysis, plot_dct_analysis
                import matplotlib
                matplotlib.use("Agg")

                status = st.status("Analiz calisiyor...", expanded=True)

                with status:
                    st.write("Kareler cikariliyor...")
                    detector = DeepfakeDetector()
                    results = detector.analyze_video(tmp_path)

                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.write(f"{results['frames_analyzed']} kare islendi.")
                        st.write("Frekans analizi tamamlandi.")

                status.update(label="Analiz tamamlandi", state="complete")

                if "error" not in results:
                    # --- Cikarilan kareler ---
                    st.markdown("#### Cikarilan Kareler")
                    if results["sample_frames"]:
                        frame_cols = st.columns(3)
                        captions = ["Baslangic", "Orta", "Son"]
                        for i, (col, frame) in enumerate(
                            zip(frame_cols, results["sample_frames"])
                        ):
                            col.image(frame, caption=captions[i],
                                      use_container_width=True)

                    st.divider()

                    # --- Frekans metrikleri ---
                    st.markdown("#### Frekans Analizi Sonuclari")

                    freq = results["frequency_analysis"]

                    with st.container(border=True):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Yuksek Frekans", f"{freq['high_freq_energy']:.4f}")
                        m2.metric("Dusuk Frekans", f"{freq['low_freq_energy']:.4f}")
                        m3.metric("Y/D Orani", f"{freq['freq_ratio']:.4f}")
                        m4.metric("Ortalama (Tum Kareler)",
                                  f"{results['avg_freq_ratio']:.4f}")

                    st.divider()

                    # --- Spektrum grafikleri ---
                    st.markdown("#### Frekans Haritalari")

                    mid_frame = (
                        results["sample_frames"][1]
                        if len(results["sample_frames"]) > 1
                        else results["sample_frames"][0]
                    )

                    tab_fft, tab_dct, tab_profile = st.tabs([
                        "FFT Spektrum", "DCT Haritasi", "Frekans Profili"
                    ])

                    with tab_fft:
                        fig_fft = plot_fft_analysis(mid_frame)
                        st.pyplot(fig_fft)

                    with tab_dct:
                        try:
                            fig_dct = plot_dct_analysis(mid_frame)
                            st.pyplot(fig_dct)
                        except ImportError:
                            st.warning("DCT analizi icin scipy kutuphanesi gereklidir.")
                        except Exception:
                            st.warning("DCT analizi sirasinda bir hata olustu.")

                    with tab_profile:
                        # Azimutal profili ayri ciz
                        azimuthal = freq.get("azimuthal_profile")
                        if azimuthal is None:
                            from src.models.frequency.frequency_model import FrequencyAnalyzer
                            gray = np.mean(mid_frame, axis=2) if len(mid_frame.shape) == 3 else mid_frame
                            spectrum = FrequencyAnalyzer.compute_fft_spectrum(gray)
                            azimuthal = FrequencyAnalyzer.compute_azimuthal_average(spectrum)

                        st.line_chart(
                            {"Frekans Genlik Profili": azimuthal},
                            x_label="Frekans (piksel)",
                            y_label="Ortalama Genlik",
                        )
                        st.caption(
                            "Yuksek frekans bolgelerindeki anormal tepeler, "
                            "GAN/difuzyon tabanli uretim izlerine isaret edebilir."
                        )

                    st.divider()

                    # --- Sonuc ---
                    st.markdown("#### Degerlendirme")

                    with st.container(border=True):
                        r1, r2 = st.columns([1, 2])
                        with r1:
                            st.metric("Std. Sapma", f"{results['std_freq_ratio']:.4f}")
                        with r2:
                            st.info(
                                "Frekans analizi tamamlandi. Nihai gercek/sahte karari "
                                "icin uzamsal ve isitsel modullerin egitilmesi, ardindan "
                                "cross-modal fuzyon katmanina baglanmasi gerekiyor."
                            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
