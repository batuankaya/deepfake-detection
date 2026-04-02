"""Deepfake Tespit Sistemi - Streamlit Arayuzu.

Spatial-Frequency Collaborative Learning ve
Hierarchical Cross-Modal Fusion tabanli deepfake tespiti.
"""

import streamlit as st
import tempfile
import os

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

        progress = st.progress(0, text="Analiz baslatiliyor...")

        col1, col2, col3 = st.columns(3)

        # Modul 1: Goruntu Analizi
        with col1:
            st.subheader("Uzamsal Analiz")
            st.caption("EfficientNet-B4 + LSTM")
            progress.progress(20, text="Goruntu analizi yapiliyor...")
            # TODO: Gercek model ciktilarini entegre et
            st.info("Modul henuz egitilmedi - entegrasyon bekleniyor")

        # Modul 2: Ses Analizi
        with col2:
            st.subheader("Isitsel Analiz")
            st.caption("Mel-Spektrogram CNN")
            progress.progress(40, text="Ses analizi yapiliyor...")
            # TODO: Gercek model ciktilarini entegre et
            st.info("Modul henuz egitilmedi - entegrasyon bekleniyor")

        # Modul 3: Frekans Analizi
        with col3:
            st.subheader("Frekans Analizi")
            st.caption("Block-wise DCT + Multi-scale CNN")
            progress.progress(60, text="Frekans analizi yapiliyor...")
            # TODO: FFT/DCT haritasini goster
            st.info("Modul henuz egitilmedi - entegrasyon bekleniyor")

        # Hierarchical Fuzyon
        progress.progress(80, text="Hierarchical Cross-Modal Fusion uygulanıyor...")
        st.markdown("---")
        st.subheader("Hierarchical Cross-Modal Fusion")

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.markdown("**Asama 1:** Shallow-layer Attention Enhancement")
            st.caption("Spatial ↔ Frequency capraz dikkat")
        with fcol2:
            st.markdown("**Asama 2:** Deep-layer Dynamic Modulation")
            st.caption("Modalite guvenilirlik agirlandirma")

        # Nihai Sonuc
        progress.progress(100, text="Analiz tamamlandi!")
        st.markdown("---")

        st.subheader("Nihai Sonuc")
        st.warning("Modeller henuz egitilmedi. Egitim tamamlandiktan sonra "
                    "burada guvenilirlik skoru ve gercek/sahte damgasi gorunecektir.")

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
