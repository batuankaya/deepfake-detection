"""Deepfake tespit — Streamlit arayuzu.

Bitirme projemin demo uygulamasi. Yuklenen videoyu uc modul (uzamsal,
frekans, isitsel) ile analiz edip karar fuzyonundan gecirir.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALLOWED_EXTENSIONS = {".mp4"}
MAX_FILE_SIZE_MB = 200
ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Deepfake tespit",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS bilerek minimal: yalniz karar metnini renklendiren bir kural.
# Geri kalan her sey Streamlit defaults — site bir tema islegi degil, bir
# ogrenci projesi demosu.
st.markdown(
    """
    <style>
    .verdict-fake { color: #e5484d; font-weight: 700; font-size: 26px; }
    .verdict-real { color: #30a46c; font-weight: 700; font-size: 26px; }
    .verdict-unk  { color: #b0b4ba; font-weight: 700; font-size: 26px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_detector():
    """Modelleri tek sefer yukler (CPU; Streamlit+Windows+CUDA crash onlemi)."""
    from src.inference import DeepfakeDetector
    return DeepfakeDetector(config={"device": "cpu"})


def run_prediction_subprocess(video_path: str) -> dict:
    """Tahmini ayri bir process'te calistirir.

    Windows'ta torch+MTCNN+librosa, Streamlit'in thread modeli icinde
    bazen native crash atiyor. Onun yerine scripts/predict_cli.py'i alt-process
    olarak cagiriyoruz, sonuc JSON donuyor; arayuz hicbir zaman cokmez.
    """
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/predict_cli.py", "--json", video_path],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Tahmin 5 dakikada bitmedi (timeout).",
                "verdict": "UNKNOWN", "confidence": 0.0}
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"error": f"Tahmin basarisiz (exit {proc.returncode})",
            "verdict": "UNKNOWN", "confidence": 0.0,
            "modules": {}, "module_weights": {}}


def _ffmpeg_path():
    """ffmpeg ikilisini bulur (PATH, WinGet veya Chocolatey)."""
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    for cand in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
    ]:
        if cand.exists():
            return str(cand)
    return None


def browser_safe_video(src_path: str) -> str:
    """Tarayici uyumlu kopya dondurur (gerekiyorsa transcode eder).

    HTML5 <video> H.264 Baseline/Main/High + yuv420p ister. FF++ original
    (yuv444p), Celeb-DF v2 (mpeg4 Simple Profile) ve bazi diger kayitlar
    uyumsuzdur — Streamlit'in player'inda kara ekran cikar.

    Cozum: ffprobe ile codec/pix_fmt'i okuyup uyumsuzsa libx264/yuv420p'ye
    yeniden kodla. Hizli (ultrafast preset). Analiz pipeline'i daima
    orijinal dosyadan calisir.
    """
    ffmpeg = _ffmpeg_path()
    if ffmpeg is None:
        return src_path

    ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe").replace(
        "ffmpeg", "ffprobe")
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "default=nw=1:nk=1", src_path],
            capture_output=True, text=True, timeout=10,
        )
        info = proc.stdout.strip().splitlines()
        codec = info[0] if info else ""
        pix = info[1] if len(info) > 1 else ""
    except Exception:
        return src_path

    if codec == "h264" and pix == "yuv420p":
        return src_path

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out.close()
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", src_path,
             "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
             "-preset", "ultrafast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart",
             out.name],
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0 or not Path(out.name).exists():
            return src_path
        return out.name
    except Exception:
        return src_path


def module_status() -> dict:
    return {
        "audio": Path("checkpoints/best_audio.pt").exists(),
        "frequency": Path("checkpoints/best_frequency.pt").exists(),
        "spatial": Path("checkpoints/best_spatial.pt").exists(),
    }


def _load_metrics(rel: str) -> dict:
    try:
        d = json.loads((ROOT / "reports" / rel / "metrics.json")
                        .read_text(encoding="utf-8"))
    except Exception:
        return {}
    return d.get("metrics", d)


def render_metrics_page():
    """Egitilen modullerin gercek degerlendirme sayilari."""
    import pandas as pd

    st.title("Başarı metrikleri")
    st.write(
        "Aşağıdaki sayılar `reports/` altındaki `metrics.json` dosyalarından "
        "doğrudan yüklenir; herhangi bir yeniden hesaplama yapılmaz."
    )

    st.subheader("Test seti — modül başına")
    test = {
        "Ses (Mel-CNN, ASVspoof)": _load_metrics("best_audio_test"),
        "Frekans (Block-DCT)": _load_metrics("best_frequency_test"),
        "Görüntü (EfficientNet-B4+BiLSTM)": _load_metrics("best_spatial_test"),
    }
    rows = []
    for name, m in test.items():
        if not m:
            continue
        rows.append({
            "Modül": name,
            "Accuracy": f"%{m['accuracy']*100:.2f}",
            "Precision": f"%{m['precision']*100:.2f}",
            "Recall": f"%{m['recall']*100:.2f}",
            "F1": f"%{m['f1']*100:.2f}",
            "AUC": f"{m['auc_roc']:.4f}",
            "EER": f"%{m['eer']*100:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    fu = json.loads((ROOT / "reports" / "fusion_eval" / "metrics.json")
                    .read_text(encoding="utf-8")) if (
        ROOT / "reports" / "fusion_eval" / "metrics.json").exists() else {}
    if fu.get("fusion"):
        f = fu["fusion"]
        st.subheader(f"Karar füzyonu — FF++ c23 test ({fu.get('n_videos', 0)} video)")
        fc = st.columns(4)
        fc[0].metric("Accuracy", f"%{f['accuracy']*100:.2f}")
        fc[1].metric("Precision", f"%{f['precision']*100:.2f}")
        fc[2].metric("Recall", f"%{f['recall']*100:.2f}")
        fc[3].metric("F1 (sahte)", f"%{f['f1']*100:.2f}")
        st.write(
            "Tek modül en iyi sonucu uzamsal verdi (%93.61). Füzyon bunu "
            "geçen tek konfigürasyon. " + fu.get("note", "")
        )

    st.subheader("Cross-dataset — Celeb-DF v2")
    cross = {
        "Görüntü": _load_metrics("best_spatial_cross_celebdf"),
        "Frekans": _load_metrics("best_frequency_cross_celebdf"),
    }
    crows = []
    for name, m in cross.items():
        if not m:
            continue
        crows.append({
            "Modül": name,
            "Accuracy": f"%{m['accuracy']*100:.2f}",
            "F1": f"%{m['f1']*100:.2f}",
            "Recall": f"%{m['recall']*100:.2f}",
            "AUC": f"{m['auc_roc']:.4f}",
        })
    st.dataframe(pd.DataFrame(crows), hide_index=True, width="stretch")

    st.subheader("Referanslarla karşılaştırma (aynı set, Celeb-DF v2)")
    st.dataframe(pd.DataFrame([
        {"Çalışma": "Qiao et al. (2025) — çekirdek ref",
         "Acc": "—", "F1": "—", "AUC": "0.7468"},
        {"Çalışma": "Yıldırım (2025) — tez 957056",
         "Acc": "%89.17", "F1": "%89.29", "AUC": "—"},
        {"Çalışma": "Bizim — uzamsal",
         "Acc": "%77.91", "F1": "%86.08", "AUC": "0.8393"},
        {"Çalışma": "Bizim — frekans",
         "Acc": "%71.79", "F1": "%81.76", "AUC": "0.7353"},
    ]), hide_index=True, width="stretch")
    st.markdown(
        "AUC ekseninde çekirdek referansı **+9.25 puan** aşıyoruz "
        "(0.8393 vs 0.7468). Accuracy düşüşü modelin değil, eşiğin "
        "domain-shift altında kaymasının ölçüsü — README'deki "
        "*Eşik & Olasılık Kalibrasyonu* başlığında ayrıntı var."
    )

    st.subheader("Değerlendirme grafikleri")
    gtabs = st.tabs(["Test ROC", "Test karışıklık m.",
                     "Cross ROC", "Cross karışıklık m."])
    with gtabs[0]:
        g = st.columns(3)
        for c, t, p in zip(
            g, ["Ses", "Frekans", "Görüntü"],
            ["best_audio_test/test_roc.png",
             "best_frequency_test/test_roc.png",
             "best_spatial_test/test_roc.png"],
        ):
            fp = ROOT / "reports" / p
            if fp.exists():
                c.image(str(fp), caption=t, width="stretch")
    with gtabs[1]:
        g = st.columns(3)
        for c, t, p in zip(
            g, ["Ses", "Frekans", "Görüntü"],
            ["best_audio_test/test_cm.png",
             "best_frequency_test/test_cm.png",
             "best_spatial_test/test_cm.png"],
        ):
            fp = ROOT / "reports" / p
            if fp.exists():
                c.image(str(fp), caption=t, width="stretch")
    with gtabs[2]:
        g = st.columns(2)
        for c, t, p in zip(
            g, ["Görüntü (cross)", "Frekans (cross)"],
            ["best_spatial_cross_celebdf/cross_celebdf_roc.png",
             "best_frequency_cross_celebdf/cross_celebdf_roc.png"],
        ):
            fp = ROOT / "reports" / p
            if fp.exists():
                c.image(str(fp), caption=t, width="stretch")
    with gtabs[3]:
        g = st.columns(2)
        for c, t, p in zip(
            g, ["Görüntü (cross)", "Frekans (cross)"],
            ["best_spatial_cross_celebdf/cross_celebdf_cm.png",
             "best_frequency_cross_celebdf/cross_celebdf_cm.png"],
        ):
            fp = ROOT / "reports" / p
            if fp.exists():
                c.image(str(fp), caption=t, width="stretch")

    st.caption("Veri kaynağı: `reports/*/metrics.json`. Tam analiz için "
               "`reports/analysis/` (ablation, kalibrasyon, latency, vahşi-doğa).")


# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("### Deepfake tespit")
    st.write("Batuhan Kaya — 232923023")
    st.write("Bitirme projesi demo arayüzü.")

    page = st.radio(
        "Sayfa",
        ["Video analizi", "Başarı metrikleri"],
        label_visibility="collapsed",
    )

    st.divider()

    get_detector()
    status = module_status()
    trained = sum(status.values())

    st.write(f"**Modüller** ({trained}/3 yüklü)")
    for name, key in [("Görüntü", "spatial"),
                       ("Frekans", "frequency"),
                       ("Ses", "audio")]:
        ok = status[key]
        st.write(f"- {name}: {'aktif' if ok else 'yok'}")
    st.write("- Karar füzyonu: AUC-ağırlıklı, EER-kalibreli")

    st.divider()

    with st.expander("Modüller ne yapar"):
        st.write(
            "**Görüntü** — EfficientNet-B4 + BiLSTM; yüzdeki mimik, ışık, "
            "doku tutarsızlıklarını yakalar.\n\n"
            "**Frekans** — Block-wise DCT + 2D-FFT + Multi-scale Conv; "
            "GAN/difüzyon üreteçlerinin yüksek frekansta bıraktığı izleri "
            "yakalar.\n\n"
            "**Ses** — Mel-spektrogram + CNN; sentetik tonlamayı yakalar "
            "(ASVspoof 2019 LA üstünde eğitildi).\n\n"
            "**Füzyon** — Her modülün EER eşiğiyle [0,1]'e kalibre edilmiş "
            "olasılığı, AUC^2 ile ağırlıklandırılıp toplanır. Bir modül 0.85 "
            "üstünde çok eminse override ile sahte alarmı verilir."
        )

    with st.expander("Bilinen sınırlamalar"):
        st.write(
            "- Video setlerimiz sessiz; fusion sayıları 2-modüllüdür "
            "(spatial+frequency).\n"
            "- Cross-dataset accuracy düşüyor (eşik FF++'a kalibre).\n"
            "- NeuralTextures en zor manipülasyon (recall %86.67).\n"
            "- Gerçek-zamanlı değil: ~3.4 s/video (CUDA), ~5.6 s (CPU).\n"
            "- Tam liste: README → Sınırlamalar."
        )

    with st.expander("Kaynaklar"):
        st.write(
            "[1] Spatial-Frequency Collaborative Learning & Hierarchical "
            "Cross-Modal Fusion, 2025 — arxiv.org/abs/2504.17223"
        )
        st.write(
            "[2] Öz Denetimli Öğrenme ile Derin Sahte Tespiti, "
            "Yıldırım 2025 — YÖK Tez No: 957056"
        )


if page == "Başarı metrikleri":
    render_metrics_page()
    st.stop()


# ---------- Video analizi sayfasi ----------

st.title("Video analizi")
st.write(
    "Bir `.mp4` yükleyin (en fazla 200 MB). Yüz kareleri çıkarılır, üç "
    "modülden geçer, karar füzyonu nihai cevabı verir. İlk yüklemede "
    "modeller belleğe alınırken biraz beklersiniz; ondan sonra hızlı."
)

uploaded_file = st.file_uploader(
    "Video dosyası", type=["mp4"],
    help="Sadece .mp4, en fazla 200 MB",
)

if uploaded_file is None:
    st.info("Buraya bir video bırakın; analiz otomatik başlamaz, üstündeki "
            "**Analizi başlat** tuşuna basmanız gerekir.")
else:
    file_ext = Path(uploaded_file.name).suffix.lower()
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

    if file_ext not in ALLOWED_EXTENSIONS:
        st.error("Sadece .mp4 destekleniyor.")
    elif file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"Dosya {file_size_mb:.1f} MB; sınır {MAX_FILE_SIZE_MB} MB.")
    else:
        tmp_path = None
        playable_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Tarayici-uyumsuz codec'leri (yuv444p, mpeg4) h264/yuv420p'ye
            # transcode et — yoksa st.video kara ekran gosterir.
            playable_path = browser_safe_video(tmp_path)
            transcoded = playable_path != tmp_path

            vid_col, info_col = st.columns([2, 1])
            with vid_col:
                with open(playable_path, "rb") as fh:
                    st.video(fh.read())
                if transcoded:
                    st.caption(
                        "Bu videonun codec'i (yuv444p / mpeg4) tarayıcıda "
                        "oynamadığı için yalnız önizleme için H.264/yuv420p'ye "
                        "kopyalandı. Analiz orijinal dosyadan çalışıyor."
                    )
            with info_col:
                st.write(f"**Dosya:** {uploaded_file.name}")
                st.write(f"**Boyut:** {file_size_mb:.1f} MB")
                st.button("Analizi başlat", type="primary",
                          key="analyze_btn", width="stretch")

            if st.session_state.get("analyze_btn"):
                from src.utils.visualize import plot_dct_analysis, plot_fft_analysis

                with st.status("Analiz çalışıyor...", expanded=True) as sbox:
                    st.write("Kareler çıkarılıyor, frekans haritası "
                             "hesaplanıyor...")
                    detector = get_detector()
                    analysis = detector.analyze_video(
                        tmp_path, run_prediction=False)
                    if "error" in analysis:
                        st.error(analysis["error"])
                    else:
                        st.write(f"{analysis['frames_analyzed']} kare işlendi. "
                                 "Modeller alt-süreçte çalışıyor (Windows'ta "
                                 "stabilite için).")
                        analysis["prediction"] = run_prediction_subprocess(
                            tmp_path)
                        pred = analysis["prediction"]
                        if "error" in pred:
                            st.error(pred["error"])
                        else:
                            st.write(
                                f"{pred.get('faces_detected', 0)} yüz "
                                f"bulundu, {pred.get('active_modules', 0)} "
                                "modül aktif."
                            )
                    sbox.update(label="Analiz tamamlandı", state="complete")

                if "error" not in analysis:
                    pred = analysis.get("prediction", {})
                    verdict = pred.get("verdict", "UNKNOWN")
                    fake_prob = pred.get("fake_probability")
                    confidence = pred.get("confidence", 0.0)

                    # Karar - sade, Streamlit'in built-in alert'leri
                    pct = f"%{fake_prob*100:.1f}" if fake_prob is not None else "—"
                    if verdict == "FAKE":
                        st.error(
                            f"**Karar: SAHTE**  ·  sahte olasılığı {pct}  ·  "
                            f"güven %{confidence*100:.1f}"
                        )
                    elif verdict == "REAL":
                        st.success(
                            f"**Karar: GERÇEK**  ·  sahte olasılığı {pct}  ·  "
                            f"güven %{confidence*100:.1f}"
                        )
                    else:
                        st.warning(
                            "**Karar: belirsiz** — modüllerin hiçbiri "
                            "yüklenmedi veya video analiz edilemedi."
                        )

                    if verdict != "UNKNOWN" and confidence < 0.3:
                        st.caption(
                            "Güven düşük (%30 altı); model bu video için "
                            "sınır bir karar verdi. Eşiğe çok yakın "
                            "kalmış olabilir."
                        )

                    # Modul olasiliklari
                    st.subheader("Modül tahminleri")
                    modules = pred.get("modules", {})
                    weights = pred.get("module_weights", {})
                    cols = st.columns(3)
                    for col, (key, label) in zip(
                        cols, [("spatial", "Görüntü"),
                                ("frequency", "Frekans"),
                                ("audio", "Ses")]):
                        val = modules.get(key)
                        with col:
                            if val is None:
                                col.metric(label, "—", help="Modül yüklü değil "
                                           "veya bu kaynak için anlamlı değil "
                                           "(ses olmayan video gibi).")
                            else:
                                col.metric(
                                    label,
                                    f"%{val*100:.1f}",
                                    delta=(f"ağırlık %{weights[key]*100:.0f}"
                                           if key in weights else None),
                                    delta_color="off",
                                    help="Bu modülün ham sahte olasılığı. "
                                         "Karar füzyonu bu sayıyı önce EER "
                                         "eşiğiyle kalibre eder, sonra "
                                         "AUC²-ağırlığıyla harmanlar.",
                                )
                                col.progress(float(val))

                    if analysis["sample_frames"]:
                        st.subheader("Çıkarılan kareler")
                        st.caption("Yüz tespitinden geçirilmiş üç örnek kare "
                                   "(başlangıç, orta, son).")
                        fcols = st.columns(3)
                        caps = ["Başlangıç", "Orta", "Son"]
                        for i, (c, fr) in enumerate(
                                zip(fcols, analysis["sample_frames"])):
                            c.image(fr, caption=caps[i], width="stretch")

                    st.subheader("Frekans haritaları")
                    st.caption(
                        "GAN/difüzyon üreteçleri yüksek frekansta yapay izler "
                        "bırakma eğilimindedir; aşağıdaki sayılar ve grafikler "
                        "bu izi görselleştirir. **Y/D oranı** modelin de "
                        "kullandığı asıl sinyaldir."
                    )
                    freq = analysis["frequency_analysis"]
                    mid = (analysis["sample_frames"][1]
                           if len(analysis["sample_frames"]) > 1
                           else analysis["sample_frames"][0])
                    mc = st.columns(4)
                    mc[0].metric(
                        "Yüksek frekans", f"{freq['high_freq_energy']:.3f}",
                        help="Karedeki ince detay/kenar/doku enerjisi "
                             "(spektrumun dış bölgesi). Yapay görüntülerde "
                             "üreteç parmak izi nedeniyle anormal yüksek "
                             "olabilir.")
                    mc[1].metric(
                        "Düşük frekans", f"{freq['low_freq_energy']:.3f}",
                        help="Karedeki genel yapı/renk/aydınlatma enerjisi "
                             "(spektrumun merkezi). Sahnenin kaba içeriğini "
                             "temsil eder, tek başına ayırt edici değildir.")
                    mc[2].metric(
                        "Y/D oranı", f"{freq['freq_ratio']:.3f}",
                        help="Yüksek ÷ Düşük frekans enerjisi. Asıl tespit "
                             "göstergesi: gerçek kamera görüntülerinde düşük "
                             "kalır, GAN/difüzyon üretiminde yükselir.")
                    mc[3].metric(
                        "Ort. (tüm kare)", f"{analysis['avg_freq_ratio']:.3f}",
                        help="Y/D oranının analiz edilen tüm karelerdeki "
                             "ortalaması. Tekil kare gürültüsüne karşı "
                             "daha kararlıdır.")

                    t1, t2, t3 = st.tabs(
                        ["FFT spektrum", "DCT haritası", "Frekans profili"])
                    with t1:
                        st.caption(
                            "Karenin 2B Fourier dönüşümü. Merkez = düşük "
                            "frekans (genel yapı), kenarlar = yüksek frekans "
                            "(ince doku). GAN/difüzyon üreteçleri yüksek "
                            "frekansta yapay ızgara/halka desenleri bırakır; "
                            "gerçek kamera görüntülerinde bu desenler "
                            "**bulunmaz**."
                        )
                        st.pyplot(plot_fft_analysis(mid))
                    with t2:
                        st.caption(
                            "Block-wise Diskret Kosinüs Dönüşümü (JPEG'in de "
                            "kullandığı dönüşüm). Görüntüyü bloklara bölüp "
                            "her bloğun frekans enerjisini gösterir. "
                            "Deepfake'lerde yüz birleştirme ve yeniden "
                            "sıkıştırma, blok sınırlarında **tutarsız DCT "
                            "katsayıları** oluşturur."
                        )
                        try:
                            st.pyplot(plot_dct_analysis(mid))
                        except Exception:
                            st.warning("DCT analizi sırasında hata oluştu.")
                    with t3:
                        st.caption(
                            "FFT spektrumunun merkezden dışa açısal ortalaması "
                            "(azimuthal average). Yatay eksen düşük→yüksek "
                            "frekans, dikey eksen o frekanstaki ortalama "
                            "enerji. Gerçek görüntülerde eğri yumuşak iner; "
                            "yapay üretimde yüksek frekansta **anormal "
                            "tepe/sıçrama** görülür (üreteç parmak izi)."
                        )
                        az = freq.get("azimuthal_profile")
                        if az is None:
                            from src.models.frequency.frequency_model import \
                                FrequencyAnalyzer
                            gray = (np.mean(mid, axis=2)
                                    if len(mid.shape) == 3 else mid)
                            spec = FrequencyAnalyzer.compute_fft_spectrum(gray)
                            az = FrequencyAnalyzer.compute_azimuthal_average(spec)
                        import pandas as pd
                        az = np.asarray(az, dtype=float)
                        st.line_chart(
                            pd.DataFrame({"Frekans (piksel)": np.arange(len(az)),
                                          "Ortalama genlik": az}),
                            x="Frekans (piksel)", y="Ortalama genlik")

                    st.divider()
                    report = {
                        "dosya": uploaded_file.name,
                        "karar": verdict,
                        "sahte_olasiligi": fake_prob,
                        "guvenilirlik": confidence,
                        "moduller": modules,
                        "modul_agirliklari": weights,
                        "kalibre": pred.get("calibrated", {}),
                        "kare_sayisi": analysis["frames_analyzed"],
                        "yuz_sayisi": pred.get("faces_detected", 0),
                    }
                    st.download_button(
                        "Analiz raporunu indir (JSON)",
                        data=json.dumps(report, indent=2, ensure_ascii=False),
                        file_name=f"rapor_{Path(uploaded_file.name).stem}.json",
                        mime="application/json", width="stretch")
        finally:
            for p in (tmp_path, playable_path):
                if p and p != tmp_path and os.path.exists(p):
                    os.unlink(p)
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
