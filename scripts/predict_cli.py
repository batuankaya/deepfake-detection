"""Streamlit'siz hizli komut satiri tahmini.

Kullanim (proje kokunden):
    python scripts/predict_cli.py <video.mp4>

Ornek:
    python scripts/predict_cli.py data/raw/faceforensics/Deepfakes/003_000.mp4
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import DeepfakeDetector


def fmt_pct(x):
    return f"%{x*100:.1f}" if x is not None else "Pasif"


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python predict_cli.py <video.mp4>")
        sys.exit(1)
    json_mode = "--json" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--json"]
    video = args[0]
    if not Path(video).exists():
        if json_mode:
            print(json.dumps({"error": f"Dosya yok: {video}"}))
        else:
            print(f"HATA: Dosya yok: {video}")
        sys.exit(1)

    if json_mode:
        det = DeepfakeDetector()
        print(json.dumps(det.predict(video), default=str))
        return

    print(f"Yukleniyor: {video}")
    print("=" * 60)

    t0 = time.time()
    det = DeepfakeDetector()
    print(f"Model yukleme: {time.time()-t0:.1f}s")

    t0 = time.time()
    r = det.predict(video)
    print(f"Tahmin: {time.time()-t0:.1f}s\n")

    print("=" * 60)
    verdict = r.get("verdict", "?")
    icon = "🟥 SAHTE" if verdict == "FAKE" else "🟩 GERCEK" if verdict == "REAL" else "⬜ BELIRSIZ"
    print(f"  SONUC: {icon}")
    fp = r.get("fake_probability")
    if fp is not None:
        print(f"  Sahte olasiligi (kalibre ort): {fp*100:.1f}%")
    print(f"  Guvenilirlik: {r['confidence']*100:.1f}%")
    print("=" * 60)

    print(f"\nModul tahminleri (ham olasilik):")
    for k, v in r["modules"].items():
        print(f"  {k:10s} : {fmt_pct(v)}")
    if "calibrated" in r:
        print(f"\nKalibre olasiliklar:")
        for k, v in r["calibrated"].items():
            print(f"  {k:10s} : %{v*100:.1f}")
    if "fake_votes" in r:
        print(f"\nOylar: {r['fake_votes']} fake / {r['real_votes']} real")
    print(f"\nKareler: {r['frames_analyzed']}, yuzler: {r['faces_detected']}")


if __name__ == "__main__":
    main()
