"""End-to-end inference latency olcumu.

Birkac ham .mp4 uzerinden DeepfakeDetector.predict() bilesenlerini
parcali zamanlar: kare cikarma, yuz tespiti, spatial, frequency, audio, fusion.
Hem CPU hem GPU (cuda varsa) icin ortalama+std raporlanir.

Cikti: reports/analysis/inference_time.json + .png
Calistir: python scripts/benchmark_inference.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import DeepfakeDetector, fuse

OUT = ROOT / "reports" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Karisik kume: 5 Celeb-DF + 3 FaceForensics (ses var)
VIDEOS = (
    sorted((ROOT / "data/raw/celeb-df-v2/Celeb-real").glob("id0_*.mp4"))[:3]
    + sorted((ROOT / "data/raw/celeb-df-v2/Celeb-synthesis").glob("*.mp4"))[:3]
    + sorted((ROOT / "data/raw/faceforensics/DeepFakeDetection").glob("*.mp4"))[:2]
)


def time_one(det: DeepfakeDetector, path: str) -> dict:
    t0 = time.perf_counter()
    frames = det.video_processor.extract_frames(path)
    t1 = time.perf_counter()
    det._ensure_face_detector()
    faces = det._face_detector.detect_faces(frames)
    t2 = time.perf_counter()
    sp = det._spatial_predict(faces) if faces else None
    t3 = time.perf_counter()
    fr = det._frequency_predict(faces) if faces else None
    t4 = time.perf_counter()
    au = det._audio_predict(path)
    t5 = time.perf_counter()
    _ = fuse({"spatial": sp, "frequency": fr, "audio": au})
    t6 = time.perf_counter()
    return {
        "video": Path(path).name,
        "n_frames": len(frames),
        "n_faces": len(faces),
        "audio_used": au is not None,
        "extract": t1 - t0,
        "face_detect": t2 - t1,
        "spatial": t3 - t2,
        "frequency": t4 - t3,
        "audio": t5 - t4,
        "fusion": t6 - t5,
        "total": t6 - t0,
    }


def run_device(device: str) -> list[dict]:
    print(f"\n=== {device.upper()} ===", flush=True)
    det = DeepfakeDetector(config={"device": device, "max_frames": 30, "frame_rate": 2})
    # Warmup (model yukle, ilk allocation)
    for v in VIDEOS[:1]:
        time_one(det, str(v))
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()

    results = []
    for i, v in enumerate(VIDEOS):
        try:
            r = time_one(det, str(v))
            print(f"  {i+1}/{len(VIDEOS)}  {r['video']:32s}  "
                  f"total={r['total']:.2f}s  faces={r['n_faces']:2d}  audio={'+' if r['audio_used'] else '-'}",
                  flush=True)
            results.append(r)
        except Exception as e:
            print(f"  {i+1} FAIL: {e}", flush=True)
    return results


def aggregate(rows: list[dict]) -> dict:
    keys = ["extract", "face_detect", "spatial", "frequency", "audio", "fusion", "total"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rows])
        summary[k] = {
            "mean_s": float(vals.mean()),
            "std_s": float(vals.std()),
            "median_s": float(np.median(vals)),
        }
    total = np.array([r["total"] for r in rows])
    nf = np.array([r["n_frames"] for r in rows])
    summary["fps_videos"] = float(1.0 / total.mean())
    summary["fps_frames"] = float((nf / total).mean())
    summary["n_videos"] = len(rows)
    return summary


def plot(summaries: dict, path: Path):
    devices = list(summaries.keys())
    components = ["extract", "face_detect", "spatial", "frequency", "audio", "fusion"]
    labels = ["Kare çıkar", "Yüz tespiti", "Spatial",
              "Frequency", "Audio", "Fusion"]

    x = np.arange(len(components))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = {"cpu": "#bb8009", "cuda": "#3fb950"}
    for i, dev in enumerate(devices):
        s = summaries[dev]
        means = [s[c]["mean_s"] * 1000 for c in components]
        stds = [s[c]["std_s"] * 1000 for c in components]
        offset = (i - (len(devices) - 1) / 2) * w
        bars = ax.bar(x + offset, means, w, yerr=stds, capsize=4,
                      label=f"{dev.upper()}  (toplam {s['total']['mean_s']*1000:.0f} ms/video)",
                      color=colors.get(dev, "#888"), alpha=0.85)
        for b in bars:
            h = b.get_height()
            if h > 5:
                ax.text(b.get_x() + b.get_width()/2, h + 2,
                        f"{h:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms) — ortalama ± std")
    ax.set_title("End-to-end Inference Latency — bileşen dökümü", weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    if not VIDEOS:
        raise SystemExit("Test icin .mp4 bulunamadi (data/raw).")
    print(f"Test videolari: {len(VIDEOS)}")
    out = {"videos": [str(v.relative_to(ROOT)) for v in VIDEOS], "devices": {}}

    # GPU varsa once GPU, sonra CPU
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")

    for dev in devices:
        rows = run_device(dev)
        out["devices"][dev] = {
            "per_video": rows,
            "summary": aggregate(rows),
        }

    (OUT / "inference_time.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    plot({d: out["devices"][d]["summary"] for d in out["devices"]},
         OUT / "inference_time.png")

    print("\n=== OZET ===")
    for dev, data in out["devices"].items():
        s = data["summary"]
        print(f"{dev.upper():5s}  total={s['total']['mean_s']*1000:6.1f}±{s['total']['std_s']*1000:.1f} ms/video  "
              f"({s['fps_videos']:.2f} video/s, {s['fps_frames']:.1f} kare/s eşdeğeri)")


if __name__ == "__main__":
    main()
