"""Vahsi-doga / training-disi sanity testi.

Egitim ve test setlerinde HIC olmayan kaynaklardan rastgele orneklenmis
videolar uzerinde DeepfakeDetector'i calistirir. Amac: laboratuvar
disinda da modulun makul tahminler verdigini gostermek.

Kaynaklar (rastgele 4 video / kaynak, seed=7):
  - FF++/original         (gercek, FF++ ham — egitim disi ID'ler)
  - FF++/DeepFakeDetection (sahte, Google DFD — train'de yok)
  - Celeb-DF/Celeb-real    (gercek, farkli dagilim)
  - Celeb-DF/Celeb-synthesis (sahte, farkli dagilim)

Cikti: reports/analysis/wild_test.{json,md}
Calistir: python scripts/wild_test.py
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import DeepfakeDetector

OUT = ROOT / "reports" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "name": "FF++ original (raw, eğitim-dışı ID)",
        "label": 0,
        "label_str": "GERÇEK",
        "dir": ROOT / "data/raw/faceforensics/original",
        "glob": "*.mp4",
    },
    {
        "name": "FF++ DeepFakeDetection (Google DFD — train'de yok)",
        "label": 1,
        "label_str": "SAHTE",
        "dir": ROOT / "data/raw/faceforensics/DeepFakeDetection",
        "glob": "*.mp4",
    },
    {
        "name": "FF++ FaceShifter (eğitim-dışı manipülasyon)",
        "label": 1,
        "label_str": "SAHTE",
        "dir": ROOT / "data/raw/faceforensics/FaceShifter",
        "glob": "*.mp4",
    },
    {
        "name": "Celeb-DF v2 Celeb-real (farklı dağılım)",
        "label": 0,
        "label_str": "GERÇEK",
        "dir": ROOT / "data/raw/celeb-df-v2/Celeb-real",
        "glob": "*.mp4",
    },
    {
        "name": "Celeb-DF v2 Celeb-synthesis (farklı dağılım)",
        "label": 1,
        "label_str": "SAHTE",
        "dir": ROOT / "data/raw/celeb-df-v2/Celeb-synthesis",
        "glob": "*.mp4",
    },
]

N_PER_SOURCE = 20
SEED = 7


def pick_videos():
    rng = random.Random(SEED)
    picked = []
    for src in SOURCES:
        all_v = sorted(src["dir"].glob(src["glob"]))
        if not all_v:
            print(f"UYARI: {src['dir']} bos", flush=True)
            continue
        sample = rng.sample(all_v, min(N_PER_SOURCE, len(all_v)))
        for v in sample:
            picked.append({
                "source": src["name"],
                "label": src["label"],
                "label_str": src["label_str"],
                "path": v,
            })
    return picked


def fmt_p(p):
    return "—" if p is None else f"{p:.3f}"


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    videos = pick_videos()
    print(f"Cihaz: {device.upper()} | Test: {len(videos)} video, "
          f"{len(SOURCES)} kaynak\n", flush=True)

    det = DeepfakeDetector(config={"device": device,
                                    "max_frames": 30, "frame_rate": 2})
    results = []
    for i, v in enumerate(videos, 1):
        t0 = time.perf_counter()
        try:
            r = det.predict(str(v["path"]))
            dt = time.perf_counter() - t0
            ok = r["verdict"] == ("FAKE" if v["label"] == 1 else "REAL")
            row = {
                "i": i,
                "source": v["source"],
                "video": v["path"].name,
                "truth": v["label_str"],
                "verdict": r["verdict"],
                "fake_prob": r.get("fake_probability"),
                "confidence": r.get("confidence"),
                "p_spatial": r["modules"].get("spatial"),
                "p_frequency": r["modules"].get("frequency"),
                "p_audio": r["modules"].get("audio"),
                "active_modules": r.get("active_modules"),
                "faces": r.get("faces_detected"),
                "correct": ok,
                "latency_s": round(dt, 2),
            }
            mark = "OK" if ok else "X "
            print(f"  [{i:2d}] {mark} {v['path'].name[:42]:42s}  "
                  f"truth={v['label_str']:6s}  -> {r['verdict']:7s}  "
                  f"(fake_p={fmt_p(r.get('fake_probability'))})  "
                  f"{dt:.1f}s", flush=True)
            results.append(row)
        except Exception as e:
            print(f"  [{i:2d}] FAIL  {v['path'].name}: {e}", flush=True)
            results.append({
                "i": i, "source": v["source"], "video": v["path"].name,
                "truth": v["label_str"], "verdict": "ERROR", "error": str(e),
                "correct": False,
            })

    # Ozet
    valid = [r for r in results if r.get("verdict") != "ERROR"]
    by_truth = {"GERÇEK": [], "SAHTE": []}
    for r in valid:
        by_truth[r["truth"]].append(r)
    summary = {
        "n_total": len(results),
        "n_valid": len(valid),
        "accuracy": sum(r["correct"] for r in valid) / max(len(valid), 1),
        "real_correct": sum(r["correct"] for r in by_truth["GERÇEK"]),
        "real_total": len(by_truth["GERÇEK"]),
        "fake_correct": sum(r["correct"] for r in by_truth["SAHTE"]),
        "fake_total": len(by_truth["SAHTE"]),
    }

    out_json = {"seed": SEED, "n_per_source": N_PER_SOURCE,
                "summary": summary, "results": results}
    (OUT / "wild_test.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # Kaynak basina ozet
    by_src = defaultdict(list)
    for r in valid:
        by_src[r["source"]].append(r)

    # Markdown raporu
    md = []
    md.append("## Vahşi-Doğa Testi (Eğitim-Dışı Kaynaklar)\n")
    md.append(f"Rastgele seçilmiş **{len(valid)} video / {len(SOURCES)} kaynak** "
              f"(seed={SEED}, n_per_source={N_PER_SOURCE}). Bu videoların hiçbiri "
              "eğitim veya test setinde değildi.\n")
    md.append(f"**Genel doğruluk: %{summary['accuracy']*100:.1f}**  "
              f"(Gerçek {summary['real_correct']}/{summary['real_total']}, "
              f"Sahte {summary['fake_correct']}/{summary['fake_total']})\n")

    md.append("### Kaynak başına özet\n")
    md.append("| Kaynak | n | Doğru | Doğruluk | Ort. spatial | Ort. freq |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for src_name, items in by_src.items():
        n = len(items)
        correct = sum(r["correct"] for r in items)
        sp_mean = np.mean([r["p_spatial"] for r in items
                           if r["p_spatial"] is not None])
        fr_mean = np.mean([r["p_frequency"] for r in items
                           if r["p_frequency"] is not None])
        md.append(
            f"| {src_name} | {n} | {correct}/{n} | "
            f"**%{correct/n*100:.1f}** | {sp_mean:.3f} | {fr_mean:.3f} |"
        )
    md.append("")

    md.append("### Per-video sonuçlar\n")
    md.append("| # | Kaynak | Video | Gerçek | Tahmin | Fake olas. | Spatial | Freq | s |")
    md.append("|---|--------|-------|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        if r.get("verdict") == "ERROR":
            md.append(f"| {r['i']} | {r['source']} | {r['video']} | "
                      f"{r['truth']} | HATA | — | — | — | — |")
            continue
        mark = "✓" if r["correct"] else "✗"
        md.append(
            f"| {r['i']} | {r['source']} | `{r['video']}` | {r['truth']} | "
            f"**{r['verdict']}** {mark} | {fmt_p(r['fake_prob'])} | "
            f"{fmt_p(r['p_spatial'])} | {fmt_p(r['p_frequency'])} | "
            f"{r['latency_s']} |"
        )
    md.append("")
    md.append("> Ses kanalı bu kaynaklarda yok — aktif modül = 2 (spatial + "
              "frequency). 3-modüllü test multimodal bir set "
              "(DFDC/FakeAVCeleb) gerektirir; bkz. README *Sınırlamalar*.")
    (OUT / "wild_test.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== OZET ===")
    print(f"Toplam: {summary['n_valid']}/{summary['n_total']}  "
          f"Acc: %{summary['accuracy']*100:.1f}  "
          f"(Gercek {summary['real_correct']}/{summary['real_total']}, "
          f"Sahte {summary['fake_correct']}/{summary['fake_total']})")
    print(f"Rapor -> {OUT / 'wild_test.md'}")


if __name__ == "__main__":
    main()
