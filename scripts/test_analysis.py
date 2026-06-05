"""FF++ test seti uzerinde derin analiz: ablation + hata analizi + threshold sweep.

Tek bir model gecisinde test setini skorlar, ham olasiliklari diske dumplar
ve uc rapor uretir:
  - reports/analysis/predictions.json     (per-video ham veriler)
  - reports/analysis/ablation.json        (fusion varyantlari karsilastirmasi)
  - reports/analysis/error_by_type.json   (FF++ manipulation tipi basina hata)
  - reports/analysis/threshold_sweep.json (spatial Acc-vs-threshold)
  - reports/analysis/*.png                (ablation, error, threshold grafikleri)

Calistir:
  python scripts/test_analysis.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import EER_THRESHOLDS, MODULE_AUC, STRONG_FAKE, _calibrate, fuse
from src.models.frequency.frequency_model import FrequencyModel
from src.models.spatial.spatial_model import SpatialModel
from src.preprocessing.dataset import DeepfakeVideoDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 8
OUT = ROOT / "reports" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def _tf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _load(name, model):
    ckpt = torch.load(f"checkpoints/best_{name}.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE).eval()


def _manip_type(vid: str, label: int) -> str:
    """FF++ klasor adina gore manipulasyon tipi.

    Real videolar: 'original' / 'youtube' / 'actors'  -> 'real'
    Fake: dosya adi tipiyle baslar -> 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'
    """
    if label == 0:
        return "real"
    return vid.split("_", 1)[0]


@torch.no_grad()
def score_test() -> list[dict]:
    """Test setini skorla, per-video predictions dondur."""
    tf = _tf()
    ds = DeepfakeVideoDataset(root_dir="data/processed", split="test")
    print(f"Cihaz: {DEVICE} | Test videolari: {len(ds)}", flush=True)

    spatial = _load("spatial", SpatialModel(lstm_hidden=256, lstm_layers=2, dropout=0.3))
    freq = _load("frequency", FrequencyModel(in_channels=3, feature_dim=256))

    rows = []
    for i in range(len(ds)):
        vid, paths, label = ds.samples[i]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        if not imgs:
            continue
        t = torch.stack([tf(im) for im in imgs]).to(DEVICE)

        n = len(imgs)
        if n >= SEQ_LEN:
            idx = np.linspace(0, n - 1, SEQ_LEN, dtype=int)
            seq = t[idx]
        else:
            seq = torch.cat([t, t[-1:].repeat(SEQ_LEN - n, 1, 1, 1)], 0)
        sp = float(torch.softmax(spatial(seq.unsqueeze(0)), 1)[0, 1])

        fr_probs = []
        for j in range(0, len(t), 16):
            fr_probs += torch.softmax(freq(t[j:j + 16]), 1)[:, 1].tolist()
        fr = float(np.mean(fr_probs))

        rows.append({
            "video_id": vid,
            "label": int(label),
            "type": _manip_type(vid, label),
            "p_spatial": sp,
            "p_frequency": fr,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)
    return rows


# ---------- Fusion varyantlari ----------

def verdict_threshold(p: float, th: float) -> int:
    return int(p >= th)


def verdict_uniform(sp: float, fr: float) -> int:
    cs = _calibrate(sp, EER_THRESHOLDS["spatial"])
    cf = _calibrate(fr, EER_THRESHOLDS["frequency"])
    return int(((cs + cf) / 2.0) >= 0.5)


def verdict_auc_weighted(sp: float, fr: float, with_strong: bool) -> int:
    f = fuse({"spatial": sp, "frequency": fr, "audio": None}) if with_strong else None
    if with_strong:
        return int(f["verdict"] == "FAKE")
    # AUC-weighted ama STRONG_FAKE override KAPALI
    cs = _calibrate(sp, EER_THRESHOLDS["spatial"])
    cf = _calibrate(fr, EER_THRESHOLDS["frequency"])
    ws = max(MODULE_AUC["spatial"] - 0.5, 1e-3) ** 2
    wf = max(MODULE_AUC["frequency"] - 0.5, 1e-3) ** 2
    fused = (cs * ws + cf * wf) / (ws + wf)
    return int(fused >= 0.5)


# ---------- 1) Ablation ----------

def ablation(rows: list[dict]) -> dict:
    y = np.array([r["label"] for r in rows])
    sp = np.array([r["p_spatial"] for r in rows])
    fr = np.array([r["p_frequency"] for r in rows])

    def metrics(y_pred, p_score=None):
        m = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "cm": confusion_matrix(y, y_pred).tolist(),
        }
        if p_score is not None:
            m["auc"] = float(roc_auc_score(y, p_score))
        return m

    variants = {}
    # 1. Spatial-only @ EER
    pred = (sp >= EER_THRESHOLDS["spatial"]).astype(int)
    variants["spatial_only"] = metrics(pred, sp)
    # 2. Frequency-only @ EER
    pred = (fr >= EER_THRESHOLDS["frequency"]).astype(int)
    variants["frequency_only"] = metrics(pred, fr)
    # 3. Uniform calibrated average
    pred = np.array([verdict_uniform(s, f) for s, f in zip(sp, fr)])
    variants["fusion_uniform"] = metrics(pred)
    # 4. AUC-weighted (no STRONG_FAKE)
    pred = np.array([verdict_auc_weighted(s, f, with_strong=False)
                     for s, f in zip(sp, fr)])
    variants["fusion_auc_weighted"] = metrics(pred)
    # 5. AUC-weighted + STRONG_FAKE override (production)
    pred = np.array([verdict_auc_weighted(s, f, with_strong=True)
                     for s, f in zip(sp, fr)])
    variants["fusion_auc_weighted_strongfake"] = metrics(pred)
    return variants


def plot_ablation(ab: dict, path: Path):
    names = ["spatial_only", "frequency_only",
             "fusion_uniform", "fusion_auc_weighted",
             "fusion_auc_weighted_strongfake"]
    labels = ["Spatial\n(EER)", "Frequency\n(EER)",
              "Uniform\ncalibre", "AUC-ağırlıklı",
              "AUC-ağırlıklı\n+ STRONG_FAKE"]
    acc = [ab[n]["accuracy"] * 100 for n in names]
    f1 = [ab[n]["f1"] * 100 for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, a = plt.subplots(figsize=(10, 4.6))
    b1 = a.bar(x - w/2, acc, w, label="Accuracy %", color="#3fb950")
    b2 = a.bar(x + w/2, f1, w, label="F1 % (sahte)", color="#1f6feb")
    a.set_xticks(x)
    a.set_xticklabels(labels, fontsize=9)
    a.set_ylim(0, 105)
    a.set_ylabel("%")
    a.set_title("Ablation — Modül kombinasyonu ve füzyon stratejisinin etkisi",
                fontsize=11, weight="bold")
    a.legend()
    a.grid(axis="y", alpha=.25)
    for bars in (b1, b2):
        for r in bars:
            a.text(r.get_x() + r.get_width()/2, r.get_height() + 0.6,
                   f"{r.get_height():.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- 2) Error analizi ----------

def error_by_type(rows: list[dict]) -> dict:
    """Her FF++ tipi (real + 4 fake) icin production fusion'in performansi."""
    by_type = defaultdict(list)
    for r in rows:
        verdict = verdict_auc_weighted(r["p_spatial"], r["p_frequency"], with_strong=True)
        r2 = dict(r)
        r2["pred"] = verdict
        by_type[r["type"]].append(r2)

    summary = {}
    for t, items in by_type.items():
        y = np.array([x["label"] for x in items])
        p = np.array([x["pred"] for x in items])
        sp = np.array([x["p_spatial"] for x in items])
        fr = np.array([x["p_frequency"] for x in items])
        # Fake icin: recall (saldiri yakalama). Real icin: specificity (= acc).
        if t == "real":
            metric = {
                "n": int(len(y)),
                "specificity": float(np.mean(p == 0)),
                "false_positive": int(np.sum(p == 1)),
                "p_spatial_mean": float(sp.mean()),
                "p_frequency_mean": float(fr.mean()),
            }
        else:
            metric = {
                "n": int(len(y)),
                "recall": float(np.mean(p == 1)),
                "false_negative": int(np.sum(p == 0)),
                "p_spatial_mean": float(sp.mean()),
                "p_frequency_mean": float(fr.mean()),
            }
        summary[t] = metric
    return summary


def plot_error(err: dict, path: Path):
    fake_types = [t for t in err if t != "real"]
    fake_types.sort()
    recalls = [err[t]["recall"] * 100 for t in fake_types]
    fns = [err[t]["false_negative"] for t in fake_types]
    ns = [err[t]["n"] for t in fake_types]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    bars = ax[0].bar(fake_types, recalls, color=["#3fb950", "#1f6feb", "#bb8009", "#a371f7"])
    ax[0].set_ylim(0, 105)
    ax[0].set_ylabel("Recall %")
    ax[0].set_title("FF++ Manipulasyon Tipine Göre Recall", weight="bold")
    ax[0].grid(axis="y", alpha=.25)
    for r, n in zip(bars, ns):
        ax[0].text(r.get_x() + r.get_width()/2, r.get_height() + 0.8,
                   f"{r.get_height():.1f}%\n(n={n})", ha="center", fontsize=9)

    ax[1].bar(fake_types, fns, color="#f85149")
    ax[1].set_ylabel("Kaçırılan (FN) sayısı")
    ax[1].set_title("Tip Başına Yanlış Negatif", weight="bold")
    ax[1].grid(axis="y", alpha=.25)
    for i, v in enumerate(fns):
        ax[1].text(i, v + 0.2, str(v), ha="center", fontsize=10)

    # Real bar (specificity bilgisi)
    if "real" in err:
        spec = err["real"]["specificity"] * 100
        fig.suptitle(f"Üretim füzyonu (AUC-ağırlık + STRONG_FAKE) — "
                     f"Real specificity: %{spec:.2f}  (n={err['real']['n']}, "
                     f"FP={err['real']['false_positive']})",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- 3) Threshold sweep ----------

def threshold_sweep(rows: list[dict]) -> dict:
    y = np.array([r["label"] for r in rows])
    sp = np.array([r["p_spatial"] for r in rows])
    fr = np.array([r["p_frequency"] for r in rows])

    ths = np.linspace(0.05, 0.95, 91)
    out = {"spatial": [], "frequency": []}
    for th in ths:
        pred_sp = (sp >= th).astype(int)
        pred_fr = (fr >= th).astype(int)
        out["spatial"].append({
            "th": float(th),
            "accuracy": float(accuracy_score(y, pred_sp)),
            "recall": float(recall_score(y, pred_sp, zero_division=0)),
            "precision": float(precision_score(y, pred_sp, zero_division=0)),
            "f1": float(f1_score(y, pred_sp, zero_division=0)),
        })
        out["frequency"].append({
            "th": float(th),
            "accuracy": float(accuracy_score(y, pred_fr)),
            "recall": float(recall_score(y, pred_fr, zero_division=0)),
            "precision": float(precision_score(y, pred_fr, zero_division=0)),
            "f1": float(f1_score(y, pred_fr, zero_division=0)),
        })

    # Best Acc, Best F1
    def best(records, key):
        return max(records, key=lambda r: r[key])

    out["spatial_best_acc"] = best(out["spatial"], "accuracy")
    out["spatial_best_f1"] = best(out["spatial"], "f1")
    out["frequency_best_acc"] = best(out["frequency"], "accuracy")
    out["frequency_best_f1"] = best(out["frequency"], "f1")
    out["spatial_eer"] = EER_THRESHOLDS["spatial"]
    out["frequency_eer"] = EER_THRESHOLDS["frequency"]
    return out


def plot_threshold(sw: dict, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, name in enumerate(["spatial", "frequency"]):
        recs = sw[name]
        ths = [r["th"] for r in recs]
        acc = [r["accuracy"] * 100 for r in recs]
        f1 = [r["f1"] * 100 for r in recs]
        rec = [r["recall"] * 100 for r in recs]
        eer = sw[f"{name}_eer"]
        best_acc = sw[f"{name}_best_acc"]
        ax[i].plot(ths, acc, label="Accuracy", color="#3fb950", lw=2)
        ax[i].plot(ths, f1, label="F1 (sahte)", color="#1f6feb", lw=2)
        ax[i].plot(ths, rec, label="Recall (sahte)", color="#bb8009", lw=1.5, ls="--")
        ax[i].axvline(eer, color="#f85149", ls=":", lw=1.5,
                      label=f"EER eşiği = {eer:.3f}")
        ax[i].axvline(best_acc["th"], color="#888", ls=":", lw=1.5,
                      label=f"Max-Acc eşiği = {best_acc['th']:.3f}")
        ax[i].set_xlabel("Karar eşiği")
        ax[i].set_ylabel("%")
        ax[i].set_title(f"{name.capitalize()} — Eşik Süpürmesi", weight="bold")
        ax[i].legend(fontsize=8, loc="lower center")
        ax[i].grid(alpha=.25)
    fig.suptitle("EER eşiği recall-odaklı; max-acc eşiği farklı bir denge — eşik seçimi "
                 "AUC'tan ayrıdır", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- main ----------

def main():
    pred_path = OUT / "predictions.json"
    if pred_path.exists():
        print(f"Onbellek bulundu -> {pred_path}", flush=True)
        rows = json.loads(pred_path.read_text(encoding="utf-8"))
    else:
        rows = score_test()
        pred_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"Kaydedildi -> {pred_path}", flush=True)

    print("\n[1/3] Ablation...", flush=True)
    ab = ablation(rows)
    (OUT / "ablation.json").write_text(
        json.dumps(ab, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_ablation(ab, OUT / "ablation.png")

    print("[2/3] Error by type...", flush=True)
    err = error_by_type(rows)
    (OUT / "error_by_type.json").write_text(
        json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_error(err, OUT / "error_by_type.png")

    print("[3/3] Threshold sweep...", flush=True)
    sw = threshold_sweep(rows)
    (OUT / "threshold_sweep.json").write_text(
        json.dumps(sw, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_threshold(sw, OUT / "threshold_sweep.png")

    print("\n=== OZET ===")
    print("Ablation (accuracy %):")
    for k, v in ab.items():
        print(f"  {k:42s}  acc={v['accuracy']*100:5.2f}  f1={v['f1']*100:5.2f}")
    print("\nManipulation tipi basina (production fusion):")
    for k, v in err.items():
        if k == "real":
            print(f"  {k:18s} n={v['n']:3d}  specificity={v['specificity']*100:5.2f}  FP={v['false_positive']}")
        else:
            print(f"  {k:18s} n={v['n']:3d}  recall    ={v['recall']*100:5.2f}  FN={v['false_negative']}")
    print(f"\nSpatial: EER={sw['spatial_eer']:.3f} -> acc={[r for r in sw['spatial'] if abs(r['th']-sw['spatial_eer'])<0.01][0]['accuracy']*100:.2f}")
    print(f"         Max-Acc th={sw['spatial_best_acc']['th']:.3f} -> acc={sw['spatial_best_acc']['accuracy']*100:.2f}")
    print(f"Frequency: EER={sw['frequency_eer']:.3f}, Max-Acc th={sw['frequency_best_acc']['th']:.3f}")


if __name__ == "__main__":
    main()
