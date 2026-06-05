"""Reliability diagram + bootstrap CI — reports/analysis/predictions.json'dan.

predictions.json'daki (label, p_spatial, p_frequency) uzerinden:
  - Reliability (kalibrasyon) egrisi + Brier + ECE — spatial ve frequency
  - Production fusion verdict + spatial + frequency icin %95 bootstrap CI
    (Accuracy, F1, AUC)

Cikti: reports/analysis/calibration.{json,png}
       reports/analysis/bootstrap_ci.json
Calistir: python scripts/calibration_ci.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import EER_THRESHOLDS, _calibrate, fuse

OUT = ROOT / "reports" / "analysis"
PRED = OUT / "predictions.json"


def expected_calibration_error(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = p[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def plot_reliability(y, probs: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "k:", label="Mükemmel kalibrasyon", lw=1.2)
    colors = {"spatial": "#1f6feb", "frequency": "#bb8009"}
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y, p)
        ece = expected_calibration_error(y, p)
        ax.plot(mean_pred, frac_pos, "o-", color=colors[name], lw=2,
                label=f"{name}  Brier={brier:.3f}  ECE={ece:.3f}")
    ax.set_xlabel("Tahmin edilen olasılık")
    ax.set_ylabel("Gözlenen sahte oranı")
    ax.set_title("Reliability Diagram — model olasılıkları gerçekten kalibre mi?",
                 weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def bootstrap_ci(y_true, y_pred, y_score, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # tek sinif geldi
            continue
        accs.append(accuracy_score(yt, y_pred[idx]))
        f1s.append(f1_score(yt, y_pred[idx], zero_division=0))
        if y_score is not None:
            aucs.append(roc_auc_score(yt, y_score[idx]))

    def ci(vals):
        if not vals:
            return None
        a = np.array(vals)
        return {
            "mean": float(a.mean()),
            "ci95_low": float(np.percentile(a, 2.5)),
            "ci95_high": float(np.percentile(a, 97.5)),
        }
    return {"accuracy": ci(accs), "f1": ci(f1s), "auc": ci(aucs)}


def main():
    rows = json.loads(PRED.read_text(encoding="utf-8"))
    y = np.array([r["label"] for r in rows])
    sp = np.array([r["p_spatial"] for r in rows])
    fr = np.array([r["p_frequency"] for r in rows])

    # --- Reliability ---
    cal_data = {
        "n": int(len(y)),
        "spatial": {
            "brier": float(brier_score_loss(y, sp)),
            "ece10": expected_calibration_error(y, sp),
        },
        "frequency": {
            "brier": float(brier_score_loss(y, fr)),
            "ece10": expected_calibration_error(y, fr),
        },
    }
    (OUT / "calibration.json").write_text(
        json.dumps(cal_data, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_reliability(y, {"spatial": sp, "frequency": fr}, OUT / "calibration.png")
    print("Reliability:", json.dumps(cal_data, indent=2, ensure_ascii=False))

    # --- Bootstrap CI ---
    pred_sp = (sp >= EER_THRESHOLDS["spatial"]).astype(int)
    pred_fr = (fr >= EER_THRESHOLDS["frequency"]).astype(int)
    pred_fusion = np.array([
        1 if fuse({"spatial": float(s), "frequency": float(f), "audio": None})["verdict"] == "FAKE" else 0
        for s, f in zip(sp, fr)
    ])

    ci_data = {
        "n_videos": int(len(y)),
        "n_boot": 1000,
        "spatial_only_EER": bootstrap_ci(y, pred_sp, sp),
        "frequency_only_EER": bootstrap_ci(y, pred_fr, fr),
        "fusion_production": bootstrap_ci(y, pred_fusion, None),
    }
    (OUT / "bootstrap_ci.json").write_text(
        json.dumps(ci_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nBootstrap %95 CI:")
    for k, v in ci_data.items():
        if isinstance(v, dict) and "accuracy" in v:
            a, f, u = v["accuracy"], v["f1"], v["auc"]
            print(f"  {k:24s} acc={a['mean']*100:5.2f}% [{a['ci95_low']*100:5.2f}, {a['ci95_high']*100:5.2f}]"
                  f"   f1={f['mean']*100:5.2f}% [{f['ci95_low']*100:5.2f}, {f['ci95_high']*100:5.2f}]"
                  + (f"   auc={u['mean']:.4f} [{u['ci95_low']:.4f}, {u['ci95_high']:.4f}]" if u else ""))


if __name__ == "__main__":
    main()
