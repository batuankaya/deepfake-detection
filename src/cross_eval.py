"""In-dataset (FF++ test) ve cross-dataset (Celeb-DF v2) degerlendirmesini
sirayla yapip karsilastirma raporu uretir.

Kullanim:
    python -m src.cross_eval --checkpoint checkpoints/best_frequency.pt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_eval(checkpoint: str, split: str, batch_size: int, num_workers: int) -> Path:
    """src.evaluate'i alt-process olarak calistirir, rapor klasorunu doner."""
    out_dir = Path("reports") / f"{Path(checkpoint).stem}_{split}"
    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--checkpoint", checkpoint,
        "--split", split,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
    ]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_dir


def load_metrics(report_dir: Path) -> dict:
    with open(report_dir / "metrics.json") as f:
        return json.load(f)


def print_comparison(in_metrics: dict, cross_metrics: dict, out_path: Path):
    rows = []
    for key in ("accuracy", "f1", "auc_roc", "eer"):
        in_v = in_metrics["metrics"][key]
        cross_v = cross_metrics["metrics"][key]
        delta = cross_v - in_v
        rows.append((key, in_v, cross_v, delta))

    lines = []
    lines.append("# Cross-Dataset Karsilastirma Raporu\n")
    lines.append(f"**Checkpoint:** `{in_metrics['checkpoint']}`")
    lines.append(f"**Modul:** {in_metrics['module']}\n")
    lines.append("| Metrik | In-dataset (FF++ test) | Cross-dataset (Celeb-DF v2) | Fark |")
    lines.append("|--------|-----------------------:|----------------------------:|-----:|")
    for key, iv, cv, dv in rows:
        sign = "+" if dv >= 0 else ""
        lines.append(f"| {key:10s} | {iv:.4f} | {cv:.4f} | {sign}{dv:.4f} |")
    lines.append("")
    lines.append(f"**In-dataset ornek sayisi:** {in_metrics['num_samples']}")
    lines.append(f"**Cross-dataset ornek sayisi:** {cross_metrics['num_samples']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nKarsilastirma raporu -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="In-dataset vs cross-dataset karsilastirma")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    in_dir = run_eval(args.checkpoint, "test", args.batch_size, args.num_workers)
    cross_dir = run_eval(args.checkpoint, "cross_celebdf", args.batch_size, args.num_workers)

    in_m = load_metrics(in_dir)
    cross_m = load_metrics(cross_dir)

    comp_path = Path("reports") / f"{Path(args.checkpoint).stem}_comparison.md"
    print_comparison(in_m, cross_m, comp_path)


if __name__ == "__main__":
    main()
