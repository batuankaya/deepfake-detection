"""reports/ klasorundeki tum metrics.json dosyalarini toplayip
Markdown ve LaTeX-uyumlu birlesik rapor uretir.

Kullanim:
    python -m src.utils.generate_report
    python -m src.utils.generate_report --out reports/SUMMARY.md
"""

import argparse
import json
from pathlib import Path

REPORTS_DIR = Path("reports")


def load_all_metrics() -> list[dict]:
    """reports/*/metrics.json dosyalarini yukler."""
    results = []
    if not REPORTS_DIR.exists():
        return results
    for metrics_file in sorted(REPORTS_DIR.rglob("metrics.json")):
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            data["_path"] = str(metrics_file.parent)
            results.append(data)
        except Exception as exc:
            print(f"  [HATA] {metrics_file}: {exc}")
    return results


def format_markdown(results: list[dict]) -> str:
    """Tum sonuclari tek tabloya basar."""
    lines = ["# Egitim ve Degerlendirme Ozeti", ""]
    lines.append("| Modul | Split | N | Acc | F1 | AUC | EER |")
    lines.append("|-------|-------|--:|----:|---:|----:|----:|")
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r.get('module', '-'):10s} "
            f"| {r.get('split', '-'):12s} "
            f"| {r.get('num_samples', 0):,} "
            f"| {m.get('accuracy', 0):.4f} "
            f"| {m.get('f1', 0):.4f} "
            f"| {m.get('auc_roc', 0):.4f} "
            f"| {m.get('eer', 0):.4f} |"
        )
    lines.append("")
    lines.append("**Klasor:** `reports/`")
    lines.append("")
    lines.append("## Detaylar")
    for r in results:
        lines.append(f"- `{r['_path']}` — modul={r.get('module')}, "
                     f"split={r.get('split')}, "
                     f"checkpoint={Path(r.get('checkpoint','')).name}")
    return "\n".join(lines)


def format_latex(results: list[dict]) -> str:
    """LaTeX longtable formatinda rapor (tez icin)."""
    lines = [
        r"\begin{longtable}{l l r r r r r}",
        r"\toprule",
        r"Modul & Split & N & Acc & F1 & AUC & EER \\",
        r"\midrule",
        r"\endhead",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"{r.get('module','-')} & {r.get('split','-')} & "
            f"{r.get('num_samples',0):,} & "
            f"{m.get('accuracy',0):.4f} & {m.get('f1',0):.4f} & "
            f"{m.get('auc_roc',0):.4f} & {m.get('eer',0):.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Birlesik rapor uret")
    parser.add_argument("--out", type=str, default="reports/SUMMARY.md")
    parser.add_argument("--latex", type=str, default=None,
                        help="LaTeX cikti dosyasi (opsiyonel)")
    args = parser.parse_args()

    results = load_all_metrics()
    if not results:
        print("Hic metrics.json bulunamadi. Once src.evaluate calistir.")
        return

    md = format_markdown(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nMarkdown -> {out_path}")

    if args.latex:
        tex = format_latex(results)
        latex_path = Path(args.latex)
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(tex, encoding="utf-8")
        print(f"LaTeX -> {latex_path}")


if __name__ == "__main__":
    main()
