"""docs/proje_onerisi.md -> docs/232923023_BatuhanKaya.pdf

Turkce karakterli ic-baglantilar (#anchor) markdown-pdf'i kirdigi icin
once duz metne cevrilir. Calistir: python scripts/build_pdf.py
"""

import re
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "proje_onerisi.md"
OUT = ROOT / "docs" / "232923023_BatuhanKaya.pdf"

CSS = """
body { font-size: 10pt; line-height: 1.35; color: #000; }
h1 { font-size: 15pt; }
h2 { font-size: 12pt; margin-top: 16px; }
h3 { font-size: 11pt; margin-top: 12px; }
h4 { font-size: 10pt; }
table { border-collapse: collapse; margin: 6px 0; width: auto; }
th, td { border: 1px solid #000; padding: 3px 6px; text-align: left;
        vertical-align: middle; font-weight: normal; }
th { font-weight: normal; }
img { max-width: 320px; height: auto; }
table.layout, table.layout td { border: none; padding: 4px 8px;
        text-align: center; vertical-align: top; }
"""


def main():
    md = SRC.read_text(encoding="utf-8")
    # [metin](#turkce-anchor) -> metin  (disa baglantilari koru)
    md = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", md)

    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(md, toc=False, root=str(ROOT)), user_css=CSS)
    pdf.meta["title"] = "Deepfake Tespit Sistemi - Proje Onerisi"
    pdf.meta["author"] = "Batuhan Kaya"
    pdf.save(str(OUT))
    print(f"PDF -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
