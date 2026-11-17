"""把结构化语料渲染成 Markdown + PDF。

两种格式内容完全一致，用于验证「上传 PDF」的真实链路。
PDF 使用 reportlab 内置的 STSong-Light CJK 字体，不依赖系统中文字体。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config.settings import DOCS_DIR
from scripts.corpus_codes import build_doc as build_code_doc
from scripts.corpus_data import DOCS

ALL_DOCS = DOCS + [build_code_doc()]

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

S_TITLE = ParagraphStyle("title", fontName="STSong-Light", fontSize=18, leading=26,
                         alignment=1, spaceAfter=10)
S_H1 = ParagraphStyle("h1", fontName="STSong-Light", fontSize=15, leading=22, spaceBefore=6,
                      spaceAfter=8)
S_H2 = ParagraphStyle("h2", fontName="STSong-Light", fontSize=12.5, leading=19, spaceBefore=10,
                      spaceAfter=6)
S_H3 = ParagraphStyle("h3", fontName="STSong-Light", fontSize=11, leading=17, spaceBefore=7,
                      spaceAfter=4)
S_BODY = ParagraphStyle("body", fontName="STSong-Light", fontSize=10.5, leading=17,
                        firstLineIndent=21, spaceAfter=6, wordWrap="CJK")
S_CELL = ParagraphStyle("cell", fontName="STSong-Light", fontSize=8.5, leading=12)


def render_markdown(doc: dict) -> str:
    lines: list[str] = []
    for kind, payload in doc["blocks"]:
        if kind == "h1":
            lines.append(f"# {payload}\n")
        elif kind == "h2":
            lines.append(f"## {payload}\n")
        elif kind == "h3":
            lines.append(f"### {payload}\n")
        elif kind == "p":
            lines.append(f"{payload}\n")
        elif kind == "table":
            header, *rows = payload
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    return "\n".join(lines)


def render_pdf(doc: dict, out_path: Path) -> None:
    story: list = []
    for kind, payload in doc["blocks"]:
        if kind == "h1":
            story.append(Paragraph(payload, S_TITLE))
        elif kind == "h2":
            story.append(Paragraph(payload, S_H2))
        elif kind == "h3":
            story.append(Paragraph(payload, S_H3))
        elif kind == "p":
            story.append(Paragraph(payload, S_BODY))
        elif kind == "table":
            header, *rows = payload
            data = [[Paragraph(str(c), S_CELL) for c in header]]
            data += [[Paragraph(str(c), S_CELL) for c in row] for row in rows]
            ncol = len(header)
            width = (A4[0] - 40 * mm) / ncol
            table = Table(data, colWidths=[width] * ncol, repeatRows=1)
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEDED")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(Spacer(1, 4))
            story.append(table)
            story.append(Spacer(1, 8))

    SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=doc["title"], author="星域智能客服中心",
    ).build(story)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for doc in ALL_DOCS:
        md_path = DOCS_DIR / f"{doc['filename']}.md"
        pdf_path = DOCS_DIR / f"{doc['filename']}.pdf"
        md_path.write_text(render_markdown(doc), encoding="utf-8")
        render_pdf(doc, pdf_path)
        md_chars = len(md_path.read_text(encoding="utf-8"))
        print(f"{doc['filename']:<24} md={md_chars:>6} 字  pdf={pdf_path.stat().st_size / 1024:>6.1f} KB")
    print(f"\n语料目录：{DOCS_DIR}")


if __name__ == "__main__":
    main()
