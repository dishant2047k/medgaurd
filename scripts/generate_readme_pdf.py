"""
Generate README.pdf from README.md using a lightweight Markdown-to-PDF renderer.

This keeps the repo's PDF guide aligned with the editable Markdown README.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
README_MD = ROOT / "README.md"
README_PDF = ROOT / "README.pdf"


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReadmeTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0b3954"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReadmeH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0b3954"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReadmeH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#145374"),
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReadmeBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReadmeBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14,
            leftIndent=14,
            firstLineIndent=0,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReadmeCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.8,
            leading=11,
            leftIndent=10,
            rightIndent=10,
            borderPadding=6,
            backColor=colors.HexColor("#f3f6f8"),
            borderColor=colors.HexColor("#d6dee4"),
            borderWidth=0.5,
            borderRadius=None,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles


def convert_inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def flush_paragraph(buffer, story, styles):
    if not buffer:
        return
    text = " ".join(line.strip() for line in buffer).strip()
    if text:
        story.append(Paragraph(convert_inline(text), styles["ReadmeBody"]))
    buffer.clear()


def flush_list(items, story, styles, ordered=False):
    if not items:
        return
    list_kwargs = {
        "leftIndent": 18,
    }
    if ordered:
        list_kwargs["bulletType"] = "1"
        list_kwargs["start"] = "1"
    else:
        list_kwargs["bulletType"] = "bullet"

    flowable = ListFlowable(
        [ListItem(Paragraph(convert_inline(item), styles["ReadmeBullet"])) for item in items],
        **list_kwargs,
    )
    story.append(flowable)
    story.append(Spacer(1, 6))
    items.clear()


def render_markdown_to_story(markdown_text: str, styles):
    story = []
    paragraph_buffer = []
    bullet_items = []
    numbered_items = []
    in_code_block = False
    code_lines = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(bullet_items, story, styles, ordered=False)
            flush_list(numbered_items, story, styles, ordered=True)
            if in_code_block:
                story.append(Preformatted("\n".join(code_lines), styles["ReadmeCode"]))
                code_lines.clear()
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(bullet_items, story, styles, ordered=False)
            flush_list(numbered_items, story, styles, ordered=True)
            story.append(Spacer(1, 2))
            continue

        if stripped.startswith("# "):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(bullet_items, story, styles, ordered=False)
            flush_list(numbered_items, story, styles, ordered=True)
            story.append(Paragraph(convert_inline(stripped[2:].strip()), styles["ReadmeTitle"]))
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(bullet_items, story, styles, ordered=False)
            flush_list(numbered_items, story, styles, ordered=True)
            story.append(Paragraph(convert_inline(stripped[3:].strip()), styles["ReadmeH1"]))
            continue

        if stripped.startswith("### "):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(bullet_items, story, styles, ordered=False)
            flush_list(numbered_items, story, styles, ordered=True)
            story.append(Paragraph(convert_inline(stripped[4:].strip()), styles["ReadmeH2"]))
            continue

        if stripped.startswith("- "):
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(numbered_items, story, styles, ordered=True)
            bullet_items.append(stripped[2:].strip())
            continue

        numbered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if numbered_match:
            flush_paragraph(paragraph_buffer, story, styles)
            flush_list(bullet_items, story, styles, ordered=False)
            numbered_items.append(numbered_match.group(1).strip())
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph(paragraph_buffer, story, styles)
    flush_list(bullet_items, story, styles, ordered=False)
    flush_list(numbered_items, story, styles, ordered=True)

    if in_code_block and code_lines:
        story.append(Preformatted("\n".join(code_lines), styles["ReadmeCode"]))

    return story


def main():
    styles = build_styles()
    markdown_text = README_MD.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(README_PDF),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="MedGuard AI README",
        author="OpenAI Codex",
    )
    story = render_markdown_to_story(markdown_text, styles)
    doc.build(story)
    print(f"Generated {README_PDF}")


if __name__ == "__main__":
    main()
