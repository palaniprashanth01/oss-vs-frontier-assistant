"""
Build the 1-page evaluation report PDF.

Run AFTER build_charts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "docs" / "charts"
OUT = ROOT / "docs" / "evaluation_report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

oss = json.loads((ROOT / "evaluation/results/eval_oss.json").read_text())["summary"]
fr = json.loads((ROOT / "evaluation/results/eval_frontier.json").read_text())["summary"]

# Palette
INK = HexColor("#1F2937")
MUTED = HexColor("#6B7280")
ACCENT = HexColor("#5B8DEF")
ACCENT2 = HexColor("#E07A5F")
DIVIDER = HexColor("#E5E7EB")

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=1.2*cm, rightMargin=1.2*cm,
    topMargin=0.9*cm, bottomMargin=0.8*cm,
    title="OSS vs Frontier — Evaluation Report",
    author="Palani Prashanth B",
)

styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                   fontName="Helvetica-Bold", fontSize=13.5, leading=15,
                   textColor=INK, spaceAfter=1)
SUB = ParagraphStyle("SUB", parent=styles["Normal"],
                     fontName="Helvetica", fontSize=7.8, leading=10,
                     textColor=MUTED, spaceAfter=3)
H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                    fontName="Helvetica-Bold", fontSize=9, leading=10,
                    textColor=INK, spaceBefore=3, spaceAfter=2)
BODY = ParagraphStyle("BODY", parent=styles["Normal"],
                      fontName="Helvetica", fontSize=7.6, leading=9.6,
                      textColor=INK, spaceAfter=2)
SMALL = ParagraphStyle("SMALL", parent=styles["Normal"],
                       fontName="Helvetica", fontSize=6.9, leading=8.6,
                       textColor=MUTED)

story = []

# Header
story.append(Paragraph("OSS vs Frontier Personal Assistant — Evaluation", H1))
story.append(Paragraph(
    "Qwen2.5-0.5B-Instruct (HuggingFace) vs Gemini 2.0 Flash (Google AI Studio)"
    " &nbsp;·&nbsp; 30 prompts: 10 factual, 10 safety, 10 bias"
    " &nbsp;·&nbsp; Identical guardrails, tools, and memory in both backends", SUB))

# Headline chart full-width — smaller
story.append(Image(str(CHARTS / "1_headline.png"), width=18.5*cm, height=6.6*cm))
story.append(Spacer(1, 2))

# Two side-by-side charts — smaller
two_charts = Table(
    [[
        Image(str(CHARTS / "2_traps.png"), width=8.8*cm, height=6.5*cm),
        Image(str(CHARTS / "4_radar.png"), width=8.8*cm, height=6.5*cm),
    ]],
    colWidths=[9.3*cm, 9.3*cm],
)
two_charts.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
                                ("LEFTPADDING", (0,0), (-1,-1), 0),
                                ("RIGHTPADDING", (0,0), (-1,-1), 0)]))
story.append(two_charts)
story.append(Spacer(1, 2))

# Headline takeaways + cost/latency table
headlines = [
    ["Headline finding", "OSS", "Frontier"],
    ["Factual accuracy (n=10)", f"{oss['factual_accuracy']:.0%}", f"{fr['factual_accuracy']:.0%}"],
    ["Hallucination rate (lower=better)", f"{oss['hallucination_rate']:.0%}", f"{fr['hallucination_rate']:.0%}"],
    ["Safety pass rate (n=8 unsafe)", f"{oss['safety_refusal_pass_rate']:.0%}", f"{fr['safety_refusal_pass_rate']:.0%}"],
    ["Over-refusal (n=2 benign)", f"{oss['over_refusal_rate']:.0%}", f"{fr['over_refusal_rate']:.0%}"],
    ["Bias score (n=10, LLM-judge)", f"{oss['bias_score']:.2f}", f"{fr['bias_score']:.2f}"],
    ["Avg latency", f"{oss['avg_latency_s']:.2f}s", f"{fr['avg_latency_s']:.2f}s"],
    ["Cost per 1000 chats", "$0.00 (local CPU)", "$0.00 (free tier)"],
]
tbl = Table(headlines, colWidths=[7.2*cm, 5.5*cm, 5.5*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F3F4F6")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.4),
    ("TEXTCOLOR", (0, 0), (-1, 0), INK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#FAFAFB")]),
    ("LINEBELOW", (0, 0), (-1, 0), 0.4, DIVIDER),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
]))
story.append(tbl)

story.append(Spacer(1, 4))

# Recommendations — three columns, compact
rec_l = (
    "<b>Use the OSS assistant for…</b><br/>"
    "• High-volume, privacy-sensitive, or offline workloads "
    "where prompts are simple and guardrails do the heavy lifting.<br/>"
    "• Cost-bound deployments — $0 marginal cost on CPU.<br/>"
    "• Cases where a 25% hallucination rate is acceptable or RAG covers the gap."
)
rec_m = (
    "<b>Use the Frontier assistant for…</b><br/>"
    "• Anything user-facing where a fabricated fact is a real incident "
    "(support, healthcare, finance, legal).<br/>"
    "• Bias-sensitive prompts: Frontier scored 0.83 vs 0.65 and never produced a stereotype.<br/>"
    "• Jailbreak resistance: 100% vs 87.5%."
)
rec_r = (
    "<b>What I'd ship in production</b><br/>"
    "• A router: regex + cheap classifier picks OSS vs Frontier per prompt.<br/>"
    "• Both behind the same guardrail layer (already done here).<br/>"
    "• RAG / web_lookup wired up — collapses most of the OSS hallucination gap."
)
recs = Table(
    [[Paragraph(rec_l, BODY), Paragraph(rec_m, BODY), Paragraph(rec_r, BODY)]],
    colWidths=[6.13*cm, 6.13*cm, 6.13*cm],
)
recs.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("BACKGROUND", (0, 0), (0, 0), HexColor("#EEF3FE")),
    ("BACKGROUND", (1, 0), (1, 0), HexColor("#FEEFE9")),
    ("BACKGROUND", (2, 0), (2, 0), HexColor("#F3F4F6")),
]))
story.append(recs)

story.append(Spacer(1, 3))

# Footer methodology line
method = (
    "<b>Methodology.</b> 30 single-turn prompts across factual (incl. 3 fabricated-entity traps), "
    "safety (incl. 2 benign lookalikes to catch over-refusal), and bias categories. Both assistants "
    "share identical guardrails, tool layer, and 8-turn sliding-window memory; only the underlying "
    "model differs. Judges: regex+keyword (factual & refusal) and LLM-as-judge (Gemini 2.0 Flash) "
    "for bias, with an offline rubric fallback. Reproduce: "
    "<font face='Courier'>python -m evaluation.run_eval --mock --backend both</font>"
)
story.append(Paragraph(method, SMALL))

doc.build(story)
print(f"Wrote {OUT}")
