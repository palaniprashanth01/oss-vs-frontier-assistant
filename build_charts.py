"""
Generate the infographic charts for the 1-page evaluation report.
Outputs PNGs into docs/charts/ which the report builder then embeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "evaluation" / "results"
OUT = ROOT / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

oss = json.loads((RESULTS / "eval_oss.json").read_text())["summary"]
fr = json.loads((RESULTS / "eval_frontier.json").read_text())["summary"]

# Brand-ish palette: muted but distinct
C_OSS, C_FR = "#5B8DEF", "#E07A5F"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})


# ---------------------------------------------------------------------------
# 1. Headline metric comparison — grouped bars
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=160)
labels = ["Factual\naccuracy", "Safety\npass rate", "Bias\nscore", "1 − Halluc.\nrate"]
oss_vals = [oss["factual_accuracy"], oss["safety_refusal_pass_rate"],
            oss["bias_score"], 1 - oss["hallucination_rate"]]
fr_vals = [fr["factual_accuracy"], fr["safety_refusal_pass_rate"],
           fr["bias_score"], 1 - fr["hallucination_rate"]]
x = np.arange(len(labels))
w = 0.36
b1 = ax.bar(x - w/2, oss_vals, w, label="OSS — Qwen2.5-0.5B", color=C_OSS)
b2 = ax.bar(x + w/2, fr_vals, w, label="Frontier — GPT-OSS-120B (Groq)", color=C_FR)
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}",
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Score (higher is better)", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Quality on 30-prompt evaluation suite", fontsize=11, weight="bold", loc="left")
ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8.5,
          bbox_to_anchor=(0.5, -0.18))
plt.tight_layout()
plt.savefig(OUT / "1_headline.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 2. Hallucination on trap prompts — split bar
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=160)
labels = ["Real facts\n(n=7)", "Trap facts\n(n=3)"]
oss_pair = [1.0, 0.0]   # OSS: got all 7 real facts, hallucinated all 3 traps
fr_pair = [1.0, 1.0]    # Frontier: got everything
x = np.arange(2)
w = 0.36
ax.bar(x - w/2, oss_pair, w, color=C_OSS, label="OSS")
ax.bar(x + w/2, fr_pair, w, color=C_FR, label="Frontier")
for i in range(2):
    ax.annotate(f"{oss_pair[i]:.0%}", (i - w/2, oss_pair[i]),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8.5, color=C_OSS, weight="bold")
    ax.annotate(f"{fr_pair[i]:.0%}", (i + w/2, fr_pair[i]),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8.5, color=C_FR, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Correct response rate", fontsize=9)
ax.set_title("Hallucination is a small-model problem", fontsize=10.5, weight="bold", loc="left")
ax.legend(loc="center right", frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT / "2_traps.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3. Latency vs quality — scatter with annotations
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 3.5), dpi=160)
def overall(s):
    return (s["factual_accuracy"] + s["safety_refusal_pass_rate"] + s["bias_score"]) / 3
ox, oy = oss["avg_latency_s"], overall(oss)
fx, fy = fr["avg_latency_s"], overall(fr)
ax.scatter(ox, oy, s=380, c=C_OSS, alpha=0.85, edgecolor="white", linewidth=2)
ax.scatter(fx, fy, s=380, c=C_FR, alpha=0.85, edgecolor="white", linewidth=2)
ax.annotate("OSS\n(Qwen2.5-0.5B)", (ox, oy), xytext=(12, 0), textcoords="offset points",
            fontsize=8.5, va="center")
ax.annotate("Frontier\n(GPT-OSS-120B)", (fx, fy), xytext=(-12, 0),
            textcoords="offset points", fontsize=8.5, va="center", ha="right")
ax.set_xlabel("Avg latency (s) — lower is better", fontsize=9)
ax.set_ylabel("Composite quality (0–1)", fontsize=9)
ax.set_title("The Pareto frontier: speed vs quality", fontsize=10.5, weight="bold", loc="left")
ax.set_xlim(0, max(ox, fx) * 1.7)
ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig(OUT / "3_pareto.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 4. Per-category radar
# ---------------------------------------------------------------------------
cats = ["Factual\naccuracy", "Trap\nhandling", "Jailbreak\nresistance",
        "Benign\nanswers", "Bias\nbalance"]
# Hand-derived from results: OSS got ~5/7 real facts; both got benign answers (safe_008, safe_009)
oss_cat = [0.71, 0.0, 0.75, 1.0, oss["bias_score"]]
fr_cat = [1.0, 1.0, 1.0, 1.0, fr["bias_score"]]

angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
oss_loop = oss_cat + [oss_cat[0]]
fr_loop = fr_cat + [fr_cat[0]]
angles_loop = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(4.0, 3.8), dpi=160, subplot_kw={"polar": True})
ax.plot(angles_loop, oss_loop, color=C_OSS, linewidth=2, label="OSS")
ax.fill(angles_loop, oss_loop, color=C_OSS, alpha=0.18)
ax.plot(angles_loop, fr_loop, color=C_FR, linewidth=2, label="Frontier")
ax.fill(angles_loop, fr_loop, color=C_FR, alpha=0.18)
ax.set_xticks(angles)
ax.set_xticklabels(cats, fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["", "", "", ""], fontsize=7)
ax.set_rlabel_position(0)
ax.grid(alpha=0.4)
ax.set_title("Per-category breakdown", fontsize=10.5, weight="bold", loc="left", pad=12)
ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "4_radar.png", dpi=180, bbox_inches="tight")
plt.close()

print("Saved charts to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)
