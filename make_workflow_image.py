"""Draw the workflow image for LinkedIn.

Pipeline: Ingest -> Retrieve -> Answer with citations -> Evaluate.
Day 1: ONLY Ingest is highlighted/active; the rest are greyed out.
Day 2: Ingest is marked done, Retrieve is highlighted/active,
Answer + Evaluate are greyed out.

Usage:
    python make_workflow_image.py            # Day 1 (default)
    python make_workflow_image.py --day 2    # Day 2
Output: assets/workflow-day<N>.png (1200 x 627, LinkedIn post size).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent

DAY_TITLES = {
    1: "Day 1 of 4 — Ingestion",
    2: "Day 2 of 4 — Retrieval",
    3: "Day 3 of 4 — Answers with citations",
    4: "Day 4 of 4 — Finished repo",
}

STEP_DEFS = [
    ("1. INGEST", "Documents in\nChunks + embeddings\nLocal vector index"),
    ("2. RETRIEVE", "Find the right\nchunks for a question"),
    ("3. ANSWER", "Answer with\ncitations to sources"),
    ("4. EVALUATE", "Score answers\nagainst a golden set"),
]


def draw(day: int) -> Path:
    out = ROOT / "assets" / f"workflow-day{day}.png"
    # state per step: "done" (green, tick), "active" (green, DONE TODAY),
    # "idle" (grey).
    states = []
    for i in range(4):
        if i < day - 1:
            states.append("done")
        elif i == day - 1:
            states.append("active")
        else:
            states.append("idle")

    active_bg, active_edge, active_text = "#1B5E20", "#1B5E20", "white"
    active_sub = "#DCEDC8"
    idle_bg, idle_edge, idle_text = "#EEEEEE", "#BDBDBD", "#9E9E9E"

    fig, ax = plt.subplots(figsize=(12, 6.27), dpi=100)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.27)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6, 5.85, "Document Q&A with Citations", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#212121")
    ax.text(6, 5.35, DAY_TITLES.get(day, f"Day {day} of 4"), ha="center", va="center",
            fontsize=14, color="#616161")

    box_w, box_h = 2.55, 2.6
    y0 = 1.55
    xs = [0.45, 3.30, 6.15, 9.00]
    for (title, sub), x, state in zip(STEP_DEFS, xs, states):
        lit = state in ("done", "active")
        bg = active_bg if lit else idle_bg
        edge = active_edge if lit else idle_edge
        ax.add_patch(FancyBboxPatch((x, y0), box_w, box_h,
                                     boxstyle="round,pad=0.08,rounding_size=0.15",
                                     facecolor=bg, edgecolor=edge, linewidth=2))
        ax.text(x + box_w / 2, y0 + box_h - 0.45, title, ha="center", va="center",
                fontsize=14, fontweight="bold",
                color=active_text if lit else idle_text)
        ax.text(x + box_w / 2, y0 + box_h / 2 - 0.35, sub, ha="center", va="center",
                fontsize=11.5, color=active_sub if lit else idle_text)
        if state == "active":
            ax.text(x + box_w / 2, y0 + 0.35, "✓ DONE TODAY", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#FFEB3B")
        elif state == "done":
            ax.text(x + box_w / 2, y0 + 0.35, "✓ DONE", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#FFEB3B")

    for x in (3.00, 5.85, 8.70):
        ax.add_patch(FancyArrowPatch((x, y0 + box_h / 2), (x + 0.28, y0 + box_h / 2),
                                     arrowstyle="-|>", mutation_scale=22,
                                     color="#616161", linewidth=2))

    ax.text(6, 0.85, "Built-in corpus: Kashmir Birds field guide (744 species)  +  upload your own PDFs",
            ha="center", va="center", fontsize=12, color="#616161")
    ax.text(6, 0.35, "Retrieval, answering and evaluation come on Days 2–4",
            ha="center", va="center", fontsize=12, color="#9E9E9E")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"saved {out} ({out.stat().st_size} bytes)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the workflow image for a build day.")
    parser.add_argument("--day", type=int, default=1, help="Build day (1-4).")
    args = parser.parse_args()
    draw(args.day)
