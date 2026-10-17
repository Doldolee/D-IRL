"""Draw the temporal diffusion architecture used in the release README."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def render(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "path", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(17, 8.6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 17), ylim=(0, 8.6))
    ax.axis("off")
    dark, gray = "#243345", "#667383"
    blue, wine, teal = "#426c91", "#8f3557", "#3c7669"

    def panel(x, y, w, h, title, body, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=.12",
                     linewidth=1.25, edgecolor=color, facecolor=fill))
        ax.text(x + w / 2, y + h * .73, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=dark)
        ax.text(x + w / 2, y + h * .34, body, ha="center", va="center",
                fontsize=11.5, linespacing=1.5, color=gray)

    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                     mutation_scale=14, linewidth=1.25, color=gray,
                     linestyle=(0, (4, 3)) if dashed else "-",
                     shrinkA=4, shrinkB=4))

    ax.text(.5, 8.22, "COHORT-SPECIFIC TRAINING", fontsize=11,
            color=gray, fontweight="bold", va="center")
    panel(.5, 4.95, 2.4, 1.7, "Driving windows",
          "Younger / older\ntrained separately", blue, "#f1f5f9")
    panel(3.25, 4.95, 2.7, 1.7, "Mixed-type corruption",
          "Numerical: Gaussian noise\nCategorical: mask noise", blue, "#f1f5f9")
    panel(6.3, 4.95, 2.1, 1.7, "Feature tokens",
          "Numerical + categorical\nembeddings", blue, "#f1f5f9")
    panel(12.75, 4.95, 3.65, 1.7, "Reconstruction heads",
          "EDM numerical values\nCategorical logits", teal, "#eff6f4")

    x, y, w, h = 8.75, 4.5, 3.65, 2.6
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=.12",
                 linewidth=1.5, edgecolor=wine, facecolor="#fbf3f6"))
    ax.text(x + w / 2, 6.73, "Transformer block × 2", ha="center", va="center",
            fontsize=13, fontweight="bold", color=dark)
    for by, label in ((5.8, "Feature attention"), (5.12, "Temporal attention")):
        ax.add_patch(FancyBboxPatch((9, by), 3.15, .48,
                     boxstyle="round,pad=.02,rounding_size=.07",
                     linewidth=.9, edgecolor=wine, facecolor="white"))
        ax.text(10.575, by + .24, label, ha="center", va="center",
                fontsize=12, color=wine)
    arrow((10.575, 5.79), (10.575, 5.61))
    ax.text(10.575, 4.76, "Hidden width 32 · 4 attention heads",
            ha="center", va="center", fontsize=10.5, color=gray)

    for a, b in ((2.9, 3.25), (5.95, 6.3), (8.4, 8.75), (12.4, 12.75)):
        arrow((a, 5.8), (b, 5.8))

    for bx, bw, text in (
        (3.25, 2.7, "Learned per-feature\nnoise schedules"),
        (6.3, 6.1, "Clean context (32 steps) · diffusion time\nPhase / progress conditions + position encoding"),
    ):
        ax.add_patch(FancyBboxPatch((bx, 7.46), bw, .6,
                     boxstyle="round,pad=.03,rounding_size=.08",
                     linewidth=.8, edgecolor="#b9c3ce", facecolor="#f7f9fb"))
        ax.text(bx + bw / 2, 7.76, text, ha="center", va="center",
                fontsize=10.7, linespacing=1.35, color=dark)
    arrow((4.6, 7.43), (4.6, 6.65))
    arrow((10.575, 7.43), (10.575, 7.1))

    ax.text(8.45, 4.08, "TRAINING OBJECTIVE", fontsize=11,
            color=gray, fontweight="bold", ha="center", va="center")
    ax.add_patch(FancyBboxPatch((.5, 2.93), 15.9, .82,
                 boxstyle="round,pad=.02,rounding_size=.1",
                 linewidth=.8, edgecolor="#d0d7df", facecolor="#f8f9fb"))
    ax.text(8.45, 3.34,
            r"$\mathcal{L}=\mathcal{L}_{\mathrm{num}}+\mathcal{L}_{\mathrm{cat}}"
            r"+0.1\mathcal{L}_{\Delta}+\mathcal{L}_{\mathrm{boundary}}"
            r"+0.1\mathcal{L}_{\mathrm{progress}}$",
            ha="center", va="center", fontsize=19, color=dark)
    arrow((1.7, 4.95), (1.7, 3.75), dashed=True)
    arrow((14.575, 4.95), (14.575, 3.75), dashed=True)

    ax.text(.5, 2.4, "SEGMENT-BY-SEGMENT GENERATION", fontsize=11,
            color=gray, fontweight="bold", va="center")
    panel(.5, .45, 2.4, 1.25, "Noise + plan",
          "Gaussian / mask tokens", blue, "#f1f5f9")
    panel(3.45, .45, 4.25, 1.25, "Iterative denoising",
          "Same Transformer · fixed clean context", wine, "#fbf3f6")
    panel(8.2, .45, 3, 1.25, "Next 128 transitions",
          "State · action · DQI", teal, "#eff6f4")
    panel(12.5, .45, 3.9, 1.25, "Full synthetic trajectory",
          "Append segments in temporal order", teal, "#eff6f4")
    for a, b in ((2.9, 3.45), (7.7, 8.2), (11.2, 12.5)):
        arrow((a, 1.075), (b, 1.075))
    ax.plot([9.7, 9.7, 5.575], [1.72, 2.08, 2.08], lw=1.1, color=gray)
    arrow((5.575, 2.08), (5.575, 1.7))
    ax.text(7.7, 2.34, "Last 32 transitions become the next context",
            ha="center", va="center", fontsize=10.5, color=gray,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    fig.canvas.draw()
    bounds = fig.get_window_extent()
    for label in ax.texts:
        box = label.get_window_extent(fig.canvas.get_renderer())
        if not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)):
            raise ValueError(f"Architecture label outside figure: {label.get_text()}")
    for extension in ("svg", "png"):
        fig.savefig(output / f"00_temporal_diffusion_overview.{extension}",
                    dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/synthetic_data/figures")
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
