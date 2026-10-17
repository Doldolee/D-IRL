"""Draw the trajectory-based temporal diffusion overview used in the README."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
import numpy as np


GOLD = "#cba32c"
GRAY = "#d6d7d9"
BLACK = "#161616"


def render(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(15, 11.4))
    fig.subplots_adjust(left=.01, right=.99, bottom=.015, top=.99)
    ax.set(xlim=(0, 15), ylim=(0, 11.4))
    ax.axis("off")

    def text(x, y, label, size=13, ha="center", **kw):
        return ax.text(x, y, label, fontsize=size, ha=ha, va="center",
                       color=BLACK, linespacing=1.25, **kw)

    def arrow(start, end, lw=1.6):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                     mutation_scale=16, linewidth=lw, color=BLACK,
                     shrinkA=2, shrinkB=2))

    def connector(points, lw=1.35):
        points = np.asarray(points)
        ax.plot(points[:-1, 0], points[:-1, 1], color=BLACK, lw=lw)
        arrow(points[-2], points[-1], lw)

    def slabs(x, y, w, h, label, *, vertical=False, size=15):
        for offset in (.12, 0):
            ax.add_patch(FancyBboxPatch((x + offset, y + offset), w, h,
                         boxstyle="round,pad=.03,rounding_size=.19",
                         facecolor=GRAY, edgecolor=BLACK, linewidth=1.6))
        text(x + w / 2, y + h / 2, label, size,
             rotation=90 if vertical else 0)

    def trajectory(x, y, w=1.9, h=1.45, *, mode="clean", full=False, seed=0):
        """Draw a schematic XY projection; state/action/DQI are not inferred."""
        rng = np.random.default_rng(seed)
        if full:
            anchors = np.array([[.03, .08], [.21, .10], [.24, .16], [.23, .69],
                                [.30, .76], [.68, .96], [.76, .87], [.76, .59],
                                [.96, .12], [1.0, .10]])
        else:
            anchors = np.array([[.05, .08], [.07, .42], [.11, .55], [.28, .66],
                                [.58, .86], [.76, .94], [.90, .80], [.93, .56]])
        distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(anchors, axis=0), axis=1))]
        u = np.linspace(0, distances[-1], 56)
        p = np.column_stack([np.interp(u, distances, anchors[:, j]) for j in (0, 1)])
        if mode == "noise":
            p = np.column_stack([np.linspace(.05, .95, 36), rng.uniform(.10, .92, 36)])
        elif mode == "corrupted":
            p += rng.normal(0, .055, p.shape)
            p = p.clip(.02, .98)
        p = p * [w, h] + [x, y]
        color = GOLD if mode == "generated" else "#4e5257"
        if mode == "generated" and full:
            edges = np.stack([p[:-1], p[1:]], axis=1)
            colors = np.array(["#b48c21", "#d0ad43", "#e0c779"])
            group = np.minimum(np.arange(len(edges)) * 3 // len(edges), 2)
            ax.add_collection(LineCollection(edges, colors=colors[group], linewidths=3.0,
                                             capstyle="round", joinstyle="round"))
        else:
            ax.plot(p[:, 0], p[:, 1], color=color, lw=2.5 if mode != "noise" else 1.4,
                    solid_capstyle="round", alpha=.70 if mode == "noise" else 1.)
        ax.scatter(p[::3, 0], p[::3, 1], s=15, facecolor="white", edgecolor=color,
                   linewidth=.8, zorder=4)
        if mode in ("corrupted", "noise"):
            masked = np.arange(4, len(p), 9)
            ax.scatter(p[masked, 0], p[masked, 1], s=22, marker="s", c=BLACK, zorder=5)
        elif mode == "generated":
            # Small movement arrows identify a path rather than a scalar signal.
            at = len(p) // 2
            ax.add_patch(FancyArrowPatch(p[at], p[at + 3], arrowstyle="-|>",
                         mutation_scale=10, color=color, lw=1.25, zorder=6))

    text(.18, 10.98, "a", 29, ha="left", fontweight="bold")
    text(.80, 10.98, "Temporal diffusion training", 20, ha="left")

    trajectory(.64, 7.52, seed=3)
    text(1.63, 9.50, "Trajectory segment", 16)
    text(1.63, 6.99, "State · action · DQI", 12)
    trajectory(3.53, 7.52, mode="corrupted", seed=12)
    text(4.49, 9.50, "Noisy segment", 16)
    text(4.49, 6.99, "Gaussian noise\n+ categorical masks", 12)
    arrow((2.72, 8.19), (3.32, 8.19))

    text(4.49, 10.47, "Learnable per-feature\nnoise schedules", 12)
    arrow((4.49, 10.10), (4.49, 9.80))

    for i in range(6):
        ax.add_patch(Rectangle((6.14, 7.25 + i * .25), .27, .17,
                              edgecolor=BLACK, facecolor="#a6a8ab", lw=.65))
    text(6.28, 9.50, "Feature\ntokens", 13)
    arrow((5.62, 8.19), (6.02, 8.19))
    arrow((6.54, 8.19), (7.00, 8.19))
    slabs(7.10, 6.90, 1.42, 2.40, "Temporal diffusion\nTransformer", vertical=True)
    text(7.89, 6.38, "2 axial blocks\nFeature ↔ temporal attention", 12)
    text(7.89, 5.85, "Width 32 · 4 heads", 11.5)
    text(7.89, 10.47, "Clean context: preceding 32 steps\nTime · phase · progress", 12)
    arrow((7.89, 10.10), (7.89, 9.52))

    slabs(9.35, 7.62, 1.17, 1.10, "EDM\n+ logits", size=13)
    text(9.99, 9.50, "Reconstruction\nheads", 13)
    arrow((8.79, 8.19), (9.23, 8.19))
    arrow((10.77, 8.19), (11.36, 8.19))
    trajectory(11.48, 7.52, mode="generated", seed=6)
    text(12.43, 9.50, "Reconstructed\nsegment", 16)
    text(12.43, 6.99, "State · action · DQI", 12)

    # Explicit solid paths replace the earlier unexplained dotted loss arrows.
    connector([(1.63, 6.64), (1.63, 5.32), (5.10, 5.32)])
    text(3.30, 5.61, "Clean target", 12)
    connector([(12.43, 6.64), (12.43, 5.32), (9.74, 5.32)])
    text(11.12, 5.61, "Reconstruction", 12)
    ax.add_patch(FancyBboxPatch((5.20, 5.03), 4.43, .58,
                 boxstyle="round,pad=.02,rounding_size=.08",
                 edgecolor=BLACK, facecolor="#fafafa", lw=1.2))
    text(7.415, 5.32, "Compare with clean segment", 13)
    arrow((7.415, 5.00), (7.415, 4.57), lw=1.35)
    text(7.42, 4.26,
         r"$\mathcal{L}=\mathcal{L}_{\mathrm{num}}+\mathcal{L}_{\mathrm{cat}}"
         r"+0.1\mathcal{L}_{\Delta}+\mathcal{L}_{\mathrm{boundary}}"
         r"+0.1\mathcal{L}_{\mathrm{progress}}$", 18)
    ax.plot([.65, 14.30], [3.80, 3.80], color="#b6b6b6", lw=.75)

    text(.18, 3.37, "b", 29, ha="left", fontweight="bold")
    text(.80, 3.37, "Segment-by-segment trajectory generation", 19, ha="left")
    trajectory(.80, .66, w=1.83, h=1.24, mode="noise", seed=16)
    text(1.73, 2.31, "Noise + plan", 15)
    text(1.73, .30, "Gaussian / mask inputs", 11.5)
    arrow((2.85, 1.25), (3.79, 1.25))
    slabs(3.98, .72, 2.55, 1.12, "Iterative\ndenoising", size=16)
    text(5.26, .30, "EMA inference weights", 11.5)
    arrow((6.83, 1.25), (7.63, 1.25))
    trajectory(7.82, .66, w=1.83, h=1.24, mode="generated")
    text(8.74, 2.31, "Next 128 steps", 15)
    text(8.74, .30, "Generated trajectory segment", 11.5)
    connector([(9.62, 1.90), (9.94, 2.07), (9.94, 2.77),
               (5.29, 2.77), (5.29, 2.08)])
    text(7.22, 3.00, "Last 32 steps become the next context", 12,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    arrow((9.87, 1.25), (10.57, 1.25))
    trajectory(10.78, .64, w=3.15, h=1.30, mode="generated", full=True)
    text(12.35, 2.31, "Full synthetic trajectory", 15)
    text(12.35, .30, "Join segments in temporal order", 11.5)

    ax.text(14.58, .015, "AI generated image", fontsize=9, ha="right",
            va="bottom", color="#666666")

    fig.canvas.draw()
    bounds = fig.get_window_extent()
    for label in ax.texts:
        box = label.get_window_extent(fig.canvas.get_renderer())
        if not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)):
            raise ValueError(f"Label outside figure: {label.get_text()}")
    for ext in ("svg", "png"):
        fig.savefig(output / f"00_temporal_diffusion_overview.{ext}",
                    dpi=240, facecolor="white", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/synthetic_data/figures")
    render(parser.parse_args(argv).output)


if __name__ == "__main__":
    main()
