"""Replot the public t-SNE, position and TA figures from bundled plot inputs.

Run inside the repository: python -m synthetic_generation.figures --output outputs/figures
This module reads only the supplied synthetic plot arrays and paper-panel PNGs.
It does not fit t-SNE, load policies, recompute scores or access research runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde


DEFAULT_INPUTS = Path(__file__).resolve().parents[1] / "docs/synthetic_data/inputs"
STEMS = {
    "tsne": ("04_synthetic_tsne",),
    "position": ("05_position_high_gap",),
    "ta": ("06_ta_distributions", "06b_ta_dsnips"),
    "ta-older": ("06c_ta_adv_older", "06d_ta_dsnips_older"),
}


def load_inputs(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        with (folder / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Plot input checksum mismatch: {name}")
    with np.load(folder / "synthetic_plot_inputs.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    settings = json.loads((folder / "render_settings.json").read_text())
    n = int(data["trajectory_offsets"][-1])
    for key in ("position_xy", "tsne_xy"):
        if data[key].shape != (n, 2):
            raise ValueError(f"Expected {n} two-dimensional coordinates: {key}")
    for key in ("position_gap", "tsne_gap", "scenarios"):
        if data[key].shape != (n,):
            raise ValueError(f"Expected one value per transition: {key}")
    if (data["trajectory_offsets"].shape != (25,)
            or data["trajectory_offsets"][0] != 0
            or np.any(np.diff(data["trajectory_offsets"]) <= 0)):
        raise ValueError("Expected 24 nonempty trajectory intervals")
    for key in ("ta_adv", "ta_dsnips"):
        if data[key].shape != (24, 2):
            raise ValueError(f"Expected 24 trajectories by two references: {key}")
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError("Plot arrays must be finite")
    groups = np.asarray(manifest["trajectory_source_groups"])
    if (groups.shape != (24,)
            or not np.isin(groups, ("ordinary", "senior")).all()
            or np.count_nonzero(groups == "senior") != 12):
        raise ValueError("Expected 12 younger-source and 12 older-source trajectories")
    data["ta_older_mask"] = groups == "senior"
    panels = {}
    for name in ("paper_tsne", "paper_position", *(f"paper_ta{i}" for i in range(4))):
        with Image.open(folder / f"{name}.png") as image:
            panels[name] = np.array(image.convert("RGB"))
    return data, settings, panels


def save(fig, output, stem):
    if fig.texts or any(ax.get_title() for ax in fig.axes):
        raise ValueError("Figures must not have titles or footer captions")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=230,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tsne(data, settings, panels, output):
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_axes([.005, .04, .47, .94])
    ax.imshow(panels["paper_tsne"])
    ax.axis("off")
    ax = fig.add_axes([.515, .07, .385, .88])
    cmap = LinearSegmentedColormap.from_list("paper_gap", ["#3A4CC0", "#F2F2F2", "#800020"])
    art = ax.scatter(*data["tsne_xy"].T,
                     c=np.clip(data["tsne_gap"] / settings["color_cap"], 0, 1),
                     s=10, cmap=cmap, norm=PowerNorm(1.8, 0, 1), alpha=.8, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(art, cax=fig.add_axes([.915, .13, .02, .72]), extend="max")
    cb.set_label("Normalized |YDR − ODR|")
    cb.set_ticks(np.linspace(0, 1, 6))
    save(fig, output, STEMS["tsne"][0])


def position(data, settings, panels, output):
    xy, gap = data["position_xy"], data["position_gap"]
    offsets = data["trajectory_offsets"]
    q99, q999 = settings["highlight_thresholds"]
    lines = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        points = xy[a:b]
        lines.append(points[np.unique(np.r_[np.arange(0, len(points), 20), len(points) - 1])])
    fig = plt.figure(figsize=(12.7, 5.2))
    ax = fig.add_axes([.005, .01, .48, .98])
    ax.imshow(panels["paper_position"])
    ax.axis("off")
    ax = fig.add_axes([.565, .13, .42, .82])
    ax.add_collection(LineCollection(lines, colors="#bfc3c7", linewidths=.25,
                                     alpha=.22, rasterized=True))
    for low, high, color in ((q99, q999, "#c95670"), (q999, np.inf, "#751b3e")):
        selected = (gap > low) & (gap <= high)
        delta = gap[selected] - q99
        areas = 9 + 160 * (delta / (delta + q999 - q99)) ** 2
        ax.scatter(*xy[selected].T, s=areas, color=color, alpha=.75,
                   linewidths=0, rasterized=True)
    low, high = xy.min(0) - .08, xy.max(0) + .08
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           xlabel="Position X", ylabel="Position Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.15)
    ax.tick_params(labelsize=9)
    # Paper-style labels point to a real synthetic row nearest each scenario's
    # coordinate median. Label placement is layout only; no real-data region
    # boundaries are copied onto synthetic observations.
    for scenario in settings["scenario_labels"]:
        points = xy[data["scenarios"] == scenario["scenario"]]
        center = np.median(points, axis=0)
        target = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
        ax.annotate(f"Scenario {scenario['scenario']}", xy=target,
                    xytext=scenario["label_xy"], ha=scenario["label_align"], va="center",
                    fontsize=9, fontweight="bold", color="#303030",
                    bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.92),
                    arrowprops=dict(arrowstyle="-", color="#404040", lw=.9), zorder=5)
    save(fig, output, STEMS["position"][0])


def ta(data, settings, panels, output, *, older_only=False):
    boxes = settings["ta_crop_boxes"]
    frames = settings["ta_plot_frames"]
    stems = STEMS["ta-older" if older_only else "ta"]
    mask = data["ta_older_mask"] if older_only else slice(None)
    label = "Older traj" if older_only else "Trajectory"
    for k, (key, stem) in enumerate(zip(("ta_adv", "ta_dsnips"), stems)):
        fig = plt.figure(figsize=(11.8, 7.0))
        for j, reference in enumerate(("ODA", "YDA")):
            idx, box, frame = 2 * k + j, boxes[2 * k + j], frames[2 * k + j]
            image_ax = fig.add_axes([.005, .53 if j == 0 else .02, .48, .46])
            image_ax.imshow(panels[f"paper_ta{idx}"])
            image_ax.axis("off")
            fig.canvas.draw()
            display = image_ax.transData.transform([
                [frame[0] - box[0], frame[3] - box[1]],
                [frame[2] - box[0], frame[1] - box[1]],
            ])
            (x0, y0), (x1, y1) = fig.transFigure.inverted().transform(display)
            ax = fig.add_axes([x0 + .5, y0, x1 - x0, y1 - y0])
            values = data[key][mask, j]
            limits = (-.05, .02) if k == 0 else (-.4, .3)
            grid = np.linspace(*limits, 1000)
            color = ("#a72c51", "#326b9a")[j]
            ax.hist(values, bins=10, density=True, color=color, alpha=.35,
                    edgecolor="white", label=f"{label} → {reference}")
            ax.plot(grid, gaussian_kde(values)(grid), color="red", lw=1.4)
            ax.axvline(0, color=".3", ls="--", lw=.8)
            ax.set(xlim=limits, xlabel="Value", ylabel="TA-Adv" if k == 0 else "TA-dSNIPS")
            ax.legend(frameon=True, fontsize=8, loc="upper left")
            ax.grid(alpha=.15, ls=":")
        save(fig, output, stem)


def ta_older(data, settings, panels, output):
    ta(data, settings, panels, output, older_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--figure", choices=("all", *STEMS), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    names = tuple(STEMS) if args.figure == "all" else (args.figure,)
    expected = [args.output / f"{stem}.{ext}" for name in names
                for stem in STEMS[name] for ext in ("png", "pdf")]
    if any(path.exists() for path in expected):
        parser.error("Figure files already exist; choose a fresh --output directory")
    data, settings, panels = load_inputs(args.inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    renderers = {"tsne": tsne, "position": position, "ta": ta, "ta-older": ta_older}
    for name in names:
        renderers[name](data, settings, panels, args.output)
    print(f"Saved {len(expected)} PNG/PDF files to {args.output}")


if __name__ == "__main__":
    main()
