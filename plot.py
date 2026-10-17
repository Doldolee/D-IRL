"""Render the four frozen public comparison figures without private inputs."""
from .utils import new_output
from .synthetic_generation.figures import load_inputs, tsne, position, ta
from .data import ROOT


def render_figures(output_dir, panel="all"):
    import matplotlib.pyplot as plt
    output = new_output(output_dir)
    inputs = ROOT / "docs/synthetic_data/inputs"
    data, settings, panels = load_inputs(inputs)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False})
    functions = {"tsne":tsne, "position":position, "ta":ta}
    for name in functions if panel == "all" else (panel,):
        functions[name](data, settings, panels, output)
