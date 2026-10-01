"""Standalone shared legend used by the multi-panel figures; writes output/legend_version2_spaced.pdf."""

import os

os.makedirs("output", exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 1. Define Palette
custom_palette = {
    "BPNet": "#C0C0C0",
    "Distilled Nucleotide Transformer": "#2E5C8A",
    "Nucleotide Transformer": "#A8C8E1",
    "Distilled DNABERT-2": "#E67E22",
    "DNABERT-2": "#FFD4A3",
    "Enformer Distilled": "#27AE60",
    "Enformer": "#82E0AA",
    "Distilled Caduceus": "#8E44AD",
    "Caduceus": "#D7BDE2",
}

# 2. Setup Figure
# Increased height (figsize=(3.5, 16)) to accommodate large vertical spacing
fig_leg = plt.figure(figsize=(3.5, 16))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

# 3. Helper Function
def create_teacher_legend_handle(ax_target, color):
    h = ax_target.scatter([], [], s=350, facecolors="white", edgecolors=color, linewidths=3.5, marker="o")
    h.set_linestyle("--")
    return h

# 4. Define Legend Elements (Original Order)
legend_elements = [
    Line2D([0], [0], color="#D3D3D3", lw=4.5, linestyle=":", label="BPNet Baseline"),
    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["BPNet"], markersize=16,
           markeredgecolor="black", markeredgewidth=2, linestyle="-", linewidth=3.5,
           color=custom_palette["BPNet"], label="BPNet"),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Distilled Nucleotide Transformer"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="NT (Student)"),
    create_teacher_legend_handle(ax_leg, custom_palette["Nucleotide Transformer"]),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Distilled DNABERT-2"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="DNABERT-2 (S)"),
    create_teacher_legend_handle(ax_leg, custom_palette["DNABERT-2"]),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Enformer Distilled"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="Enformer (S)"),
    create_teacher_legend_handle(ax_leg, custom_palette["Enformer"]),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Distilled Caduceus"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="Caduceus (S)"),
    create_teacher_legend_handle(ax_leg, custom_palette["Caduceus"]),

    mpatches.Patch(facecolor="#FFEEEE", edgecolor="red", linestyle="--", linewidth=2, alpha=0.5, label="Best Distilled Model"),
]

labels = [
    "BPNet Base", "BPNet",
    "NT (Student)", "NT (Teacher)",
    "DNABERT-2 (S)", "DNABERT-2 (T)",
    "Enformer (S)", "Enformer (T)",
    "Caduceus (S)", "Caduceus (T)",
    "Best Distilled"
]

# 5. Create Legend
fig_leg.legend(
    handles=legend_elements,
    labels=labels,
    loc="center",
    ncol=1,
    fontsize=16,
    frameon=False,
    labelspacing=2.5,   # <--- CHANGE THIS to adjust vertical whitespace
    handletextpad=1.5   # Increases space between the dot/line and the text
)

plt.savefig("output/legend_version2_spaced.pdf", bbox_inches="tight", dpi=300)
plt.show()