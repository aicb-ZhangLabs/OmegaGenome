"""Panel A of Figure 2: the 3x3 lollipop grid of per-task MCC; writes output/lollipop_3rows_no_legend.{pdf,png}."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# Set seaborn theme
sns.set_theme(style="white", palette="deep", font="DejaVu Sans")
sns.set_context("poster")

# Configuration
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

def load_preprocessed_long(csv_path=f"{DATA_DIR}/model_comparison_long_final.csv"):
    """Load the long-format dataframe."""
    df = pd.read_csv(csv_path)
    return df

# Load data
df_long = load_preprocessed_long()

# Define the 3 rows
epigenetic_tasks = ["H3K27ac", "H3K4me3", "H3K27me3"]
enhancer_promoter_tasks = ["enhancers_types", "promoter_tata", "promoter_no_tata"]
splice_tasks = ["splice_sites_donors", "splice_sites_all", "splice_sites_acceptors"]

task_rows = [epigenetic_tasks, enhancer_promoter_tasks, splice_tasks]

row_titles = ["Histone Modifications MCC", "CREs MCC", "Splicing Sites MCC"]

# Palette
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

# ==========================================
# PART 1: MAIN PLOT (NO LEGEND)
# ==========================================

fig = plt.figure(figsize=(15, 18))

gs = fig.add_gridspec(3, 4, hspace=0.1, wspace=0.15,
                      width_ratios=[0.15, 1, 1, 1])

family_order = [
    "BPNet",
    "Nucleotide Transformer",
    "DNABERT-2",
    "Enformer",
    "Caduceus",
]

# CHANGE: Larger marker size (s=650)
MARKER_SIZE = 650

def find_best_distilled_family(task_data):
    best_score = -1
    best_family = None
    for family in family_order[1:]:
        family_data = task_data[task_data["Model Family"] == family]
        student_data = family_data[family_data["Type"] == "Student"]
        if not student_data.empty:
            score = student_data["Score"].iloc[0]
            if score > best_score:
                best_score = score
                best_family = family
    return best_family, best_score

for row_idx, task_row in enumerate(task_rows):
    # Row label
    ax_label = fig.add_subplot(gs[row_idx, 0])

    # Text closer to plot
    ax_label.text(0.9, 0.5, row_titles[row_idx], fontsize=20, fontweight="normal",
                  rotation=90, va="center", ha="right", transform=ax_label.transAxes)
    ax_label.axis("off")

    for col_idx, task in enumerate(task_row):
        ax = fig.add_subplot(gs[row_idx, col_idx + 1])
        task_data = df_long[df_long["Task"] == task].copy()
        best_distilled_family, best_distilled_score = find_best_distilled_family(task_data)

        spacing_factor = 0.07
        group_centers = np.arange(len(family_order)) * spacing_factor

        max_score_in_plot = 0

        for i, family in enumerate(family_order):
            group_center_x = group_centers[i]
            family_data = task_data[task_data["Model Family"] == family]

            # Highlight best distilled
            if family == best_distilled_family:
                highlight_rect = mpatches.FancyBboxPatch(
                    (group_center_x - (spacing_factor * 0.25), 0),
                    spacing_factor * 0.5, 1.0,
                    boxstyle="round,pad=0.02,rounding_size=0.02",
                    facecolor="#FFEEEE",
                    alpha=0.6,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=2.0,
                    zorder=0,
                    transform=ax.get_xaxis_transform(),
                )
                ax.add_patch(highlight_rect)

            if family == "BPNet":
                if not family_data.empty:
                    score = family_data["Score"].iloc[0]
                    model_name = family_data["Model"].iloc[0]

                    ax.plot(
                        [group_center_x, group_center_x],
                        [0, score],
                        color=custom_palette[model_name],
                        linewidth=3.5,
                        zorder=1
                    )
                    ax.scatter(
                        group_center_x,
                        score,
                        s=MARKER_SIZE,
                        color=custom_palette[model_name],
                        edgecolor="black",
                        linewidth=2,
                        zorder=2,
                        marker="o"
                    )
                    ax.annotate(
                        f"{score:.2f}",
                        (group_center_x, score),
                        textcoords="offset points",
                        xytext=(0, 15),
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        fontweight="normal",
                        rotation=90,
                        color="black"
                    )
            else:
                student_data = family_data[family_data["Type"] == "Student"]
                teacher_data = family_data[family_data["Type"] == "Teacher"]

                if not student_data.empty and not teacher_data.empty:
                    student_score = student_data["Score"].iloc[0]
                    student_model = student_data["Model"].iloc[0]
                    teacher_score = teacher_data["Score"].iloc[0]
                    teacher_model = teacher_data["Model"].iloc[0]

                    max_score_pair = max(student_score, teacher_score)

                    # Stick
                    ax.plot(
                        [group_center_x, group_center_x],
                        [0, max_score_pair],
                        color="#CCCCCC",
                        linewidth=3.5,
                        zorder=1,
                        alpha=0.6
                    )

                    # Connection line
                    ax.plot(
                        [group_center_x, group_center_x],
                        [student_score, teacher_score],
                        color=custom_palette[teacher_model],
                        linewidth=2.5,
                        linestyle=":",
                        zorder=2,
                        alpha=0.7
                    )

                    # Teacher Marker
                    sc_teacher = ax.scatter(
                        group_center_x,
                        teacher_score,
                        s=MARKER_SIZE,
                        facecolors="white",
                        edgecolors=custom_palette[teacher_model],
                        linewidth=3.5,
                        zorder=3,
                        marker="o"
                    )
                    # CHANGE: Dashed boundary for teacher in plot
                    sc_teacher.set_linestyle("--")

                    # Student Marker
                    ax.scatter(
                        group_center_x,
                        student_score,
                        s=MARKER_SIZE,
                        color=custom_palette[student_model],
                        edgecolor="black",
                        linewidth=2,
                        zorder=3,
                        marker="o"
                    )

                    top_val = max(student_score, teacher_score)
                    bot_val = min(student_score, teacher_score)

                    # Labels
                    ax.annotate(
                        f"{top_val:.2f}",
                        (group_center_x, top_val),
                        textcoords="offset points",
                        xytext=(0, 15),
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        fontweight="normal",
                        color="black",
                        rotation=90
                    )

                    ax.annotate(
                        f"{bot_val:.2f}",
                        (group_center_x, bot_val),
                        textcoords="offset points",
                        xytext=(0, -21),
                        ha="center",
                        va="top",
                        fontsize=14,
                        fontweight="normal",
                        color="black",
                        rotation=90
                    )

        # Baseline
        bpnet_rows = task_data[task_data["Model"] == "BPNet"]["Score"].values
        if len(bpnet_rows) > 0:
            # CHANGE: Thicker baseline
            ax.axhline(y=bpnet_rows[0], color="#D3D3D3", linestyle=":", linewidth=4.5, alpha=0.8, zorder=0)

        # Limits & Titles
        scores = task_data["Score"].dropna().values
        if len(scores) > 0:
            min_val, max_val = scores.min(), scores.max()
            padding = (max_val - min_val) * 0.3
            ax.set_ylim(max(0, min_val - padding), max_val + padding)
            ax.set_title(task.replace("_", " ").title(), fontsize=20, fontweight="normal", pad=12)

        ax.set_ylabel("")

        # Y Axis Formatting
        ax.tick_params(axis="y", labelsize=14, pad=5, left=True, length=6, width=1.5, labelrotation=90)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # X-Axis Tight fit
        ax.set_xlabel("")
        ax.set_xticks(group_centers)
        ax.set_xticklabels([])
        margin = spacing_factor * 0.8
        ax.set_xlim(group_centers[0] - margin, group_centers[-1] + margin)

        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5, axis="y")
        ax.set_axisbelow(True)

# Save the main plot without legend
plt.tight_layout()
plt.savefig("output/lollipop_3rows_no_legend.pdf", bbox_inches="tight", dpi=300)
plt.savefig("output/lollipop_3rows_no_legend.png", bbox_inches="tight", dpi=300)
print("Main plot (no legend) saved.")


# ==========================================
# PART 2: SEPARATE LEGEND PLOT (VERTICAL)
# ==========================================

# Create a figure specifically for the legend
fig_leg = plt.figure(figsize=(4, 12)) # Vertical shape
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

def create_teacher_legend_handle(ax_target, color):
    """
    Creates a scatter object to serve as a legend handle.
    """
    # Use the passed axis to create the dummy collection
    h = ax_target.scatter([], [], s=350, facecolors="white", edgecolors=color, linewidths=3.5, marker="o")
    h.set_linestyle("--")
    return h

legend_elements = [
    Line2D([0], [0], color="#D3D3D3", lw=4.5, linestyle=":", label="BPNet Baseline"),
    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["BPNet"], markersize=16,
           markeredgecolor="black", markeredgewidth=2, linestyle="-", linewidth=3.5,
           color=custom_palette["BPNet"], label="BPNet"),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Distilled Nucleotide Transformer"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="NT (Student)"),

    # Teacher handle (passed ax_leg)
    create_teacher_legend_handle(ax_leg, custom_palette["Nucleotide Transformer"]),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Distilled DNABERT-2"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="DNABERT-2 (Student)"),

    # Teacher handle
    create_teacher_legend_handle(ax_leg, custom_palette["DNABERT-2"]),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Enformer Distilled"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="Enformer (Student)"),

    # Teacher handle
    create_teacher_legend_handle(ax_leg, custom_palette["Enformer"]),

    Line2D([0], [0], marker="o", markerfacecolor=custom_palette["Distilled Caduceus"],
           markersize=16, markeredgecolor="black", markeredgewidth=2, linestyle="-",
           linewidth=3.5, color="#CCCCCC", label="Caduceus (Student)"),

    # Teacher handle
    create_teacher_legend_handle(ax_leg, custom_palette["Caduceus"]),

    mpatches.Patch(facecolor="#FFEEEE", edgecolor="red", linestyle="--", linewidth=2, alpha=0.5, label="Best Distilled Model"),
]

labels = [
    "BPNet Baseline", "BPNet",
    "NT (Student)", "NT (Teacher)",
    "DNABERT-2 (Student)", "DNABERT-2 (Teacher)",
    "Enformer (Student)", "Enformer (Teacher)",
    "Caduceus (Student)", "Caduceus (Teacher)",
    "Best Distilled Model"
]

# Create the vertical legend (ncol=1)
fig_leg.legend(handles=legend_elements, labels=labels, loc="center", ncol=1, fontsize=18, frameon=False)

plt.savefig("output/legend_vertical.pdf", bbox_inches="tight", dpi=300)
plt.savefig("output/legend_vertical.png", bbox_inches="tight", dpi=300)

print("Separate vertical legend saved.")