"""Accuracy-versus-cost scatter over the five teachers and their students, including the best distilled configuration (reads data/model_comparison_long.csv)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import os

os.makedirs("output", exist_ok=True)

sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")

# ===========================================================================
# CONFIGURATION PARAMETERS
# ===========================================================================
CSV_PATH = Path("data/model_comparison_long.csv")  # <— your local CSV
SHOW_TOP_EFFICIENCY_CIRCLES = False
SHOW_EFFICIENCY_FRONTIER = True
ANNOTATE_EFFICIENCY = True
ANNOTATE_SIZE = True
ANNOTATE_ACCURACY = True

# ===========================================================================
# 1) LOAD LONG-FORM CSV AND PIVOT TO WIDE
#    Expected columns (case-insensitive ok): Task, Model, Score, Model Family, Type
#    Example rows:
#      H3K9ac, BPNet, 0.52134, BPNet, Baseline
#      H3K9ac, Distilled Nucleotide Transformer, 0.51054, Nucleotide Transformer, Student
# ===========================================================================
df_long = pd.read_csv(CSV_PATH)

# Normalize column names
df_long.columns = [c.strip() for c in df_long.columns]
colmap = {c.lower(): c for c in df_long.columns}
# Accept either "Task" or "task_name"
if "task" not in colmap and "task_name" in colmap:
    df_long = df_long.rename(columns={colmap["task_name"]: "Task"})
elif "task" in colmap:
    df_long = df_long.rename(columns={colmap["task"]: "Task"})

# Ensure required columns exist
required = {"Task", "Model", "Score"}
missing = [c for c in required if c not in df_long.columns]
if missing:
    raise ValueError(
        f"CSV is missing required columns: {missing}. Found columns: {list(df_long.columns)}"
    )

# Pivot to get a task x model wide table of scores (mean if duplicates)
df_main = df_long.pivot_table(index="Task", columns="Model", values="Score", aggfunc="mean")

# ===========================================================================
# 2) CALCULATE MEAN PERFORMANCE AND MODEL SIZES
# ===========================================================================
mean_performance = df_main.mean().to_dict()

# Known sizes (M params). You can extend this dict if you add models later.
model_sizes = {
    "BPNet": 0.12,
    "Caduceus": 1.90,
    "DNABERT-2": 117,
    "Enformer": 250,
    "Nucleotide Transformer": 2550,
    "Distilled Nucleotide Transformer": 0.12,
    "Distilled DNABERT-2": 0.12,
    "Enformer Distilled": 0.12,
    "Distilled Caduceus": 0.12,
}


def infer_size(model_name: str) -> float:
    # default: distilled ~0.12M, else look up known sizes (NaN if unknown)
    if "distilled" in model_name.lower() or model_name.lower().startswith("best"):
        return 0.12
    return model_sizes.get(model_name, np.nan)


def infer_type(model_name: str) -> str:
    if "distilled" in model_name.lower() or "student" in model_name.lower():
        return "Distilled"
    if model_name == "BPNet":
        return "Baseline"
    return "Teacher Model"


plot_data = []
for name, mcc in mean_performance.items():
    size = infer_size(name)
    mtype = infer_type(name)
    plot_data.append({"Model": name, "Size (M params)": size, "Mean MCC": mcc, "Type": mtype})

df_plot = pd.DataFrame(plot_data)

# Drop models with unknown sizes to avoid plotting issues
df_plot = df_plot.dropna(subset=["Size (M params)"])

# ===========================================================================
# 3) ADD "BEST DISTILLATION MODEL" (row-wise max across distilled columns)
# ===========================================================================
# Identify distilled columns by name; if you prefer using the CSV "Type" column,
# you could collect models where Type == "Student" and intersect with df_main.columns.
distilled_cols = [c for c in df_main.columns if "distilled" in c.lower()]
if len(distilled_cols) > 0:
    df_main["Best Distilled"] = df_main[distilled_cols].max(axis=1)
    best_distilled_mean_mcc = df_main["Best Distilled"].mean()
    df_plot = pd.concat(
        [
            df_plot,
            pd.DataFrame(
                [
                    {
                        "Model": "Best Distillation Model",
                        "Size (M params)": 0.12,
                        "Mean MCC": best_distilled_mean_mcc,
                        "Type": "Distilled",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

# ===========================================================================
# 4) PLOT: COMPUTATIONAL EFFICIENCY
# ===========================================================================
fig, ax = plt.subplots(figsize=(16, 10))

colors = {"Teacher Model": "#2E86AB", "Distilled": "#A23B72", "Baseline": "#F18F01"}
markers = {"Teacher Model": "o", "Distilled": "D", "Baseline": "s"}

df_plot["Efficiency"] = df_plot["Mean MCC"] / np.log10(df_plot["Size (M params)"] + 1)

for model_type in ["Teacher Model", "Distilled", "Baseline"]:
    data = df_plot[df_plot["Type"] == model_type]
    bubble_sizes = 10 + np.log10(data["Size (M params)"] + 1) * 6000
    ax.scatter(
        data["Size (M params)"],
        data["Mean MCC"],
        s=bubble_sizes,
        c=colors[model_type],
        alpha=0.7,
        edgecolors="white",
        linewidth=2,
        marker=markers[model_type],
        label=model_type,
        zorder=3,
    )

distilled_models = df_plot[df_plot["Type"] == "Distilled"]
other_models = df_plot[df_plot["Type"] != "Distilled"]

# Manual annotation positions (optional fine-tuning)
manual_positions_other = {
    "BPNet": (1, 0.59),
    "Caduceus": (10, 0.62),
    "DNABERT-2": (90, 0.63),
    "Enformer": (400, 0.63),
    "Nucleotide Transformer": (2200, 0.63),
}

for _, row in other_models.iterrows():
    label = "NT-2.5B" if row["Model"] == "Nucleotide Transformer" else row["Model"]
    text_x, text_y = manual_positions_other.get(
        row["Model"],
        (row["Size (M params)"], row["Mean MCC"] + 0.01),
    )
    annotation_text = []
    if ANNOTATE_EFFICIENCY:
        annotation_text.append(f"E: {row['Efficiency']:.2f}")
    if ANNOTATE_SIZE:
        annotation_text.append(f"S: {row['Size (M params)']:.2f}M")
    if ANNOTATE_ACCURACY:
        annotation_text.append(f"A: {row['Mean MCC']:.3f}")
    full_label = f"{label}\n" + "\n".join(annotation_text) if annotation_text else label

    ax.annotate(
        full_label,
        xy=(row["Size (M params)"], row["Mean MCC"]),
        xytext=(text_x, text_y),
        fontsize=20,
        ha="center",
        va="center",
        fontweight="medium",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.9,
            edgecolor=colors[row["Type"]],
            linewidth=1,
        ),
        arrowprops=dict(
            arrowstyle="-", connectionstyle="arc3,rad=0.3", color="gray", alpha=0.5, lw=1
        ),
        zorder=4,
    )

distilled_annotation_positions = {
    "Distilled DNABERT-2": (0.4, 0.655),
    "Enformer Distilled": (1.7, 0.655),
    "Distilled Nucleotide Transformer": (0.4, 0.63),
    "Distilled Caduceus": (0.4, 0.61),
    "Best Distillation Model": (0.08, 0.655),
}

distilled_sorted = distilled_models.sort_values("Mean MCC", ascending=False)
for _, row in distilled_sorted.iterrows():
    label_map = {
        "Distilled Nucleotide Transformer": "Distilled NT",
        "Distilled DNABERT-2": "Distilled\nDNABERT-2",
        "Enformer Distilled": "Distilled\nEnformer",
        "Distilled Caduceus": "Distilled\nCaduceus",
        "Best Distillation Model": "Best\nDistillation\nModel",
    }
    label = label_map.get(row["Model"], row["Model"])
    text_x, text_y = distilled_annotation_positions.get(row["Model"], (0.08, 0.64))

    annotation_text = []
    if ANNOTATE_EFFICIENCY:
        annotation_text.append(f"E: {row['Efficiency']:.2f}")
    if ANNOTATE_SIZE:
        annotation_text.append(f"S: {row['Size (M params)']:.2f}M")
    if ANNOTATE_ACCURACY:
        annotation_text.append(f"A: {row['Mean MCC']:.3f}")
    full_label = f"{label}\n" + "\n".join(annotation_text) if annotation_text else label

    font_weight = "bold" if row["Model"] == "Best Distillation Model" else "medium"
    ax.annotate(
        full_label,
        xy=(row["Size (M params)"], row["Mean MCC"]),
        xytext=(text_x, text_y),
        fontsize=20,
        ha="left",
        va="center",
        fontweight=font_weight,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.9,
            edgecolor=colors[row["Type"]],
            linewidth=1,
        ),
        arrowprops=dict(
            arrowstyle="-", connectionstyle="arc3,rad=0.3", color="gray", alpha=0.5, lw=1
        ),
        zorder=4,
    )

# Optional: Efficiency/Pareto frontier
if SHOW_EFFICIENCY_FRONTIER:

    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient

    costs = np.column_stack([df_plot["Size (M params)"].values, -df_plot["Mean MCC"].values])
    pareto_mask = is_pareto_efficient(costs)
    pareto_points = df_plot[pareto_mask].sort_values("Size (M params)")
    if "Nucleotide Transformer" not in pareto_points["Model"].values:
        nt_row = df_plot[df_plot["Model"] == "Nucleotide Transformer"]
        pareto_points = pd.concat([pareto_points, nt_row]).sort_values("Size (M params)")
    ax.plot(
        pareto_points["Size (M params)"],
        pareto_points["Mean MCC"],
        "k--",
        alpha=0.3,
        linewidth=2,
        zorder=1,
        label="Efficiency Frontier",
    )

# Styling
ax.set_xscale("log")
ax.set_xlabel("Model Size (Million Parameters)", fontsize=16, fontweight="medium")
ax.set_ylabel("Mean MCC Score", fontsize=22, fontweight="medium")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.set_xlim(0.05, 5000)
ax.set_xticks([0.1, 1, 10, 100, 1000])
ax.set_xticklabels(["0.1M", "1M", "10M", "100M", "1B"])
# You can widen this if your new CSV ranges differ:
ax.set_ylim(0.58, 0.68)

# Legend
custom_handles = [
    plt.Line2D(
        [0],
        [0],
        marker=markers[label],
        color="w",
        markerfacecolor=colors[label],
        markersize=12,
        label=label,
    )
    for label in ["Teacher Model", "Distilled", "Baseline"]
]
legend = ax.legend(
    handles=custom_handles,
    loc="lower right",
    frameon=True,
    fancybox=True,
    shadow=True,
    ncol=1,
    fontsize=20,
)
legend.get_frame().set_alpha(0.9)

# Bubble-size hint
ax.text(
    0.02,
    0.98,
    "Bubble size ∝ log(Model Size)",
    transform=ax.transAxes,
    fontsize=18,
    ha="left",
    va="top",
    style="italic",
    alpha=0.7,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.savefig(
    "output/computational_efficiency_frontier_with_best_distillation_model.pdf", bbox_inches="tight"
)
plt.show()
