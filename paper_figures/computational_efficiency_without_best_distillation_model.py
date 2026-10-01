"""Accuracy-versus-cost scatter over the five teachers and their students, excluding the best distilled configuration (reads data/model_comparison_long.csv)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.makedirs("output", exist_ok=True)

sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")

# ===========================================================================
# CONFIGURATION PARAMETERS
# ===========================================================================
SHOW_TOP_EFFICIENCY_CIRCLES = False  # Set to False to hide golden circles
SHOW_EFFICIENCY_FRONTIER = True  # Set to False to hide Pareto frontier
ANNOTATE_EFFICIENCY = True  # Show efficiency annotations
ANNOTATE_SIZE = True  # Show size annotations
ANNOTATE_ACCURACY = True  # Show accuracy annotations

# ===========================================================================
# 1) LOAD: Read long-format CSV from local and prepare tables
#    Expected columns (case/space-insensitive): Task, Model, Score, Model Family, Type
# ===========================================================================
csv_path = "data/model_comparison_long.csv"  # <- ensure this file is in your working directory

df_long = pd.read_csv(csv_path)

# Normalize/clean column names just in case
df_long.columns = [c.strip() for c in df_long.columns]
rename_map = {
    "task": "Task",
    "TASK": "Task",
    "model": "Model",
    "MODEL": "Model",
    "score": "Score",
    "SCORE": "Score",
    "model family": "Model Family",
    "ModelFamily": "Model Family",
    "type": "Type",
    "TYPE": "Type",
}
df_long = df_long.rename(columns={k: v for k, v in rename_map.items() if k in df_long.columns})

required_cols = {"Task", "Model", "Score"}
missing = required_cols - set(df_long.columns)
if missing:
    raise ValueError(
        f"CSV missing required columns: {missing}. Found columns: {df_long.columns.tolist()}"
    )

# Ensure numeric Score
df_long["Score"] = pd.to_numeric(df_long["Score"], errors="coerce")
df_long = df_long.dropna(subset=["Score"])

# Optional: keep only the models we care to size/plot later (you can add more sizes below)
# Pivot to a wide table (tasks x models) so we can reuse the original plotting pipeline expectations
df_main = df_long.pivot_table(index="Task", columns="Model", values="Score", aggfunc="mean")

# ===========================================================================
# 2) Calculate Mean Performance and Model Sizes
# ===========================================================================
mean_performance = df_main.mean().to_dict()

# Define (approx.) sizes in millions of parameters for the models that appear in your CSV
# Extend this dict if you add more models
model_sizes = {
    "BPNet": 0.12,
    "Caduceus": 1.90,
    "DNABERT-2": 117,
    "Enformer": 250,
    "Nucleotide Transformer": 2550,
    "Enformer Distilled": 0.12,
    "Distilled Nucleotide Transformer": 0.12,
    "Distilled DNABERT-2": 0.12,
    "Distilled Caduceus": 0.12,
}

# Keep only models that have both mean performance and a known size
available_models = [m for m in mean_performance.keys() if m in model_sizes]


# If you want to infer Type from names (fallback when CSV Type isn't used downstream):
def infer_type(name: str) -> str:
    if "Distilled" in name:
        return "Distilled"
    if name == "BPNet":
        return "Baseline"
    return "Teacher Model"


plot_data = [
    {
        "Model": name,
        "Size (M params)": model_sizes[name],
        "Mean MCC": mean_performance[name],
        "Type": infer_type(name),
    }
    for name in available_models
]
df_plot = pd.DataFrame(plot_data)

# ===========================================================================
# 3) Create Computational Efficiency Plot
# ===========================================================================
fig, ax = plt.subplots(figsize=(16, 10))

colors = {"Teacher Model": "#2E86AB", "Distilled": "#A23B72", "Baseline": "#F18F01"}
markers = {"Teacher Model": "o", "Distilled": "D", "Baseline": "s"}

df_plot["Efficiency"] = df_plot["Mean MCC"] / np.log10(df_plot["Size (M params)"] + 1)

for model_type in ["Teacher Model", "Distilled", "Baseline"]:
    data = df_plot[df_plot["Type"] == model_type]
    if data.empty:
        continue
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

# ===========================================================================
# Manual (x, y) positions for Teacher and Baseline model annotations
# Update these if your mean scores/axis limits change
# ===========================================================================
manual_positions_other = {
    "BPNet": (1, 0.59),  # Baseline
    "Caduceus": (10, 0.62),
    "DNABERT-2": (90, 0.63),
    "Enformer": (400, 0.63),
    "Nucleotide Transformer": (2200, 0.63),
}

for _, row in other_models.iterrows():
    label = "NT-2.5B" if row["Model"] == "Nucleotide Transformer" else row["Model"]
    text_x, text_y = manual_positions_other.get(
        row["Model"], (row["Size (M params)"], row["Mean MCC"] + 0.01)
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

# Manual positions for Distilled models
distilled_annotation_positions = {
    "Enformer Distilled": (0.45, 0.655),
    "Distilled Nucleotide Transformer": (1.7, 0.65),
    "Distilled DNABERT-2": (0.08, 0.655),
    "Distilled Caduceus": (0.4, 0.61),
}

distilled_sorted = distilled_models.sort_values("Mean MCC", ascending=False)
for _, row in distilled_sorted.iterrows():
    label_map = {
        "Distilled Nucleotide Transformer": "Distilled\nNT",
        "Distilled DNABERT-2": "Distilled\nDNABERT-2",
        "Enformer Distilled": "Distilled\nEnformer",
        "Distilled Caduceus": "Distilled\nCaduceus",
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

    ax.annotate(
        full_label,
        xy=(row["Size (M params)"], row["Mean MCC"]),
        xytext=(text_x, text_y),
        fontsize=20,
        ha="left",
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

# ===========================================================================
# Efficiency frontier (Pareto frontier)
# ===========================================================================
if SHOW_EFFICIENCY_FRONTIER and not df_plot.empty:

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
    if (
        "Nucleotide Transformer" not in pareto_points["Model"].values
        and "Nucleotide Transformer" in df_plot["Model"].values
    ):
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

# ===========================================================================
# Styling
# ===========================================================================
ax.set_xscale("log")
ax.set_xlabel("Model Size (Million Parameters)", fontsize=16, fontweight="medium")
ax.set_ylabel("Mean MCC Score", fontsize=22, fontweight="medium")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.set_xlim(0.05, 5000)
ax.set_xticks([0.1, 1, 10, 100, 1000])
ax.set_xticklabels(["0.1M", "1M", "10M", "100M", "1B"])
ax.set_ylim(0.58, 0.68)  # adjust if your means fall outside this range

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
    if label in df_plot["Type"].unique()
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
    "output/computational_efficiency_frontier_without_best_distillation_model.pdf",
    bbox_inches="tight",
)
plt.show()
