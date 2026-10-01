"""18-task mean MCC per model, excluding the best distilled configuration; writes output/model_mean_without_best_distillation_model.pdf."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.makedirs("output", exist_ok=True)
# ===========================================================================
# 1. SETUP: Load and Prepare Actual Data  (LOCAL CSV, LONG FORMAT)
# ===========================================================================
# Expected columns in model_comparison_long.csv:
#   Task, Model, Score, Model Family, Type
# Where "Type" is one of: Baseline / Teacher / Student (or your chosen labels)

df_strip = pd.read_csv("data/model_comparison_long.csv")

# (Optional) sanity checks / light normalization
required_cols = {"Task", "Model", "Score", "Type"}
missing = required_cols - set(df_strip.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# Make sure Score is numeric
df_strip["Score"] = pd.to_numeric(df_strip["Score"], errors="coerce")

# --- Desired y-axis model order and labels (unchanged) ---
y_axis_order = [
    "BPNet",
    "Nucleotide Transformer",
    "Distilled Nucleotide Transformer",
    "DNABERT-2",
    "Distilled DNABERT-2",
    "Enformer",
    "Enformer Distilled",
    "Caduceus",
    "Distilled Caduceus",
]

y_axis_labels = [
    "BPNet",
    "Nucleotide\nTransformer",
    "Distilled Nucleotide\nTransformer",
    "DNABERT-2",
    "Distilled\nDNABERT-2",
    "Enformer",
    "Enformer\nDistilled",
    "Caduceus",
    "Distilled\nCaduceus",
]

# If your CSV uses different "Type" names, optionally remap them here to match your palette below:
type_map = {"Teacher": "Teacher Model", "Student": "Distilled", "Baseline": "Baseline"}
df_strip["Type"] = df_strip["Type"].map(type_map).fillna(df_strip["Type"])
print("Creating plot...")

# Define the desired order for the y-axis to group models
y_axis_order = [
    "BPNet",
    "Nucleotide Transformer",
    "Distilled Nucleotide Transformer",
    "DNABERT-2",
    "Distilled DNABERT-2",
    "Enformer",
    "Enformer Distilled",
    "Caduceus",
    "Distilled Caduceus",
]

# --- MODIFICATION: Create new labels with line breaks ---
y_axis_labels = [
    "BPNet",
    "Nucleotide\nTransformer",
    "Distilled Nucleotide\nTransformer",
    "DNABERT-2",
    "Distilled\nDNABERT-2",
    "Enformer",
    "Enformer\nDistilled",
    "Caduceus",
    "Distilled\nCaduceus",
]

# Set plot style and figure size
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(18, 12))  # Increased figure width for legend space #(16, 12)

# --- Plotting ---
# Show each observation with a scatterplot
sns.stripplot(
    data=df_strip,
    y="Model",
    x="Score",
    hue="Type",
    dodge=True,
    alpha=0.25,
    zorder=1,
    legend=False,
    size=12,
    order=y_axis_order,
    ax=ax,
)

# Show the conditional means as diamond markers
sns.pointplot(
    data=df_strip,
    y="Model",
    x="Score",
    hue="Type",
    dodge=0.4,  # Match the dodge amount for text placement
    palette={"Teacher Model": "g", "Distilled": "r", "Baseline": "b"},  # Explicitly set colors
    errorbar=None,
    markers="d",
    markersize=12,
    linestyle="none",
    order=y_axis_order,
    ax=ax,
)

# --- Annotations and Baseline ---
# Calculate the mean score for BPNet to use as a baseline
bpnet_mean_score = df_strip[df_strip["Model"] == "BPNet"]["Score"].mean()

# Add a dotted vertical line for the BPNet baseline
ax.axvline(
    bpnet_mean_score,
    color="gray",
    linestyle=":",
    linewidth=2,
    zorder=0,
    label=f"BPNet Mean ({bpnet_mean_score:.3f})",
)


# --- Add Text Annotations for Mean Scores ---
mean_scores = df_strip.groupby(["Model", "Type"])["Score"].mean().reset_index()
model_positions = {model: i for i, model in enumerate(y_axis_order)}
hue_order = sorted(df_strip["Type"].unique())

if len(hue_order) > 1:
    dodge_val = 0.4
    hue_offsets = np.linspace(-dodge_val / 2, dodge_val / 2, len(hue_order))
    type_positions = {type_name: offset for type_name, offset in zip(hue_order, hue_offsets)}
else:
    type_positions = {hue_order[0]: 0}


for _, row in mean_scores.iterrows():
    model_name = row["Model"]
    type_name = row["Type"]
    mean_val = row["Score"]

    if model_name in model_positions:
        y_pos = model_positions[model_name]
        if type_name in type_positions:
            y_pos += type_positions[type_name]
        ax.text(
            mean_val + 0.01,
            y_pos,
            f"{mean_val:.3f}",
            color="black",
            ha="left",
            va="center",
            fontweight="light",
            fontsize=16,
        )


# --- Final Touches ---
ax.set_xlabel("Score", fontsize=24, labelpad=15)
ax.set_ylabel("Model", fontsize=24, labelpad=15)

# --- MODIFICATION: Set the new y-tick labels and increase their font size ---
ax.set_yticklabels(y_axis_labels, fontsize=20)  # Increased font size
plt.xticks(fontsize=20)  # You can change 14 to any size you like

# Recreate the legend to include the baseline and have better control
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
# --- MODIFICATION: Move legend outside the plot ---
ax.legend(
    unique_labels.values(),
    unique_labels.keys(),
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    ncol=1,
    frameon=True,
    fontsize=18,
    title_fontsize=20,
    columnspacing=1,
    handletextpad=0.5,
    title="Type",
)


plt.tight_layout(pad=1.5)
plt.savefig("output/model_mean_without_best_distillation_model.pdf", bbox_inches="tight")
plt.show()
