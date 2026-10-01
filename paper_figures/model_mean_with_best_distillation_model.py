"""18-task mean MCC per model, including the best distilled configuration; writes output/model_mean_with_best_distillation_model.pdf."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.makedirs("output", exist_ok=True)

# ===========================================================================
# 1) LOAD LONG-FORM CSV AND NORMALIZE
#    Expected columns: Task, Model, Score, Model Family, Type
#    "Type" can be: Baseline / Teacher / Student (→ we normalize to 3 buckets)
# ===========================================================================
csv_path = "data/model_comparison_long.csv"  # change if needed
df_long = pd.read_csv(csv_path)

required = {"Task", "Model", "Score", "Type"}
missing = required - set(df_long.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# Ensure Score numeric
df_long["Score"] = pd.to_numeric(df_long["Score"], errors="coerce")

# Normalize "Type" values
type_map = {
    "Teacher": "Teacher Model",
    "Student": "Distilled",
    "Baseline": "Baseline",
    # pass-through for already-normalized values:
    "Teacher Model": "Teacher Model",
    "Distilled": "Distilled",
}
df_long["Type"] = df_long["Type"].map(type_map).fillna(df_long["Type"])

# ===========================================================================
# 2) ADD "Best Distillation Model"
#    For each Task, take the max Score among rows with Type == "Distilled"
# ===========================================================================
best_distilled = (
    df_long[df_long["Type"] == "Distilled"].groupby("Task", as_index=False)["Score"].max()
)
best_distilled["Model"] = "Best Distillation Model"
best_distilled["Model Family"] = "Best Distillation Model"
best_distilled["Type"] = "Distilled"

# Combine with original long data for plotting
df_plot = pd.concat(
    [
        df_long[["Task", "Model", "Model Family", "Score", "Type"]],
        best_distilled[["Task", "Model", "Model Family", "Score", "Type"]],
    ],
    ignore_index=True,
)

# ===========================================================================
# 3) AXIS ORDER & LABELS (filter to what actually exists in your CSV)
# ===========================================================================
desired_order = [
    "BPNet",
    "Nucleotide Transformer",
    "Distilled Nucleotide Transformer",
    "DNABERT-2",
    "Distilled DNABERT-2",
    "Enformer",
    "Enformer Distilled",
    "Caduceus",
    "Distilled Caduceus",
    "Best Distillation Model",
]
desired_labels = [
    "BPNet",
    "Nucleotide\nTransformer",
    "Distilled Nucleotide\nTransformer",
    "DNABERT-2",
    "Distilled\nDNABERT-2",
    "Enformer",
    "Enformer\nDistilled",
    "Caduceus",
    "Distilled\nCaduceus",
    "Best Distillation\nModel",
]

present_models = [m for m in desired_order if m in df_plot["Model Family"].unique()]
present_labels = [
    lab for m, lab in zip(desired_order, desired_labels) if m in df_plot["Model"].unique()
]

if not present_models:
    raise ValueError("No recognized models found in CSV for plotting. Check 'Model' names.")

# ===========================================================================
# 4) PLOT
# ===========================================================================
print("Creating plot with 'Best Distillation Model' row from CSV...")

sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")
# fig, ax = plt.subplots(figsize=(18, 14))

# Update the Type mapping to include "Distillation Model"
df_plot["Type"] = df_plot["Type"].replace("Distilled", "Distillation Model")


# Scatter points
grid = sns.catplot(
    data=df_plot,
    x="Score",
    y="Model Family",
    hue="Type",
    hue_order=["Baseline", "Teacher Model", "Distillation Model"],
    kind="strip",
    dodge=True,
    jitter=False,
    alpha=0.6,
    size=8,
    height=5,
    aspect=2,
    order=present_models,
)

# Mean diamonds
sns.pointplot(
    data=df_plot,
    y="Model Family",
    x="Score",
    hue="Type",
    hue_order=["Baseline", "Teacher Model", "Distillation Model"],
    dodge=0.5,
    errorbar=None,
    markers="|",
    markersize=18,
    markeredgewidth=4,
    linestyle="none",
    legend=False,
    order=present_models,
)

plt.savefig("output/model_mean_with_best_distillation_model.pdf", bbox_inches="tight")
# plt.show()
