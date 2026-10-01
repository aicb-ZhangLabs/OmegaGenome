# Save provided content to CSV, load from local, reproduce processing, and generate per-task plots (matplotlib-only).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

csv_path = "data/distillation_method_comparison_enformer.tsv"
# with open(csv_path, "w", encoding="utf-8") as f:
#     f.write(distillation_summary_enhanced_content.strip() + ("\n" if not distillation_summary_enhanced_content.endswith("\n") else ""))

# --- Load the enhanced CSV data (from local file) ---
df_enhanced = pd.read_csv(csv_path, sep="\t")


os.makedirs("output", exist_ok=True)

# Set theme with specified parameters
sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")

# # Assuming df_enhanced is already loaded from the enhanced CSV
# # Load the enhanced CSV data (distillation_summary_enhanced_content)
# df_enhanced = pd.read_csv(io.StringIO(distillation_summary_enhanced_content), sep='\t')
df_enhanced["hyperparameters"] = df_enhanced["hyperparameters"].apply(ast.literal_eval)
hyperparams_enhanced_df = df_enhanced["hyperparameters"].apply(pd.Series)
if "distill_method" in hyperparams_enhanced_df.columns:
    hyperparams_enhanced_df = hyperparams_enhanced_df.drop("distill_method", axis=1)
df_enhanced = pd.concat(
    [df_enhanced.drop(["hyperparameters"], axis=1), hyperparams_enhanced_df], axis=1
)

# Define tasks and plot order
plot_task_order = [
    "H2AFZ",
    "H3K9ac",
    "H3K27ac",
    "enhancers",
    "enhancers_types",
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K27me3",
    "H3K36me3",
    "H3K9me3",
    "H4K20me1",
    "splice_sites_donors",
    "splice_sites_all",
    "splice_sites_acceptors",
]

# Teacher scores (Enformer)
teacher_scores = {
    "H3K9ac": 0.626,
    "H3K27ac": 0.5784,
    "enhancers": 0.6435,
    "enhancers_types": 0.5991,
    "promoter_all": 0.7452,
    "promoter_tata": 0.8252,
    "promoter_no_tata": 0.7791,
    "H3K4me1": 0.5603,
    "H3K4me2": 0.6801,
    "H3K4me3": 0.6788,
    "H2AFZ": 0.6066,
    "H3K27me3": 0.6165,
    "H3K36me3": 0.6218,
    "H3K9me3": 0.5038,
    "H4K20me1": 0.6481,
    "splice_sites_donors": 0.8589,
    "splice_sites_all": 0.8384,
    "splice_sites_acceptors": 0.8566,
}

# BPNet baseline scores
bpnet_scores = {
    "H3K9ac": 0.52134,
    "H3K27ac": 0.42754,
    "enhancers": 0.47267,
    "enhancers_types": 0.453,
    "promoter_all": 0.67679,
    "promoter_tata": 0.78333,
    "promoter_no_tata": 0.69826,
    "H3K4me1": 0.46064,
    "H3K4me2": 0.53658,
    "H3K4me3": 0.62384,
    "H2AFZ": 0.4568,
    "H3K27me3": 0.54969,
    "H3K36me3": 0.55461,
    "H3K9me3": 0.32946,
    "H4K20me1": 0.60608,
    "splice_sites_donors": 0.85273,
    "splice_sites_all": 0.7571,
    "splice_sites_acceptors": 0.82799,
}

# Get unique distillation methods and create shorter labels
distill_methods = df_enhanced["distill_method"].unique()

# Create abbreviated labels for methods
method_labels = {
    "vanilla": "VKD",  # Vanilla Knowledge Distillation
    "logit_standard": "LSD",  # Logit Standardization Distillation
    "dkd": "DKD",  # Decoupled Knowledge Distillation
    "dist": "DIST",  # DIST method
}

# Prepare data for plotting
plot_data = []
for method in distill_methods:
    label = method_labels.get(method, method.upper())

    for task in plot_task_order:
        # Get all data for this task and method, then take the first/best result
        task_data = df_enhanced[
            (df_enhanced["task_name"] == task) & (df_enhanced["distill_method"] == method)
        ]
        if not task_data.empty:
            # Take the first occurrence (or you could take max if multiple runs)
            score = task_data.iloc[0]["student_test_mcc"]
            plot_data.append(
                {
                    "Task": task,
                    "Method": label,
                    "Score": score,
                    "Teacher": teacher_scores.get(task, np.nan),
                    "BPNet": bpnet_scores.get(task, np.nan),
                }
            )

df_plot = pd.DataFrame(plot_data)

# Create figure with subplots using sns.barplot
fig, axes = plt.subplots(6, 3, figsize=(24, 36))
axes = axes.flatten()

# Define elegant color palette (shades of green to match Enformer theme)
green_palette = sns.color_palette("Greens_r", n_colors=len(df_plot["Method"].unique()))

for i, task in enumerate(plot_task_order):
    ax = axes[i]

    # Filter data for current task
    task_df = df_plot[df_plot["Task"] == task]

    if not task_df.empty:
        # Sort by method for consistent ordering
        task_df = task_df.sort_values("Method")

        # Create barplot with elegant color palette
        sns.barplot(
            data=task_df,
            x="Method",
            y="Score",
            ax=ax,
            palette=green_palette,
            edgecolor="#27632a",
            linewidth=1.5,
            order=["VKD", "LSD", "DKD", "DIST"],  # Ensure consistent order
        )

        # Add reference lines
        teacher_score = task_df.iloc[0]["Teacher"]
        bpnet_score = task_df.iloc[0]["BPNet"]

        ax.axhline(
            y=teacher_score,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label="Teacher",
        )
        ax.axhline(
            y=bpnet_score, color="#757575", linestyle=":", linewidth=2.5, alpha=0.8, label="BPNet"
        )

        # Value annotations with better positioning
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=14, fontweight="bold")

        # Styling
        ax.set_xlabel("", fontsize=18)
        ax.set_ylabel("MCC Score", fontsize=18, fontweight="bold")
        ax.tick_params(axis="both", labelsize=16)

        # Set y-axis limits with proper padding
        ymin = min(task_df["Score"].min(), bpnet_score) - 0.05
        ymax = max(task_df["Score"].max(), teacher_score) + 0.08
        ax.set_ylim(ymin, ymax)

        # Add legend with better positioning (only for first subplot)
        if i == 0:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color="#2e7d32",
                    linestyle="--",
                    linewidth=2.5,
                    label="Teacher (Enformer)",
                ),
                plt.Line2D(
                    [0], [0], color="#757575", linestyle=":", linewidth=2.5, label="BPNet Baseline"
                ),
            ]
            ax.legend(
                handles=handles,
                loc="upper left",
                fontsize=14,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        # Add task name as subplot title
        ax.text(
            0.5,
            1.02,
            task.replace("_", " ").title(),
            transform=ax.transAxes,
            ha="center",
            fontsize=18,
            fontweight="bold",
        )

        # Grid for better readability
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

# Hide unused subplots
for j in range(len(plot_task_order), len(axes)):
    axes[j].axis("off")

# Adjust layout
plt.tight_layout(rect=[0, 0.02, 1, 0.98])

# Add legend at bottom with method explanations
legend_text = "Distillation Methods: VKD=Vanilla KD | LSD=Logit Standardization | DKD=Decoupled KD | DIST=DIST Method"
fig.text(0.5, 0.01, legend_text, ha="center", fontsize=16, style="italic")

# Save figure
plt.savefig(
    "output/distillation_methods_comparison.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()

print("Distillation methods comparison plot created successfully!")
print("\nMethod Abbreviations:")
print("VKD = Vanilla Knowledge Distillation")
print("LSD = Logit Standardization Distillation")
print("DKD = Decoupled Knowledge Distillation")
print("DIST = DIST Method")
