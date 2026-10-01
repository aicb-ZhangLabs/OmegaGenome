"""Per-task hyperparameter comparison for the Enformer teacher (reads data/distillation_hyperparameter_comparison_enformer.csv)."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.makedirs("output", exist_ok=True)

# ---- Theme (unchanged) ----
sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")

# ---- Load the already-prepared plotting dataframe ----
df_plot = pd.read_csv("data/distillation_hyperparameter_comparison_enformer.csv")

# ---- The task order & everything else identical to your original plotting section ----
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

# Create figure
fig, axes = plt.subplots(6, 3, figsize=(24, 36))
axes = axes.flatten()

# Elegant color palette (same length as unique settings in df_plot)
blue_palette = sns.color_palette("Blues_r", n_colors=df_plot["Setting"].nunique())

for i, task in enumerate(plot_task_order):
    ax = axes[i]
    task_df = df_plot[df_plot["Task"] == task]

    if not task_df.empty:
        sns.barplot(
            data=task_df,
            x="Setting",
            y="Score",
            ax=ax,
            palette=blue_palette,
            edgecolor="#2c3e50",
            linewidth=1.5,
        )

        # Reference lines (read from columns we saved within df_plot)
        teacher_score = task_df.iloc[0]["Teacher"]
        bpnet_score = task_df.iloc[0]["BPNet"]
        ax.axhline(y=teacher_score, color="#e74c3c", linestyle="--", linewidth=2.5, alpha=0.7)
        ax.axhline(y=bpnet_score, color="#34495e", linestyle=":", linewidth=2.5, alpha=0.7)

        # Value labels (unchanged positioning)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=14, fontweight="bold")

        ax.set_xlabel("", fontsize=18)
        ax.set_ylabel("MCC Score", fontsize=18, fontweight="bold")
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        ymin = min(task_df["Score"].min(), bpnet_score) - 0.05
        ymax = max(task_df["Score"].max(), teacher_score) + 0.08
        ax.set_ylim(ymin, ymax)

        if i == 0:
            handles = [
                plt.Line2D(
                    [0], [0], color="#e74c3c", linestyle="--", linewidth=2.5, label="Teacher"
                ),
                plt.Line2D([0], [0], color="#34495e", linestyle=":", linewidth=2.5, label="BPNet"),
            ]
            ax.legend(
                handles=handles,
                loc="upper left",
                fontsize=14,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        ax.text(
            0.5,
            1.02,
            task.replace("_", " ").title(),
            transform=ax.transAxes,
            ha="center",
            fontsize=18,
            fontweight="bold",
        )

        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

# Hide unused axes (if any)
for j in range(len(plot_task_order), len(axes)):
    axes[j].axis("off")

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig(
    "output/distillation_hyperparameter_comparison_enformer.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()
