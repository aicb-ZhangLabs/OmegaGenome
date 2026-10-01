"""Per-task benchmark panel over the 18 revised Nucleotide Transformer tasks; writes output/nt_revised_benchmark.pdf."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import os  # === NEW: for creating a data folder ===

os.makedirs("output", exist_ok=True)

# Set seaborn theme for a clean and professional look
sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")

# -------(OPTIONAL) Replay mode: set to True to bypass preprocessing and load from CSV -------
REPLAY_FROM_CSV = True  # flip to True when you want to reproduce from saved CSV only
DATA_DIR = "data"  # "data"        # === NEW: where CSVs go
os.makedirs(DATA_DIR, exist_ok=True)


# === NEW: small loader util ===
def load_preprocessed_long(csv_path=f"{DATA_DIR}/model_comparison_long.csv"):
    """Load the long-format dataframe (Task, Model, Score, Model Family, Type) from CSV."""
    df = pd.read_csv(csv_path)
    # Optional: enforce categories for consistent plotting order later if you like
    return df


# If replaying, jump straight to plotting
if REPLAY_FROM_CSV:
    df_long = load_preprocessed_long()

# ------- Plotting (works for both fresh + replay-from-CSV) -------
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

custom_palette = {
    "BPNet": "#8B8B8B",
    "Distilled Nucleotide Transformer": "#A8C8E1",
    "Nucleotide Transformer": "#5692C4",
    "Distilled DNABERT-2": "#FFDAB9",
    "DNABERT-2": "#FFA07A",
    "Enformer": "#3CB371",
    "Enformer Distilled": "#90EE90",
    "Caduceus": "#9370DB",
    "Distilled Caduceus": "#DDA0DD",
}

fig = plt.figure(figsize=(22, 34))
gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.25)

bar_width = 0.4

for idx, task in enumerate(plot_task_order):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    # Filter data for this task
    task_data = df_long[df_long["Task"] == task].copy()

    family_order = [
        "BPNet",
        "Nucleotide Transformer",
        "DNABERT-2",
        "Enformer",
        "Caduceus",
    ]
    group_centers = np.arange(len(family_order))

    for i, family in enumerate(family_order):
        group_center_x = group_centers[i]
        family_data = task_data[task_data["Model Family"] == family]

        if family == "BPNet":
            score = family_data["Score"].iloc[0]
            model_name = family_data["Model"].iloc[0]
            ax.bar(
                group_center_x,
                score,
                width=bar_width,
                color=custom_palette[model_name],
                edgecolor="black",
                linewidth=1.5,
            )
            ax.annotate(
                f"{score:.3f}",
                (group_center_x, score),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            student_data = family_data[family_data["Type"] == "Student"]
            teacher_data = family_data[family_data["Type"] == "Teacher"]
            if not student_data.empty and not teacher_data.empty:
                student_score = student_data["Score"].iloc[0]
                student_model = student_data["Model"].iloc[0]
                teacher_score = teacher_data["Score"].iloc[0]
                teacher_model = teacher_data["Model"].iloc[0]

                student_x = group_center_x - bar_width / 2
                teacher_x = group_center_x + bar_width / 2

                ax.bar(
                    student_x,
                    student_score,
                    width=bar_width,
                    color=custom_palette[student_model],
                    hatch="///",
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax.bar(
                    teacher_x,
                    teacher_score,
                    width=bar_width,
                    color=custom_palette[teacher_model],
                    edgecolor="black",
                    linewidth=1.5,
                )

                ax.annotate(
                    f"{student_score:.3f}",
                    (student_x, student_score),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
                ax.annotate(
                    f"{teacher_score:.3f}",
                    (teacher_x, teacher_score),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # BPNet baseline line (grab from the tidy frame)
    bpnet_rows = task_data[task_data["Model"] == "BPNet"]["Score"].values
    if len(bpnet_rows) > 0:
        bpnet_score = bpnet_rows[0]
        ax.axhline(y=bpnet_score, color="r", linestyle="--", linewidth=3, alpha=0.7)

    ax.set_title(task.replace("_", " ").title(), fontsize=20, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=20)

    ax.set_xticks([0.3, 1.15, 2 + 0.5, 3 + 0.5, 4 + 0.65])
    ax.set_xticklabels(
        ["BPNet", "NT", "DNABERT-2", "Enformer", "Caduceus"],
        rotation=0,
        ha="right",
        fontsize=14,
    )

    scores = task_data["Score"].dropna().values
    if len(scores) > 0:
        min_val, max_val = scores.min(), scores.max()
        padding = (max_val - min_val) * 0.15
        ax.set_ylim(max(0, min_val - padding), max_val + padding)

    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

# Hide any unused subplots
for idx in range(len(plot_task_order), 18):
    row, col = divmod(idx, 3)
    fig.add_subplot(gs[row, col]).axis("off")


legend_elements = [
    Line2D([0], [0], color="r", lw=3, linestyle="--", label="BPNet Baseline"),
    Patch(facecolor=custom_palette["BPNet"], edgecolor="black", label="BPNet"),
    Patch(
        facecolor=custom_palette["Distilled Nucleotide Transformer"],
        edgecolor="black",
        hatch="///",
        label="NT (Student)",
    ),
    Patch(
        facecolor=custom_palette["Nucleotide Transformer"],
        edgecolor="black",
        label="NT (Teacher)",
    ),
    Patch(
        facecolor=custom_palette["Distilled DNABERT-2"],
        edgecolor="black",
        hatch="///",
        label="DNABERT-2 (Student)",
    ),
    Patch(
        facecolor=custom_palette["DNABERT-2"],
        edgecolor="black",
        label="DNABERT-2 (Teacher)",
    ),
    Patch(
        facecolor=custom_palette["Enformer Distilled"],
        edgecolor="black",
        hatch="///",
        label="Enformer (Student)",
    ),
    Patch(
        facecolor=custom_palette["Enformer"],
        edgecolor="black",
        label="Enformer (Teacher)",
    ),
    Patch(
        facecolor=custom_palette["Distilled Caduceus"],
        edgecolor="black",
        hatch="///",
        label="Caduceus (Student)",
    ),
    Patch(
        facecolor=custom_palette["Caduceus"],
        edgecolor="black",
        label="Caduceus (Teacher)",
    ),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.06),
    ncol=5,
    fontsize=18,
    frameon=False,
    # fancybox=True,
    # shadow=True,
)

plt.tight_layout(rect=[0, 0.05, 1, 0.99])
plt.savefig("output/nt_revised_benchmark.pdf", bbox_inches="tight")

print("Grouped plot with manual positioning generated successfully!")
