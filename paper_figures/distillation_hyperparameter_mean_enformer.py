"""Task-averaged hyperparameter effect for the Enformer teacher (reads data/distillation_method_comparison_enformer.tsv)."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

os.makedirs("output", exist_ok=True)

# Set theme with specified parameters
sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")

# Assuming df_distill and df_enhanced are already loaded
# csv_path = "data/distillation_method_comparison_enformer.tsv"
# with open(csv_path, "w", encoding="utf-8") as f:
#     f.write(distillation_summary_enhanced_content.strip() + ("\n" if not distillation_summary_enhanced_content.endswith("\n") else ""))


csv_path_hyper = "data/distillation_summary_expanded.csv"
# with open(csv_path, "w", encoding="utf-8") as f:
#     f.write(distillation_summary_enhanced_content.strip() + ("\n" if not distillation_summary_enhanced_content.endswith("\n") else ""))
df_distill = pd.read_csv(csv_path_hyper, sep=",")


# Add option to include teacher scores
INCLUDE_TEACHER = True  # Set to True to show teacher distribution

# Teacher scores (Enformer) for all tasks
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

# ============================================
# PLOT 1: Hyperparameter Settings Distribution
# ============================================

# Prepare data for hyperparameter settings distribution
hyperparam_data = []
for idx, row in df_distill.iterrows():
    ce, kl, mse = row["lambda_ce"], row["lambda_kl"], row["lambda_mse"]
    if mse == 0.0:
        label = f"C{ce:.1f}-K{kl:.1f}"
    else:
        label = f"C{ce:.1f}-K{kl:.1f}-M{mse:.1f}"

    hyperparam_data.append(
        {"Setting": label, "Score": row["student_test_mcc"], "Task": row["task_name"]}
    )

# Add teacher scores if enabled
if INCLUDE_TEACHER:
    for task, score in teacher_scores.items():
        hyperparam_data.append({"Setting": "Teacher", "Score": score, "Task": task})

df_hyperparam = pd.DataFrame(hyperparam_data)

# Get unique settings and order them
unique_settings = [s for s in df_hyperparam["Setting"].unique() if s != "Teacher"]
setting_order = sorted(unique_settings)
if INCLUDE_TEACHER:
    setting_order.append("Teacher")

# Create figure for hyperparameter settings
fig1, ax1 = plt.subplots(1, 1, figsize=(16, 10))

# Create elegant purple color palette
n_settings = len(setting_order) - (1 if INCLUDE_TEACHER else 0)
purple_colors = sns.color_palette("Purples_r", n_colors=n_settings)
if INCLUDE_TEACHER:
    colors = purple_colors + ["#FF6B35"]  # Orange for teacher
else:
    colors = purple_colors

# Create violin plot
parts = ax1.violinplot(
    [df_hyperparam[df_hyperparam["Setting"] == s]["Score"].values for s in setting_order],
    positions=range(len(setting_order)),
    widths=0.7,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

# Style the violin plots
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor("#2d1b69" if i < n_settings else "#CC4422")
    pc.set_linewidth(2)

# Add strip plot for individual points
for i, setting in enumerate(setting_order):
    data = df_hyperparam[df_hyperparam["Setting"] == setting]["Score"].values
    x = np.random.normal(i, 0.03, size=len(data))
    ax1.scatter(
        x, data, alpha=0.5, s=40, color=colors[i], edgecolors="white", linewidth=0.8, zorder=2
    )

# Calculate and plot means
means = [df_hyperparam[df_hyperparam["Setting"] == s]["Score"].mean() for s in setting_order]
stds = [df_hyperparam[df_hyperparam["Setting"] == s]["Score"].std() for s in setting_order]

# Plot mean line (exclude Teacher from line connection)
student_positions = range(len(setting_order) - (1 if INCLUDE_TEACHER else 0))
student_means = means[:-1] if INCLUDE_TEACHER else means
ax1.plot(
    student_positions,
    student_means,
    "o-",
    color="#2d1b69",
    linewidth=3,
    markersize=14,
    markeredgecolor="white",
    markeredgewidth=3,
    label="Student Mean",
    zorder=5,
)

# Add Teacher mean point if included
if INCLUDE_TEACHER:
    ax1.plot(
        len(setting_order) - 1,
        means[-1],
        "o",
        color="#FF6B35",
        markersize=14,
        markeredgecolor="white",
        markeredgewidth=3,
        label="Teacher Mean",
        zorder=5,
    )

# Add mean value annotations with better visibility
for i, (mean, std) in enumerate(zip(means, stds)):
    # White background box for better readability
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")

    # Mean annotation
    color = "#2d1b69" if i < n_settings else "#FF6B35"
    ax1.annotate(
        f"{mean:.3f}",
        xy=(i, mean),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color=color,
        bbox=bbox_props,
        zorder=6,
    )

    # Std annotation
    if std > 0:  # Don't show std for single values
        ax1.annotate(
            f"±{std:.3f}",
            xy=(i, mean),
            xytext=(0, -30),
            textcoords="offset points",
            ha="center",
            fontsize=13,
            color="#666666",
            style="italic",
            zorder=6,
        )

# Styling
ax1.set_xticks(range(len(setting_order)))
ax1.set_xticklabels(setting_order, rotation=45, ha="right", fontsize=15)
ax1.set_ylabel("MCC Score", fontsize=20, fontweight="bold")
ax1.set_xlabel("Hyperparameter Settings", fontsize=20, fontweight="bold")
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_axisbelow(True)
ax1.set_ylim([df_hyperparam["Score"].min() - 0.05, df_hyperparam["Score"].max() + 0.1])

# Add legend
ax1.legend(loc="upper left", fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=0.95)

# Add subtitle
ax1.text(
    0.5,
    -0.5,
    "Distribution across 18 genomic tasks | C=CE Loss, K=KL Loss, M=MSE Loss",
    transform=ax1.transAxes,
    ha="center",
    fontsize=13,
    style="italic",
    color="#666666",
)

plt.tight_layout()
plt.savefig(
    "output/hyperparameter_distribution_elegant.pdf", bbox_inches="tight", facecolor="white"
)
plt.show()
