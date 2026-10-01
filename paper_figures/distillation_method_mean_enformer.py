"""Task-averaged distillation-method comparison for the Enformer teacher; writes output/method_distribution_elegant.pdf."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

os.makedirs("output", exist_ok=True)

# Set theme with specified parameters
sns.set_theme(style="white", palette="deep", font="Helvetica Neue")
sns.set_context("poster")
INCLUDE_TEACHER = True
# --- Load the enhanced CSV data (from local file) ---
# --- Load the enhanced CSV data (from local file) ---
csv_path_method = "data/distillation_method_comparison_enformer.tsv"
df_enhanced = pd.read_csv(csv_path_method, sep="\t")
# ============================================
# PLOT 2: Distillation Methods Distribution
# ============================================
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
# Prepare data for method distribution
method_data = []
for idx, row in df_enhanced.iterrows():
    method = row["distill_method"]
    method_label = {"vanilla": "VKD", "logit_standard": "LSD", "dkd": "DKD", "dist": "DIST"}.get(
        method, method.upper()
    )

    method_data.append(
        {"Method": method_label, "Score": row["student_test_mcc"], "Task": row["task_name"]}
    )

# Add teacher scores if enabled
if INCLUDE_TEACHER:
    for task, score in teacher_scores.items():
        method_data.append({"Method": "Teacher", "Score": score, "Task": task})

df_method = pd.DataFrame(method_data)

# Method order
method_order = ["VKD", "LSD", "DKD", "DIST"]
if INCLUDE_TEACHER:
    method_order.append("Teacher")

# Create figure for distillation methods
fig2, ax2 = plt.subplots(1, 1, figsize=(16, 10))

# Create elegant orange color palette
n_methods = len(method_order) - (1 if INCLUDE_TEACHER else 0)
orange_colors = sns.color_palette("Oranges_r", n_colors=n_methods)
if INCLUDE_TEACHER:
    colors = orange_colors + ["#6A4C93"]  # Purple for teacher
else:
    colors = orange_colors

# Create violin plot
parts = ax2.violinplot(
    [df_method[df_method["Method"] == m]["Score"].values for m in method_order],
    positions=range(len(method_order)),
    widths=0.65,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

# Style the violin plots
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor("#CC5500" if i < n_methods else "#4A3066")
    pc.set_linewidth(2)

# Add strip plot for individual points
for i, method in enumerate(method_order):
    data = df_method[df_method["Method"] == method]["Score"].values
    x = np.random.normal(i, 0.03, size=len(data))
    ax2.scatter(
        x, data, alpha=0.5, s=40, color=colors[i], edgecolors="white", linewidth=0.8, zorder=2
    )

# Calculate and plot means
means = [df_method[df_method["Method"] == m]["Score"].mean() for m in method_order]
stds = [df_method[df_method["Method"] == m]["Score"].std() for m in method_order]

# Plot mean line (exclude Teacher from line)
student_positions = range(len(method_order) - (1 if INCLUDE_TEACHER else 0))
student_means = means[:-1] if INCLUDE_TEACHER else means
ax2.plot(
    student_positions,
    student_means,
    "o-",
    color="#CC5500",
    linewidth=3,
    markersize=14,
    markeredgecolor="white",
    markeredgewidth=3,
    label="Student Mean",
    zorder=5,
)

# Add Teacher mean point if included
if INCLUDE_TEACHER:
    ax2.plot(
        len(method_order) - 1,
        means[-1],
        "o",
        color="#6A4C93",
        markersize=14,
        markeredgecolor="white",
        markeredgewidth=3,
        label="Teacher Mean",
        zorder=5,
    )

# Add mean value annotations with improved visibility
for i, (mean, std) in enumerate(zip(means, stds)):
    # White background for readability
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")

    # Mean annotation
    color = "#CC5500" if i < n_methods else "#6A4C93"
    ax2.annotate(
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
    if std > 0:
        ax2.annotate(
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

# Add horizontal reference line for overall student mean
student_data = df_method[df_method["Method"] != "Teacher"] if INCLUDE_TEACHER else df_method
overall_mean = student_data["Score"].mean()
ax2.axhline(y=overall_mean, color="#888888", linestyle="--", linewidth=1.5, alpha=0.5)
ax2.text(
    len(method_order) - 0.5,
    overall_mean + 0.01,
    f"Student Avg: {overall_mean:.3f}",
    fontsize=12,
    color="#666666",
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
)

# Styling
ax2.set_xticks(range(len(method_order)))
ax2.set_xticklabels(method_order, fontsize=17)
ax2.set_ylabel("MCC Score", fontsize=20, fontweight="bold")
ax2.set_xlabel("Distillation Method", fontsize=20, fontweight="bold")
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_axisbelow(True)
ax2.set_ylim([df_method["Score"].min() - 0.05, df_method["Score"].max() + 0.1])

# Add legend
ax2.legend(loc="upper left", fontsize=14, frameon=True, fancybox=True, shadow=True, framealpha=0.95)

# Add subtitle
method_text = "VKD=Vanilla KD | LSD=Logit Standardization | DKD=Decoupled KD | DIST=DIST Method"
ax2.text(
    0.5,
    -0.12,
    method_text,
    transform=ax2.transAxes,
    ha="center",
    fontsize=13,
    style="italic",
    color="#666666",
)

plt.tight_layout()
plt.savefig("output/method_distribution_elegant.pdf", bbox_inches="tight", facecolor="white")
plt.show()

print("Elegant distribution plots created successfully!")
print("\nSummary Statistics:")
# print("\nHyperparameter Settings:")

# for setting in setting_order:
#     data = df_hyperparam[df_hyperparam['Setting'] == setting]['Score']
#     print(f"{setting}: Mean={data.mean():.3f}, Std={data.std():.3f}")

print("\nDistillation Methods:")
for method in method_order:
    data = df_method[df_method["Method"] == method]["Score"]
    print(f"{method}: Mean={data.mean():.3f}, Std={data.std():.3f}")
