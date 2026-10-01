"""Inference latency and peak-memory bars from the measured metrics table; writes output/latency.pdf."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

os.makedirs("output", exist_ok=True)

df_latency = pd.read_csv("data/latency_metrics.csv")
seq_length_map = {
    "promoter_all": 300,
    "promoter_tata": 300,
    "promoter_no_tata": 300,
    "enhancers": 400,
    "enhancers_types": 400,
    "splice_sites_all": 600,
    "splice_sites_acceptors": 600,
    "splice_sites_donors": 600,
    "H2AFZ": 1000,
    "H3K27ac": 1000,
    "H3K27me3": 1000,
    "H3K36me3": 1000,
    "H3K4me1": 1000,
    "H3K4me2": 1000,
    "H3K4me3": 1000,
    "H3K9ac": 1000,
    "H3K9me3": 1000,
    "H4K20me1": 1000,
    "H3": 1000,
    "H4": 1000,
    "H3K14ac": 1000,
    "H3K79me3": 1000,
}


sns.set_theme(palette="deep")
sns.set_context("poster")

df_latency["Length"] = df_latency["task_name"].map(lambda x: seq_length_map.get(x, 200))

df_latency.rename(
    columns={
        "teacher_total_time_ms": "Teacher Total Time (ms)",
        "student_total_time_ms": "Student Total Time (ms)",
        "speedup_factor": "Speedup",
    },
    inplace=True,
)

cmap = sns.color_palette("Blues", as_cmap=True)
fig = sns.relplot(
    df_latency,
    x="Teacher Total Time (ms)",
    y="Student Total Time (ms)",
    height=7.5,
    hue="Speedup",
    size="Length",
    palette=cmap,
)

fig.ax.axline(xy1=(0, 0), slope=1, color="gray", linestyle="--")
fig.set(xscale="log", yscale="log")
fig.ax.set_xlim(10, 10000)
fig.ax.set_ylim(10, 10000)

# texts = []
# x_data = df_latency['Teacher Total Time (ms)']
# y_data = df_latency['Student Total Time (ms)']
# for i, task in enumerate(df_latency['task_name']):
#     texts.append(fig.ax.text(x_data.iloc[i], y_data.iloc[i], task.replace("_", " ").title(), fontsize=9))

# # Use adjust_text to prevent labels from overlapping
# adjust_text(texts, ax=fig.ax, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

# plt.tight_layout()
plt.savefig("output/latency.pdf", bbox_inches="tight")
