"""Panels B and C of Figure 2: 18-task mean MCC and relative improvement over the baseline; writes output/aligned_model_comparison_split_lines.pdf."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# ===========================================================================
# 1) CONFIGURATION & DATA LOADING
# ===========================================================================
# Path matches your original code
csv_path = "data/model_comparison_long_final.csv"

# Direct load as requested (no dummy data generation)
df_long = pd.read_csv(csv_path)

# Normalize "Score"
df_long["Score"] = pd.to_numeric(df_long["Score"], errors="coerce")

# Normalize "Type"
df_long["Type"] = df_long["Type"].astype(str).str.strip()
type_map = {
    "Teacher": "Teacher Model", "teacher": "Teacher Model",
    "Student": "Distilled", "student": "Distilled",
    "Baseline": "Baseline", "baseline": "Baseline",
    "Teacher Model": "Teacher Model",
    "Distilled": "Distilled",
}
df_long["Type"] = df_long["Type"].map(type_map).fillna(df_long["Type"])

# Normalize "Model Family"
def update_family_name(row):
    fam = str(row["Model Family"])
    if fam in ["Nucleotide Transformer", "NT", "nt"]: return "Nucleotide Transformer"
    if "dnabert" in fam.lower(): return "DNABERT-2"
    return fam

df_long["Model Family"] = df_long.apply(update_family_name, axis=1)

# Short Names Mapping
name_shortener = {
    "Nucleotide Transformer": "NT",
    "Distilled NT": "Dist. NT",
    "DNABERT-2": "DNABERT-2",
    "Distilled DNABERT-2": "Dist. DNABERT-2",
    "Enformer": "Enformer",
    "Distilled Enformer": "Dist. Enformer",
    "Caduceus": "Caduceus",
    "Distilled Caduceus": "Dist. Caduceus",
    "Best Distillation Model": "Best Distilled",
    "BPNet": "BPNet"
}

df_long["Model Family Short"] = df_long["Model Family"].map(name_shortener).fillna(df_long["Model Family"])

# Handle the "Dist." prefix logic
def enforce_short_naming(row):
    short = row["Model Family Short"]
    typ = row["Type"]
    if typ == "Distilled" and "Dist." not in short and "Best" not in short:
        if short in ["NT", "Enformer", "Caduceus", "DNABERT-2"]:
            return f"Dist. {short}"
    return short

df_long["Model Family Short"] = df_long.apply(enforce_short_naming, axis=1)

# ---------------------------------------------------------------------------
# PREPARE DATASET
# ---------------------------------------------------------------------------
best_distilled = df_long[df_long["Type"] == "Distilled"].groupby("Task", as_index=False)["Score"].max()
best_distilled["Model"] = "Best Distillation Model"
best_distilled["Model Family Short"] = "Best Distilled"
best_distilled["Type"] = "Distilled"

df_plot = pd.concat([df_long, best_distilled], ignore_index=True)

# Colors Palette (Teacher = Light, Student = Dark)
short_palette = {
    "BPNet": "#7f7f7f",
    "NT": "#aec7e8", "Dist. NT": "#1f77b4",
    "DNABERT-2": "#ffbb78", "Dist. DNABERT-2": "#ff7f0e",
    "Enformer": "#98df8a", "Dist. Enformer": "#2ca02c",
    "Caduceus": "#c5b0d5", "Dist. Caduceus": "#9467bd",
    "Best Distilled": "#d62728"
}

# ===========================================================================
# 2) DEFINE COORDINATE SYSTEM
# ===========================================================================
groups_order = ["BPNet", "NT", "DNABERT-2", "Enformer", "Caduceus", "Best Distilled"]
group_centers = {name: i for i, name in enumerate(groups_order)}
offset = 0.2

def get_x_coord(row):
    fam = row["Model Family Short"]
    if fam == "BPNet": return group_centers["BPNet"]
    if fam == "Best Distilled": return group_centers["Best Distilled"]

    base_fam = fam.replace("Dist. ", "")
    if base_fam in group_centers:
        center = group_centers[base_fam]
        if "Dist." in fam or row["Type"] == "Distilled":
            return center + offset
        else:
            return center - offset
    return np.nan

df_plot["X_Coord"] = df_plot.apply(get_x_coord, axis=1)

# ===========================================================================
# 3) PREPARE BAR DATA
# ===========================================================================
baseline_df = df_plot[df_plot["Model Family Short"] == "BPNet"]
if baseline_df.empty:
     baseline_df = df_plot[df_plot["Model"].astype(str).str.contains("BPNet", case=False)]
baseline_score = baseline_df["Score"].mean()

means_by_coord = df_plot.groupby("X_Coord")["Score"].mean()
bar_data = []
target_families = ["NT", "DNABERT-2", "Enformer", "Caduceus"]

for fam in target_families:
    center = group_centers[fam]
    teacher_x = center - offset
    student_x = center + offset

    # Teacher (Light)
    if teacher_x in means_by_coord:
        score = means_by_coord[teacher_x]
        imp = (score - baseline_score) / baseline_score * 100
        bar_data.append({
            "x": teacher_x,
            "height": imp,
            "color": short_palette.get(fam, "#333"),
            "label": f"{imp:.1f}%"
        })

    # Student (Dark)
    if student_x in means_by_coord:
        score = means_by_coord[student_x]
        imp = (score - baseline_score) / baseline_score * 100
        color_key = f"Dist. {fam}"
        bar_data.append({
            "x": student_x,
            "height": imp,
            "color": short_palette.get(color_key, "#888"),
            "label": f"{imp:.1f}%"
        })

# ===========================================================================
# 4) PLOT
# ===========================================================================
sns.set_theme(style="white", font="Helvetica Neue", context="poster")

# Use sharex=False so we can have different labels for top and bottom
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 16),
    sharex=False,
    gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.15}
)

# ---------------------------------------------------------------------------
# X-AXIS LIMITS & ALIGNMENT
# ---------------------------------------------------------------------------
common_xlim = (-0.5, len(groups_order) - 0.5)
ax1.set_xlim(common_xlim)

# --- TOP PLOT: SCATTER ---
df_plot_clean = df_plot.dropna(subset=["X_Coord"])

# Draw Vertical Split Lines
for i in range(len(groups_order) - 1):
    split_x = i + 0.5
    for ax in [ax1, ax2]:
        ax.axvline(x=split_x, color='#999999', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)

# ----------------------------------------------------------------------
# CUSTOM PLOTTING LOGIC: LOLLIPOP STYLES
# ----------------------------------------------------------------------
# Style Constants
POINT_SIZE = 300  # Increased to make white fill/teacher style visible
LW_TEACHER = 3.0
LW_STUDENT = 1.5

# 1. BPNet (Baseline) -> Student Style (Solid fill, black edge)
bp_data = df_plot_clean[df_plot_clean["Model Family Short"] == "BPNet"]
if not bp_data.empty:
    ax1.scatter(
        bp_data["X_Coord"], bp_data["Score"],
        s=POINT_SIZE,
        facecolors=short_palette["BPNet"],
        edgecolors='black',
        linewidth=LW_STUDENT,
        zorder=2
    )

# 2. Best Distilled -> Student Style (Solid fill, black edge)
best_data = df_plot_clean[df_plot_clean["Model Family Short"] == "Best Distilled"]
if not best_data.empty:
    ax1.scatter(
        best_data["X_Coord"], best_data["Score"],
        s=POINT_SIZE,
        facecolors=short_palette["Best Distilled"],
        edgecolors='black',
        linewidth=LW_STUDENT,
        zorder=2
    )

# 3. Main Families (Distinguish Teacher vs. Student)
for fam in target_families:
    # --- A. TEACHER DATA ---
    # Style: White Face, Family Color Edge, Dashed Line
    t_data = df_plot_clean[(df_plot_clean["Model Family Short"] == fam) & (df_plot_clean["Type"] == "Teacher Model")]
    if not t_data.empty:
        t_color = short_palette.get(fam, "#333")
        sc = ax1.scatter(
            t_data["X_Coord"], t_data["Score"],
            s=POINT_SIZE,
            facecolors='white',
            edgecolors=t_color,
            linewidth=LW_TEACHER,
            zorder=2
        )
        sc.set_linestyle('--')  # Key Lollipop Teacher Style

    # --- B. STUDENT DATA ---
    # Style: Family Color Face, Black Edge, Solid Line
    dist_fam = f"Dist. {fam}"
    s_data = df_plot_clean[df_plot_clean["Model Family Short"] == dist_fam]
    if not s_data.empty:
        s_color = short_palette.get(dist_fam, "#888")
        ax1.scatter(
            s_data["X_Coord"], s_data["Score"],
            s=POINT_SIZE,
            facecolors=s_color,
            edgecolors='black',
            linewidth=LW_STUDENT,
            zorder=2
        )

# Plot Mean Markers and Connection Lines
for fam in target_families:
    center = group_centers[fam]
    t_x = center - offset
    s_x = center + offset

    # Connect Teacher & Student Mean
    if t_x in means_by_coord and s_x in means_by_coord:
        t_mean = means_by_coord[t_x]
        s_mean = means_by_coord[s_x]
        ax1.plot([t_x, s_x], [t_mean, s_mean],
                 color='#555555', linestyle='--', linewidth=2, alpha=0.6, zorder=1)

# Annotate each mean point
for x_val, score in means_by_coord.items():
    fam_matches = df_plot_clean[df_plot_clean["X_Coord"] == x_val]["Model Family Short"].mode()

    if not fam_matches.empty:
        fam_name = fam_matches[0]
        color = short_palette.get(fam_name, "#333")

        # Mean Marker
        ax1.scatter(
            x=x_val, y=score,
            marker="_", s=1000, linewidths=4,
            color=color,
            zorder=5
        )

        # Number Annotation
        # MODIFICATION: Added offset to y coordinate to move text up
        annotation_offset = 0.01
        ax1.text(
            x=x_val,
            y=score + annotation_offset,
            s=f"{score:.3f}",
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color='#333333',
            zorder=6
        )

ax1.set_ylabel("MCC Score")
ax1.set_xlabel("")
ax1.grid(False)

# Vertical Y-Ticks for Top Plot
ax1.tick_params(axis='y', labelrotation=90, left=True, length=6, width=1.5)

# --- TOP PLOT X-AXIS LABELS (ALL) ---
ax1.set_xticks(list(group_centers.values()))
ax1.set_xticklabels(list(group_centers.keys()), rotation=0, fontsize=16)

# --- BOTTOM PLOT: BAR ---
max_height = 0
min_height = 0

for bar in bar_data:
    h = bar["height"]
    max_height = max(max_height, h)
    min_height = min(min_height, h)

    ax2.bar(
        x=bar["x"], height=h, width=0.35,
        color='white', edgecolor=bar["color"], hatch='///', linewidth=2,
        zorder=3
    )

    va = 'bottom' if h >= 0 else 'top'
    offset_y = 0.3 if h >= 0 else -1.2
    y_pos = h + offset_y

    ax2.text(bar["x"], y_pos, bar["label"], ha='center', va=va, fontsize=16, color='#333')

if max_height > 0:
    ax2.set_ylim(top=max_height * 1.25)
if min_height < 0:
    ax2.set_ylim(bottom=min_height * 1.25)

ax2.axhline(0, color='#333', linewidth=1)
ax2.set_ylabel("Relative Improvement (%)")
ax2.grid(False)

# Vertical Y-Ticks for Bottom Plot
ax2.tick_params(axis='y', labelrotation=90, left=True, length=6, width=1.5)

# --- BOTTOM PLOT X-AXIS LABELS (FILTERED) ---
filtered_centers = {k: v for k, v in group_centers.items() if k not in ["BPNet", "Best Distilled"]}
ax2.set_xticks(list(filtered_centers.values()))
ax2.set_xticklabels(list(filtered_centers.keys()), rotation=0, ha='center', fontsize=16)

# Legend
legend_elements = [
    Patch(facecolor='white', edgecolor='#cccccc', hatch='///', label='Teacher'),
    Patch(facecolor='white', edgecolor='#444444', hatch='///', label='Distilled')
]
ax2.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=18)

# --- BOUNDARIES ---
for ax in [ax1, ax2]:
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(1.5)

# ---------------------------------------------------------------------------
# FINAL LAYOUT ADJUSTMENT FOR ALIGNMENT
# ---------------------------------------------------------------------------
plt.tight_layout()

# IMPORTANT: Adjust ax2 width to match ax1's data columns exactly
pos1 = ax1.get_position()

# Ratios relative to top plot
total_slots = 6.0
visible_slots = 4.0
start_offset = 1.0 # Skipping index 0 (BPNet)

new_width = pos1.width * (visible_slots / total_slots)
new_left = pos1.x0 + pos1.width * (start_offset / total_slots)

pos2 = ax2.get_position()
# Set new position: shifted right, narrower width, same y-height
ax2.set_position([new_left, pos2.y0, new_width, pos2.height])
# Set limits to exactly matching indices (0.5 to 4.5)
ax2.set_xlim(0.5, 4.5)

os.makedirs('output', exist_ok=True)
plt.savefig("output/aligned_model_comparison_split_lines.pdf", bbox_inches="tight")
plt.show()