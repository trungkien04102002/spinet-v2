"""Regenerate the two dataset-distribution figures with all-English in-figure
text and no baked-in title (the LaTeX caption carries the description).

Data values are read off the original Vietnamese figures
(rsna_class_distribution.png / spider_class_distribution.png), which are kept.
Outputs new files with an _en suffix so the originals are untouched:
    rsna_class_distribution_en.png
    spider_class_distribution_en.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 21,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "font.family": "DejaVu Sans",
})

# ---------------------------------------------------------------- RSNA (grouped)
conditions = ["Spinal Canal\nStenosis", "Left Foraminal\nNarrowing",
              "Right Foraminal\nNarrowing"]
normal_mild = [87.7, 77.6, 78.0]
moderate    = [7.5, 18.4, 18.1]
severe      = [4.8, 4.1, 3.9]

c_green, c_orange, c_red = "#4FA06E", "#E89A4F", "#E23B3B"
x = np.arange(len(conditions))
w = 0.26

fig, ax = plt.subplots(figsize=(11, 5.2))
b1 = ax.bar(x - w, normal_mild, w, label="Normal/Mild", color=c_green,
            edgecolor="black", linewidth=0.6)
b2 = ax.bar(x,     moderate,    w, label="Moderate",    color=c_orange,
            edgecolor="black", linewidth=0.6)
b3 = ax.bar(x + w, severe,      w, label="Severe",      color=c_red,
            edgecolor="black", linewidth=0.6)

for bars in (b1, b2, b3):
    for r in bars:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylabel("Percentage of samples (%)")
ax.set_xlabel("Conditions")
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 100)
ax.legend(title="Severity", loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=3, frameon=False, columnspacing=1.3, handletextpad=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig("rsna_class_distribution_en.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------- SPIDER (barh)
# top -> bottom as displayed in the original
spider_labels = ["Disc bulging", "Lower endplate damage", "Upper endplate damage",
                 "Disc narrowing", "Modic changes (any)", "Pfirrmann grade $\\geq$ 4",
                 "Disc herniation", "Spondylolisthesis"]
spider_vals = [49.1, 41.0, 40.4, 35.8, 33.8, 31.2, 4.7, 2.8]
# rare positive classes (< 5%) highlighted in red, the rest in blue
c_blue, c_red2 = "#4A90D9", "#E2473C"
spider_colors = [c_red2 if v < 5 else c_blue for v in spider_vals]

y = np.arange(len(spider_labels))[::-1]   # so first item sits at the top
fig, ax = plt.subplots(figsize=(11, 5.6))
bars = ax.barh(y, spider_vals, color=spider_colors, edgecolor="black", linewidth=0.6)
for r, v in zip(bars, spider_vals):
    ax.annotate(f"{v:.1f}%", (v, r.get_y() + r.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=13, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(spider_labels)
ax.set_xlabel("Positive / abnormal rate (%)")
ax.set_xlim(0, 60)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig("spider_class_distribution_en.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("wrote rsna_class_distribution_en.png and spider_class_distribution_en.png")
