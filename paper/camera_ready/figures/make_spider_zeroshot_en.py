"""Redraw the SPIDER zero-shot comparison for the conference slide.

The previous figure (ablation_hybrid_vs_naked.png) had four defects: the delta
panel printed its labels on top of the bars, "Pfirrmann" was spelled with one
n, the label names still carried the underscores of the column names, and a
title was baked into the image that repeated the frame title on the slide.

The lower delta panel is dropped as well. On a slide it doubled the ink to
carry information the grouped bars already show, and the audience has about
one minute for this figure.

Values are Table 4 of paper/camera_ready/main.tex.

    python3 figures/make_spider_zeroshot_en.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "font.family": "DejaVu Sans",
})

# label, off-the-shelf BiomedCLIP, RSNA-trained hybrid
ROWS = [
    ("Disc bulging",      0.363, 0.613),
    ("Disc herniation",   0.532, 0.563),
    ("Disc narrowing",    0.403, 0.586),
    ("Upper endplate",    0.569, 0.467),
    ("Lower endplate",    0.558, 0.468),
    ("Pfirrmann",         0.152, 0.155),
    ("Modic",             0.071, 0.018),
    ("Spondylolisthesis", 0.500, 0.028),
]
N_MORPH = 3  # the first three are the disc-morphology group

labels = [r[0] for r in ROWS]
shelf = np.array([r[1] for r in ROWS])
hybrid = np.array([r[2] for r in ROWS])

C_HYB, C_SHELF = "#0072B2", "#9A9A96"
x = np.arange(len(ROWS))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 5.4))

# Mark the group the paper claims a win on, so the eye can find it.
ax.axvspan(-0.6, N_MORPH - 0.4, color="#0072B2", alpha=0.07, zorder=0)
ax.text((N_MORPH - 1) / 2, 0.70, "disc morphology, closest to the RSNA schema",
        ha="center", va="bottom", fontsize=13, color="#0072B2")

b1 = ax.bar(x - w / 2, hybrid, w, label="Hybrid (RSNA-trained), ours",
            color=C_HYB, edgecolor="black", linewidth=0.6, zorder=3)
b2 = ax.bar(x + w / 2, shelf, w, label="BiomedCLIP off the shelf",
            color=C_SHELF, edgecolor="black", linewidth=0.6, zorder=3)

# Labels always sit above the bar, never on it.
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12)

ax.set_ylabel("F1 macro")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylim(0, 0.78)
ax.set_xlim(-0.7, len(ROWS) - 0.3)
ax.legend(loc="upper right", frameon=False, ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
fig.tight_layout()
out = "figures/spider_zeroshot_en.png"
fig.savefig(out, dpi=200)
print("wrote", out)
