"""The paper's Figure 2 severity distribution, redrawn for the infographic.

Same data, same grouping, same numbers as figures/make_dataset_distribution_en.py
and therefore as Figure 2 of the paper, so a reader can check one against the
other. Two things change, both because the destination is different:

  transparent background, since the page is not white and a white plot panel
  would sit on it as a visible box;

  the page's palette instead of green/orange/red, so the infographic keeps one
  colour system. Severe stays red in both, so the element that carries the
  argument looks the same in the paper, the slides and here.

Sized for a 94 mm wide slot, so in-figure type is set large enough to survive
the reduction.

    python3 figures/make_dist_compact.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 21, "font.family": "DejaVu Sans"})

conditions = ["Spinal canal\nstenosis", "Left foraminal\nnarrowing",
              "Right foraminal\nnarrowing"]
normal_mild = [87.7, 77.6, 78.0]
moderate    = [7.5, 18.4, 18.1]
severe      = [4.8, 4.1, 3.9]

C_NM, C_MOD, C_SEV = "#C3C9CF", "#8B939B", "#E23B3B"
INK = "#55595E"
x = np.arange(len(conditions))
w = 0.27

fig, ax = plt.subplots(figsize=(9.4, 3.9))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

bars = [ax.bar(x - w, normal_mild, w, label="Normal / Mild", color=C_NM),
        ax.bar(x,     moderate,    w, label="Moderate",      color=C_MOD),
        ax.bar(x + w, severe,      w, label="Severe",        color=C_SEV)]

for group, vals in zip(bars, (normal_mild, moderate, severe)):
    for r, v in zip(group, vals):
        ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=17, color=INK)

ax.set_ylabel("% of samples", color=INK, fontsize=19)
ax.set_xticks(x)
ax.set_xticklabels(conditions, color=INK, fontsize=19)
ax.set_ylim(0, 105)
ax.tick_params(axis="y", colors=INK, labelsize=17)
ax.tick_params(axis="x", length=0)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#C9CDD2")
ax.grid(axis="y", linestyle="-", color="#DDE2E6", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
          frameon=False, fontsize=19, labelcolor=INK, handlelength=1.2,
          columnspacing=1.6, handletextpad=0.5)
fig.tight_layout()
out = "paper/camera_ready/figures/rsna_dist_compact.png"
fig.savefig(out, dpi=200, transparent=True)
print("wrote", out)
