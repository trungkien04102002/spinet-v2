"""Plot cross-phase Mean F1 macro comparison."""
import matplotlib.pyplot as plt
import numpy as np


def main():
    methods = [
        "Baseline\nRSNA",
        "CBAM\nRSNA",
        "Hybrid\nRSNA",
        "Hybrid\nSPIDER 0-shot",
        "Naked BMC\nSPIDER 0-shot",
        "Vanilla\nSPIDER",
        "CBAM\nSPIDER",
        "Hybrid frozen\nSPIDER",
        "Hybrid unfreeze\nSPIDER",
    ]
    f1m = [0.420, 0.509, 0.516, 0.362, 0.394, 0.610, 0.597, 0.623, 0.622]
    phases = [1, 2, 2, 3, 3, 4, 4, 4, 4]

    phase_colors = {1: "#7986CB", 2: "#43A047", 3: "#FB8C00", 4: "#1E88E5"}
    colors = [phase_colors[p] for p in phases]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, f1m, color=colors, edgecolor="black", linewidth=0.7)

    for i, (bar, v) in enumerate(zip(bars, f1m)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    best_idx = [2, 4, 7]  # Hybrid RSNA, Naked BMC zs, Hybrid frozen SPIDER
    for i in best_idx:
        bars[i].set_edgecolor("#D32F2F")
        bars[i].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Mean F1 macro", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cross-phase Mean F1 macro — SpineNetV2 + BiomedCLIP",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0, 0.75)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=phase_colors[1], label="Phase 1: RSNA baseline"),
        plt.Rectangle((0, 0), 1, 1, fc=phase_colors[2], label="Phase 2: RSNA + CBAM/Hybrid"),
        plt.Rectangle((0, 0), 1, 1, fc=phase_colors[3], label="Phase 3: SPIDER zero-shot"),
        plt.Rectangle((0, 0), 1, 1, fc=phase_colors[4], label="Phase 4: SPIDER transfer"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc="white",
            ec="#D32F2F",
            lw=2.5,
            label="Best in phase",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
    )

    plt.tight_layout()
    out = "experiments/cross_phase_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
