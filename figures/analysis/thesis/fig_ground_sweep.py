"""F10 (fig:ground-sweep): frames rejected and test scores vs the ground percentile.
 (a) % of the 385 synced frames rejected by the plane-plausibility check, before / after the 3 deg tilt limit
     source: orchard-weed-detection/reports/height_gate_change_20261008.md section 4 (table 'by ground percentile')
 (b) YOLO v2 + gate on test_C1 (after the fix), centre-rule precision and recall
     source: orchard-weed-detection/results/c1/summary.txt, last block ('over the ground percentile')
"""
from thesis_style import *

PCT = [5, 10, 15, 20, 25]
BEFORE = [23.1, 15.3, 7.8, 1.0, 0.3]
AFTER = [8.3, 5.5, 2.1, 0.3, 0.3]
P = [0.34, 0.39, 0.39, 0.39, 0.39]
R = [0.35, 0.45, 0.45, 0.45, 0.48]
CHOSEN = 15


def main():
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 4.6 * CM),
                            gridspec_kw=dict(wspace=0.32, left=0.085, right=0.985, top=0.80, bottom=0.2))
    ax = axs[0]
    ax.axvspan(CHOSEN - 1.2, CHOSEN + 1.2, color="#EDEDED", lw=0, zorder=0)
    ax.plot(PCT, BEFORE, "o--", color=COL["grey"], ms=3.4, lw=0.9, mfc="white", label="plane free to tilt")
    ax.plot(PCT, AFTER, "o-", color="#222222", ms=3.4, lw=1.1, label=r"tilt $\leq 3^\circ$")
    for x, b, a in zip(PCT, BEFORE, AFTER):
        if b - a > 2: ax.text(x + 0.5, b + 0.4, f"{b:.0f}", fontsize=6, color=COL["grey"], va="bottom")
        if b - a > 2: ax.text(x + 0.5, a - 0.5, f"{a:.1f}", fontsize=6, color="#222222", va="top")
    ax.set_xticks(PCT); ax.set_xlabel("ground percentile"); ax.set_ylabel("frames rejected (%)")
    ax.set_ylim(-1.5, 26); ax.legend(loc="upper right", fontsize=6.5)
    ax.text(CHOSEN, 25.0, "used", fontsize=6.5, ha="center", va="top", color="#555555")
    ax = axs[1]
    ax.axvspan(CHOSEN - 1.2, CHOSEN + 1.2, color="#EDEDED", lw=0, zorder=0)
    ax.plot(PCT, P, "o-", color=COL["yolo"], ms=3.4, lw=1.1, label="precision")
    ax.plot(PCT, R, "s--", color=COL["yolo"], ms=3.2, lw=0.9, mfc="white", label="recall")
    ax.set_xticks(PCT); ax.set_xlabel("ground percentile"); ax.set_ylabel("YOLO v2 + gate, test set")
    ax.set_ylim(0.25, 0.6); ax.legend(loc="lower right", fontsize=6.5)
    for a_, s in zip(axs, "ab"):
        a_.text(-0.02 if s == "b" else -0.1, 1.17, f"({s})", transform=a_.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
    save(fig, "ground_sweep")


if __name__ == "__main__":
    main()
