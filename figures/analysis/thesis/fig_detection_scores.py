"""F9 (fig:detection-scores): precision and recall on test_C1, scored vs audited.
 scored P, R + 95 % CI : orchard-weed-detection/results/c1/summary.json  (centre rule, gate at ground percentile 15, 1000 frame resamples)
 audited P (+ range)   : data_set/weed_labelling_20261008/fp_audit/audit_summary.json  (key '<detector>|full'; range = unknown crops all wrong / all right)
Recall is unchanged by the audit (denominator = the 31 labelled weeds), so it has one marker.
"""
import json
from pathlib import Path
from thesis_style import *

ROOT = Path("/home/afonso/thesis")
S = json.load(open(ROOT / "orchard-weed-detection/results/c1/summary.json"))["detectors"]
A = json.load(open(ROOT / "data_set/weed_labelling_20261008/fp_audit/audit_summary.json"))
DET = [("colour_orig+gate", "Colour, original", "colour", 0.45),
       ("colour_tuned+gate", "Colour, tuned", "colour", 1.0),
       ("yolo_v1+gate", "YOLO v1", "yolo", 0.55),
       ("yolo_v2+gate", "YOLO v2", "yolo", 1.0)]


def main():
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 4.4 * CM), sharey=True,
                            gridspec_kw=dict(wspace=0.08, left=0.19, right=0.985, top=0.80, bottom=0.2))
    n = len(DET); y = {d[0]: n - 1 - i for i, d in enumerate(DET)}
    for ax in axs:
        ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(0, 1.0); ax.grid(axis="x", lw=0.3, color="#DDDDDD"); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "", "0.5", "", "1"])
    axs[0].set_yticks([y[d[0]] for d in DET]); axs[0].set_yticklabels([d[1] for d in DET])
    for t, d in zip(axs[0].get_yticklabels(), DET):
        t.set_color(COL[d[2]]); t.set_fontweight("bold")
    rows = []
    for key, name, fam, al in DET:
        c = COL[fam]; yy = y[key]
        p, pci = S[key]["P_c"]["value"], S[key]["P_c"]["ci95"]
        r, rci = S[key]["R_c"]["value"], S[key]["R_c"]["ci95"]
        a = A[key + "|full"]
        pa, pa_lo, pa_hi = a["P_audited"], a["P_lo"], a["P_hi"]
        rows.append((name, p, pci[0], pci[1], pa, pa_lo, pa_hi, r, rci[0], rci[1]))
        ax = axs[0]
        ax.hlines(yy + 0.14, pci[0], pci[1], color=c, lw=1.0, alpha=0.6)
        ax.annotate("", xy=(pa, yy - 0.14), xytext=(p, yy + 0.14),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=c, alpha=0.6, shrinkA=2.2, shrinkB=2.2, mutation_scale=5))
        ax.plot(p, yy + 0.14, "o", ms=4.0, mfc="white", mec=c, mew=0.9, zorder=3)
        ax.hlines(yy - 0.14, pa_lo, pa_hi, color=c, lw=1.6)
        ax.plot(pa, yy - 0.14, "o", ms=4.2, color=c, mec="white", mew=0.3, zorder=3)
        ax = axs[1]
        ax.hlines(yy, rci[0], rci[1], color=c, lw=1.0, alpha=0.6)
        ax.plot(r, yy, "o", ms=4.2, color=c, mec="white", mew=0.3, zorder=3)
    axs[0].set_title("Precision", fontsize=8, pad=4); axs[1].set_title("Recall", fontsize=8, pad=4)
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec="#555555", mew=0.9, label="scored against labels"),
                           Line2D([], [], marker="o", ls="", ms=4, color="#555555", label="after audit")],
                  loc="upper right", fontsize=6, handletextpad=0.2, labelspacing=0.2, borderaxespad=0.2)
    for ax, s in zip(axs, "ab"):
        ax.text(0.0, 1.17, f"({s})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
    save(fig, "detection_scores")
    import csv
    with open(Path(__file__).resolve().parent.parent / "data/detection_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["detector", "P_scored", "P_lo", "P_hi", "P_audited", "P_aud_lo", "P_aud_hi", "R", "R_lo", "R_hi"])
        w.writerows([[r[0]] + [round(v, 3) for v in r[1:]] for r in rows])
    for r in rows: print(r)


if __name__ == "__main__":
    main()
