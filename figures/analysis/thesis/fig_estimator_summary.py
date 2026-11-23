"""F4 (fig:estimator-summary): ATE, RPE50, map sharpness and compute per estimator.

Every number is computed or read from a file; the run table is written to
figures/data/estimator_runs.csv.
 ATE / RPE50  orchard-slam/analysis/ate_vs_gps.py::evaluate on the trajectories listed in RUNS (one dot per run)
 map sharpness orchard-slam/data/thesis_jobs_20261008/entropy_fair/mme_summary.csv, passes b_J1kf, b_J2kf, c_J2odom
               (exact pose for every scan; compare estimators only within a pass -> plotted as MME - MME[LIO-SAM])
 compute      orchard-slam/data/thesis_jobs_20261008/compute_table.md (typed in below, source line per value)
 FAST-LIO     orchard-slam/data/slam_runs_report_20261004-05.md section 5: 16 configurations listed there (text says ~18), none completed
"""
import sys, csv
from pathlib import Path
import numpy as np
from thesis_style import *

ROOT = Path("/home/afonso/thesis/orchard-slam")
sys.path.insert(0, str(ROOT / "analysis"))
import ate_vs_gps as av

T7 = ROOT / "data/thesis_jobs_20261007/for_thesis_20261007/trajectories"
T8 = ROOT / "data/thesis_jobs_20261008/for_thesis_20261008/trajectories"
RUNS = {
    "LIO-SAM": [("r08", T7 / "liosam_baseline_r08_keyframes.csv"), ("rep1", T7 / "liosam_rep1_keyframes.csv"),
                ("rep2", T7 / "liosam_rep2_keyframes.csv"), ("J1", T7 / "liosam_J1_baseline_loops_keyframes.csv")],
    "KISS-ICP": [("ng_p20", T7 / "kiss_tuned_ng_p20_nodeskew.csv"), ("rep1", T7 / "kiss_tuned_rep1.csv"),
                 ("rep2", T7 / "kiss_tuned_rep2.csv"), ("rep3", T8 / "kiss_tuned_rep3.csv")],
    "DLIO": [("base_full", T7 / "dlio_base_full.csv"), ("J5b", T7 / "dlio_J5b_rep4.csv")],
}
KEYS = {"LIO-SAM": "liosam", "KISS-ICP": "kiss", "DLIO": "dlio"}
# compute_table.md: mean cores at 1x (range over runs) and peak cores over 10 s (range)
CORES = {"LIO-SAM": ((0.68, 0.72), (1.20, 1.25)), "KISS-ICP": ((1.55, 1.55), (3.3, 3.3)), "DLIO": ((0.31, 0.31), (0.55, 0.55))}
MME_CSV = ROOT / "data/thesis_jobs_20261008/entropy_fair/mme_summary.csv"
PASSES = [("b_J1kf", "liosam_J1_kf", "o", "exact poses, J1 keyframe scans"),
          ("b_J2kf", "liosam_J2_kf", "s", "exact poses, J2 keyframe scans"),
          ("c_J2odom", "liosam_J2_odom", "^", "exact poses, 2.5 Hz scans")]
MME_LABEL = {"LIO-SAM": None, "KISS-ICP": "kiss_tuned_rep1", "DLIO": "dlio_base_full"}


def main():
    g = av.load_gps()
    rows = []
    res = {}
    for est, runs in RUNS.items():
        res[est] = []
        for name, p in runs:
            r = av.evaluate(av.load_traj(p), g)
            res[est].append((r["ate"], r["rpe50"]))
            rows.append(dict(estimator=est, run=name, ate_m=round(r["ate"], 3), rpe50_pct=round(r["rpe50"], 2),
                             t_end_s=round(r["t_end"]), file=str(p.relative_to(ROOT.parent))))
    outdir = Path(__file__).resolve().parent.parent / "data"
    with open(outdir / "estimator_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(r["estimator"], r["run"], r["ate_m"], r["rpe50_pct"], r["t_end_s"])

    mme = {}
    for row in csv.DictReader(open(MME_CSV)):
        mme[(row["mode"], row["label"])] = float(row["mme"])

    ests = ["LIO-SAM", "KISS-ICP", "DLIO", "FAST-LIO"]
    y = {e: len(ests) - 1 - i for i, e in enumerate(ests)}
    fig, axs = plt.subplots(1, 4, figsize=(FIG_W, 4.6 * CM), sharey=True,
                            gridspec_kw=dict(wspace=0.12, left=0.115, right=0.995, top=0.82, bottom=0.2))
    for ax in axs:
        ax.set_ylim(-0.6, len(ests) - 0.4); ax.grid(axis="x", lw=0.3, color="#DDDDDD"); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    axs[0].set_yticks([y[e] for e in ests]); axs[0].set_yticklabels(ests)
    for t, e in zip(axs[0].get_yticklabels(), ests):
        if e in KEYS: t.set_color(COL[KEYS[e]]); t.set_fontweight("bold")
        else: t.set_color(COL["grey"])

    rng = np.random.default_rng(1)
    for k, (ax, title, xl) in enumerate([(axs[0], "ATE vs GNSS", "RMSE (m)"), (axs[1], "RPE over 50 m", "% of path")]):
        for e in KEYS:
            v = np.array([r[k] for r in res[e]])
            ax.hlines(y[e], v.min(), v.max(), color=COL[KEYS[e]], lw=2.2, alpha=0.35, capstyle="round")
            ax.plot(v, y[e] + rng.uniform(-0.12, 0.12, len(v)), "o", ms=3.2, color=COL[KEYS[e]], mec="white", mew=0.3, zorder=3)
        ax.set_title(title, fontsize=8, pad=4); ax.set_xlabel(xl)
        ax.text(0.5, y["FAST-LIO"], "diverged in every\nconfiguration tried", transform=ax.get_yaxis_transform(), ha="center",
                va="center", fontsize=6.5, color=COL["grey"], style="italic")
    axs[0].set_xlim(1.6, 2.1); axs[1].set_xlim(4.4, 5.0)
    axs[0].set_xticks([1.6, 1.8, 2.0]); axs[1].set_xticks([4.4, 4.6, 4.8, 5.0])
    axs[0].text(0.02, 0.0, "", transform=axs[0].transAxes)

    ax = axs[2]
    for e in KEYS:
        for pas, ref, mk, _ in PASSES:
            lab = ref if e == "LIO-SAM" else MME_LABEL[e]
            d = mme[(pas, lab)] - mme[(pas, ref)]
            ax.plot(d, y[e], mk, ms=3.4, mfc=COL[KEYS[e]], mec="white", mew=0.3, zorder=3)
    ax.axvline(0, color=COL["grey"], lw=0.5)
    ax.set_xlim(-0.02, 0.17); ax.set_xticks([0, 0.05, 0.1, 0.15])
    ax.set_title("Map blur vs LIO-SAM", fontsize=8, pad=4); ax.set_xlabel(r"$\Delta$MME (higher = blurrier)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=m, ls="", ms=3.4, mfc="#777777", mec="white", mew=0.3, label=["J1 keyframes", "J2 keyframes", "2.5 Hz scans"][i])
                       for i, (_, _, m, _) in enumerate(PASSES)],
              loc="lower right", fontsize=6, handletextpad=0.1, labelspacing=0.15, borderaxespad=0.1)

    ax = axs[3]
    for e in KEYS:
        (m0, m1), (p0, p1) = CORES[e]
        ax.hlines(y[e], m0, p1, color=COL[KEYS[e]], lw=0.8, alpha=0.5)
        ax.plot([m0, m1], [y[e]] * 2, "o", ms=3.6, color=COL[KEYS[e]], mec="white", mew=0.3, zorder=3)
        ax.plot([p0, p1], [y[e]] * 2, "o", ms=3.6, mfc="white", mec=COL[KEYS[e]], mew=0.8, zorder=3)
    ax.axvline(4, color=COL["grey"], lw=0.5, ls=":"); ax.text(3.95, 3.45, "4 vCPU", fontsize=6, color=COL["grey"], ha="right", va="center")
    ax.set_xlim(0, 4.2); ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_title("CPU at real time", fontsize=8, pad=4); ax.set_xlabel("cores")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.6, mfc="#777777", mec="white", mew=0.3, label="mean"),
                       Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec="#777777", mew=0.8, label="peak (10 s)")],
              loc="lower right", fontsize=6, handletextpad=0.1, labelspacing=0.15, borderaxespad=0.1)
    for ax, s in zip(axs, "abcd"): panel_label(ax, f"({s})", x=-0.02, y=1.0) if False else None
    for ax, s in zip(axs, "abcd"):
        ax.text(0.0, 1.17, f"({s})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
    save(fig, "estimator_summary")


if __name__ == "__main__":
    main()
