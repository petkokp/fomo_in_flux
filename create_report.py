# python create_report.py --perds per_dataset_results.csv --out_dir report

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def md_table(df, order=None, floats=(), rnd=2):
    if order:
        df = df.loc[:, [c for c in order if c in df.columns]]
    df = df.copy()
    for c in floats:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.{rnd}f}" if pd.notnull(v) else "")
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join(["---"]*len(df.columns)) + "|\n"
    rows = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False, name=None))
    return head + rows + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perds", required=True, help="per_dataset_results.csv from collect_results.py")
    ap.add_argument("--out_dir", default="compact_report")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--round", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.perds)

    # Focus on fair comparison: eval-only rows
    df = df[df["role"].astype(str) == "eval"].copy()

    # ---------- Pivot (kept as a compact “all results” CSV) ----------
    df["group"] = df["method"] + "|t" + df["tasks"].astype(str) + "|n" + df["n_samples"].astype(str)
    pivot = df.pivot_table(index="group", columns="dataset", values="delta_vs_zs", aggfunc="mean")
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.var(axis=0).sort_values(ascending=False).index
    pivot = pivot.reindex(index=row_order, columns=col_order)
    pivot.to_csv(os.path.join(args.out_dir, "pivot_delta_all_groups.csv"))

    # ---------- Group summary (eval-only) ----------
    def agg(x):
        return pd.Series({
            "rows": len(x),
            "mean_acc_final": x["acc_final"].mean(),
            "std_acc_final": x["acc_final"].std(),
            "mean_delta": x["delta_vs_zs"].mean(),
            "std_delta": x["delta_vs_zs"].std(),
        })
    group_summary = df.groupby(["method","tasks","n_samples"], dropna=False).apply(agg).reset_index()
    for c in ["mean_acc_final","std_acc_final","mean_delta","std_delta"]:
        group_summary[c] = group_summary[c].map(lambda v: np.nan if pd.isna(v) else round(v, args.round))
    group_summary.to_csv(os.path.join(args.out_dir, "group_summary.csv"), index=False)

    # ---------- FIGURE A: Cleveland dot plot (Δ vs ZS by dataset & method) ----------
    dm = df.groupby(["dataset","method"], as_index=False)["delta_vs_zs"].mean()

    preferred = ["tda","vte","finetune","ema","lora","linear"]
    methods_present = [m for m in preferred if m in set(dm["method"].astype(str))]
    extras = [m for m in sorted(dm["method"].unique()) if m not in methods_present]
    methods = methods_present + extras

    pivot_dm = dm.pivot(index="dataset", columns="method", values="delta_vs_zs")
    if any(m in pivot_dm.columns for m in ["tda","vte"]):
        tta_cols = [c for c in ["tda","vte"] if c in pivot_dm.columns]
        score = pivot_dm[tta_cols].mean(axis=1, skipna=True).fillna(pivot_dm.mean(axis=1, skipna=True))
    else:
        score = pivot_dm.mean(axis=1, skipna=True)
    datasets_ordered = score.sort_values().index.tolist()

    y_positions = {ds: i for i, ds in enumerate(datasets_ordered)}
    plt.figure(figsize=(10, max(4, 0.45*len(datasets_ordered))))
    # faint horizontal guides
    xmin = float(np.nanmin(dm["delta_vs_zs"]))
    xmax = float(np.nanmax(dm["delta_vs_zs"]))
    for y in range(len(datasets_ordered)):
        plt.hlines(y, xmin=xmin, xmax=xmax, linewidth=0.3)
    for m in methods:
        sub = dm[dm["method"] == m].copy()
        sub["y"] = sub["dataset"].map(y_positions)
        plt.scatter(sub["delta_vs_zs"], sub["y"], label=m, s=24)
    plt.axvline(0.0, linewidth=1)
    plt.yticks(range(len(datasets_ordered)), datasets_ordered)
    plt.xlabel("Δ vs ZS (eval-only)")
    plt.title("Cleveland dot plot — per-dataset mean Δ vs ZS by method (eval-only)")
    plt.legend(loc="best", ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    cleveland_path = os.path.join(args.out_dir, "cleveland_delta_by_dataset.png")
    plt.savefig(cleveland_path, dpi=200)
    plt.close()

    # ---------- FIGURE B: Global mean Δ per method ----------
    meth = df.groupby("method", dropna=False)["delta_vs_zs"].mean().sort_values(ascending=False)
    plt.figure(figsize=(max(6, 0.5*len(meth)), 3.8))
    plt.bar(range(len(meth)), meth.values)
    plt.axhline(0, linewidth=1)
    plt.xticks(range(len(meth)), meth.index, rotation=15, ha="right")
    plt.ylabel("Mean Δ vs ZS (eval-only)")
    plt.title("Global mean Δ vs ZS by method")
    plt.tight_layout()
    bar_path = os.path.join(args.out_dir, "bar_mean_delta_by_method.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # ---------- FIGURE C: Waterfall plots (one per method) ----------
    # For each method, sort per-dataset mean Δ and plot as a waterfall (contributions).
    method_list = sorted(df["method"].unique())
    for m in method_list:
        dsm = df[df["method"] == m].groupby("dataset", as_index=False)["delta_vs_zs"].mean()
        if dsm.empty:
            continue
        dsm = dsm.sort_values("delta_vs_zs")  # negatives first
        values = dsm["delta_vs_zs"].values
        labels = dsm["dataset"].values

        # Classic “stacked” waterfall: cumulative with baseline at 0
        cum = np.cumsum(values)
        starts = np.concatenate(([0], cum[:-1]))
        plt.figure(figsize=(10, max(3.5, 0.35*len(values))))
        # draw bars; default color scheme (no explicit colors)
        for i, (s, v) in enumerate(zip(starts, values)):
            plt.bar(i, v, bottom=s)
        plt.axhline(0, linewidth=1)
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.ylabel("Δ vs ZS (eval-only)")
        plt.title(f"Waterfall — per-dataset Δ contributions ({m})")
        plt.tight_layout()
        outp = os.path.join(args.out_dir, f"waterfall_{m}.png")
        plt.savefig(outp, dpi=200)
        plt.close()

    # ---------- FIGURE D: Slopegraphs (ZS → Method accuracy by dataset) ----------
    # Build per-dataset mean ZS (averaged across rows; should be constant across groups).
    zs_by_ds = df.groupby("dataset", as_index=False)["zs_baseline"].mean().rename(columns={"zs_baseline":"ZS"})
    for m in method_list:
        dsm_acc = df[df["method"] == m].groupby("dataset", as_index=False)["acc_final"].mean().rename(columns={"acc_final":"ACC"})
        sg = zs_by_ds.merge(dsm_acc, on="dataset", how="inner")
        if sg.empty:
            continue
        # order datasets by improvement (ACC - ZS)
        sg["delta_abs"] = sg["ACC"] - sg["ZS"]
        sg = sg.sort_values("delta_abs")

        plt.figure(figsize=(8, max(3.5, 0.35*len(sg))))
        y = np.arange(len(sg))
        # two vertical “axes”
        plt.vlines(x=0, ymin=-1, ymax=len(sg), linewidth=1)
        plt.vlines(x=1, ymin=-1, ymax=len(sg), linewidth=1)
        # lines from ZS to ACC
        for i, r in sg.iterrows():
            plt.plot([0,1], [y[len(sg)-1 - list(sg.index).index(i)], y[len(sg)-1 - list(sg.index).index(i)]])  # guide lines
        # points
        plt.scatter(np.zeros(len(sg)), y, s=18)
        plt.scatter(np.ones(len(sg)),  y, s=18)
        # labels on left/right
        for i, r in enumerate(sg.itertuples(index=False)):
            yy = i
            plt.text(-0.02, yy, f"{r.dataset}", ha="right", va="center", fontsize=8)
            plt.text(0.02,  yy, f"{r.ZS:.1f}", ha="left", va="center", fontsize=8)
            plt.text(0.98, yy, f"{r.ACC:.1f}", ha="right", va="center", fontsize=8)
        plt.xticks([0,1], ["ZS", m])
        plt.yticks([])
        plt.title(f"Slopegraph — ZS to {m} accuracy by dataset (eval-only)")
        plt.tight_layout()
        outp = os.path.join(args.out_dir, f"slopegraph_{m}.png")
        plt.savefig(outp, dpi=200)
        plt.close()

    # ---------- REPORT.md ----------
    show_cols = ["method","tasks","n_samples","rows","mean_delta","mean_acc_final"]
    tbl = md_table(group_summary.loc[:, show_cols],
                   order=show_cols, floats=["mean_delta","mean_acc_final"], rnd=args.round)

    df2 = df.dropna(subset=["delta_vs_zs"]).copy()
    top = df2.sort_values("delta_vs_zs", ascending=False).head(args.topk)
    bot = df2.sort_values("delta_vs_zs", ascending=True).head(args.topk)
    colsTB = ["method","tasks","n_samples","dataset","delta_vs_zs","acc_final","zs_baseline"]
    top_md = "#### Top improvements (Δ vs ZS)\n\n" + md_table(top.loc[:, colsTB],
             order=colsTB, floats=["delta_vs_zs","acc_final","zs_baseline"], rnd=args.round)
    bot_md = "#### Biggest drops (Δ vs ZS)\n\n" + md_table(bot.loc[:, colsTB],
             order=colsTB, floats=["delta_vs_zs","acc_final","zs_baseline"], rnd=args.round)

    figures_list = [
        "- `cleveland_delta_by_dataset.png` — Cleveland dot plot (per-dataset mean Δ by method).",
        "- `bar_mean_delta_by_method.png` — global mean Δ per method.",
        "- `waterfall_<method>.png` — per-method Δ distribution (sorted contributions).",
        "- `slopegraph_<method>.png` — ZS → method absolute accuracy per dataset."
    ]

    report = f"""# Compact Results Summary (eval-only)

Artifacts:
- `pivot_delta_all_groups.csv` — ALL Δ vs ZS (rows=groups `method|tX|nY`, cols=datasets).
- `group_summary.csv` — per-group means/stdevs (eval-only).
{os.linesep.join(figures_list)}

## Table — Group summary (eval-only)
{tbl}

## Appendix — Global extremes
{top_md}

{bot_md}
"""
    with open(os.path.join(args.out_dir, "REPORT.md"), "w") as f:
        f.write(report)

    print("Wrote compact report to", args.out_dir)
    print(" - pivot_delta_all_groups.csv")
    print(" - group_summary.csv")
    print(" - cleveland_delta_by_dataset.png")
    print(" - bar_mean_delta_by_method.png")
    print(" - waterfall_<method>.png (one per method)")
    print(" - slopegraph_<method>.png (one per method)")
    print(" - REPORT.md")

if __name__ == "__main__":
    main()
