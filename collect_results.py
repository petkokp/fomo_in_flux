# python collect_results.py --entity petko-petkov987-none --project default_project

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, numpy as np, pandas as pd
import wandb

# ------------------------------ ROLE MAPS (fixed lists you provided) ------------------------------
ADAPTATION_DATASETS = set([
    'ai2diagrams','artbench10','birdsnap','caltech101','caltech256','cars196','cifar100','cifar10',
    'clrs','country211','cub200','df20mini','dollarstreet','domainnet_clipart',
    'domainnet_infograph','domainnet_painting','domainnet_quickdraw',
    'domainnet_sketch','dsprites','dtd','eurosat','fashionmnist','fgvcaircraft',
    'flowers102','food101','fru92','gtsrb','inaturalist2021','mitstates','mnist',
    'monkeys10','mtsd','mvtecad_adapt','objectnet','obsc_animals','obsc_things','oxford_pets',
    'openimages','patternnet','places365','quilt','resisc45','shapes3d','snake_clef','sun397','stl10',
    'svhn','synthclip106','veg200','zappos50k','eurosat'
])

EVALUATION_DATASETS = set([
    'caltech101','caltech256','cars196','cifar10','domainnet_quickdraw','eurosat','fashionmnist',
    'flickr30k','food101','gtsrb','imagenet','imagenet_a','imagenet_d','imagenet_r','imagenet_s',
    'imagenet_v2','mnist','monkeys10','mscoco','oxford_pets','stl10','svhn', "clevr", "mvtecad_eval", "plantvillage"
])

def role_of_dataset(ds: str) -> str:
    in_adapt = ds in ADAPTATION_DATASETS
    in_eval  = ds in EVALUATION_DATASETS
    if in_adapt and in_eval: return "both"
    if in_adapt: return "train"
    if in_eval:  return "eval"
    return "unknown"

# ------------------------------ Helpers ------------------------------
def as_scalar_float(x):
    """Return a plain float or np.nan from series/arrays/scalars."""
    import numpy as _np
    import pandas as _pd
    if x is None: return _np.nan
    if isinstance(x, (_pd.Series, _pd.Index)):
        if len(x) == 0: return _np.nan
        return as_scalar_float(x.iloc[-1])
    if isinstance(x, _np.ndarray):
        if x.size == 0: return _np.nan
        if x.ndim == 0: return float(x.item())
        return as_scalar_float(x.reshape(-1)[-1])
    try:
        return float(x)
    except Exception:
        return _np.nan

def fetch_full_history_df(run):
    """Fetch full history across all pages using scan_history()."""
    rows = []
    for row in run.scan_history():   # robust against pagination
        rows.append(row)
    return pd.DataFrame(rows)

def get_run_config_safe(run):
    cfg = getattr(run, "config", {})
    try:
        if hasattr(cfg, "as_dict"): return cfg.as_dict()
        if hasattr(cfg, "_items"):  return dict(cfg._items)
        if isinstance(cfg, dict):   return cfg
        if hasattr(cfg, "items"):   return dict(cfg.items())
    except Exception:
        pass
    return {}

def parse_name_meta(run):
    method, tasks, n_samples = "unknown", "NA", "NA"
    
    if "tda" in run.name or "vte" in run.name:
        return {"method": "vte", "tasks": "N.A.", "n_samples": "N.A."}
    
    try:
        parts = (run.name or "").split("_")
        method = parts[0]
        tasks = parts[1].replace("task","").replace("s","")
        n_samples = parts[2].replace("samples","")
    except Exception:
        pass
    return {"method": method, "tasks": tasks, "n_samples": n_samples}

def robust_meta(run):
    cfg  = get_run_config_safe(run)
    tags = set(getattr(run, "tags", []) or [])

    # method
    method = cfg.get("continual.method")
    if not method:
        # tag-based fallback
        for m in ("tda","vte","ema","finetune","lora","linear_probe","linear"):
            if m in tags:
                method = m
                break
    if not method:
        method = (run.name or "unknown").split("_")[0].lower()

    # tasks & n_samples
    tasks = cfg.get("experiment.task.num")
    if tasks is None and "experiment.task.num" in run.summary:
        tasks = run.summary["experiment.task.num"]
    n_samples = cfg.get("experiment.task.n_samples")
    if n_samples is None and "experiment.task.n_samples" in run.summary:
        n_samples = run.summary["experiment.task.n_samples"]

    if tasks is None or n_samples is None:
        parsed = parse_name_meta(run)
        if tasks is None: tasks = parsed["tasks"]
        if n_samples is None: n_samples = parsed["n_samples"]

    return {
        "method": str(method),
        "tasks": str(tasks if tasks is not None else "NA"),
        "n_samples": str(n_samples if n_samples is not None else "NA"),
    }

def last_valid_value(df, col):
    if col not in df.columns: return np.nan
    s = df[col]
    s = s[pd.notna(s)]
    if s.empty: return np.nan
    return as_scalar_float(s.iloc[-1])

# ------------------------------ Baselines ------------------------------
def build_zs_baselines(all_runs):
    """
    Build dataset-level ZS baselines.
    Preferred source: runs tagged 'zs' or (zeroshot_only=True & n_samples=0).
    Fallback: start_zeroshot.<dataset>.total.accuracy from ANY run.
    """
    zs_rows = []

    def is_finished(r):
        return r.state in ("finished","crashed","failed")

    def cfg_flags(r):
        cfg = get_run_config_safe(r)
        zso = str(cfg.get("evaluation.zeroshot_only", cfg.get("zeroshot_only", False))).lower() == "true"
        ns0 = str(cfg.get("experiment.task.n_samples", "0")) in ("0","0.0")
        return zso, ns0

    # 1) explicit ZS runs
    zs_candidates = []
    for r in all_runs:
        if not is_finished(r): continue
        tags = set(getattr(r, "tags", []) or [])
        zso, ns0 = cfg_flags(r)
        if "zs" in tags or (zso and ns0):
            zs_candidates.append(r)

    def append_from_df(df, prefix):
        for col in df.columns:
            if col.startswith(prefix+".") and col.endswith(".total.accuracy") and "alldata-" not in col:
                ds = col.split(".")[1]
                vals = df[col].dropna()
                if not vals.empty:
                    zs_rows.append({"dataset": ds, "zs_baseline": round(float(vals.iloc[0]), 4)})

    for zr in zs_candidates:
        zhist = fetch_full_history_df(zr)
        # ZS may be logged in adaptation.* or start_zeroshot.*
        append_from_df(zhist, "adaptation")
        append_from_df(zhist, "start_zeroshot")

    # 2) fallback from arbitrary runs
    if not zs_rows:
        for r in all_runs:
            if not is_finished(r): continue
            h = fetch_full_history_df(r)
            append_from_df(h, "start_zeroshot")

    zs_df = pd.DataFrame(zs_rows)
    if zs_df.empty:
        return zs_df
    zs_df = zs_df.groupby("dataset", as_index=False)["zs_baseline"].mean()
    return zs_df

# ------------------------------ Overall KA/ZS ------------------------------
def fetch_overall_metrics(run):
    """
    Returns KA_final, ZS_final from aggregated keys (alldata-*)
    """
    hist = fetch_full_history_df(run)
    ka = last_valid_value(hist, "adaptation.alldata-trainonly.total.accuracy")
    zs = last_valid_value(hist, "adaptation.alldata-evalonly.total.accuracy")
    return as_scalar_float(ka), as_scalar_float(zs)

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out_overall", default="overall_accuracy_results.csv")
    ap.add_argument("--out_perds", default="per_dataset_results.csv")
    args = ap.parse_args()

    wandb.login()
    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    # Build ZS baselines once
    zs_df = build_zs_baselines(runs)
    if zs_df.empty:
        print("[WARN] No ZS baseline found (explicit or fallback). per-dataset deltas will be NaN.")

    overall_rows = []
    perds_rows = []

    for run in runs:
        if run.state not in ("finished","crashed","failed"):
            continue

        meta = robust_meta(run)

        # --- overall KA/ZS/GM ---
        KA_final, ZS_final = fetch_overall_metrics(run)
        KA_final = as_scalar_float(KA_final)
        ZS_final = as_scalar_float(ZS_final)
        GM_final = (math.sqrt(KA_final * ZS_final)
                    if np.isfinite(KA_final) and np.isfinite(ZS_final) else np.nan)
        overall_rows.append({
            **meta,
            "KA_final": round(KA_final, 2) if np.isfinite(KA_final) else np.nan,
            "ZS_final": round(ZS_final, 2) if np.isfinite(ZS_final) else np.nan,
            "GM_final": round(GM_final, 2) if np.isfinite(GM_final) else np.nan
        })

        # --- per-dataset finals ---
        hist = fetch_full_history_df(run)
        if hist is None or hist.empty:
            continue
        cols = list(hist.columns)

        # We take per-dataset accuracy from adaptation.<dataset>.total.accuracy (last value)
        adapt_cols = [
            c for c in cols
            if c.startswith("adaptation.") and c.endswith(".total.accuracy") and ("alldata-" not in c)
        ]
        for c in adapt_cols:
            dataset = c.split(".")[1]
            final_acc = last_valid_value(hist, c)
            final_acc = as_scalar_float(final_acc)
            perds_rows.append({
                **meta,
                "dataset": dataset,
                "role": role_of_dataset(dataset),
                "acc_final": round(final_acc, 4) if np.isfinite(final_acc) else np.nan
            })

    # --- Save overall ---
    df_overall = pd.DataFrame(overall_rows)
    df_overall.to_csv(args.out_overall, index=False)
    print(f"[OK] wrote {args.out_overall} with {len(df_overall)} rows")

    # --- Save per-dataset with ZS merge ---
    df_perds = pd.DataFrame(perds_rows)
    if df_perds.empty:
        print("[WARN] No per-dataset rows extracted; nothing to write.")
        return

    # merge ZS baselines (dataset-level). If missing, keep NaN
    if not zs_df.empty:
        df_perds = df_perds.merge(zs_df, on="dataset", how="left")
    else:
        df_perds["zs_baseline"] = np.nan

    df_perds["delta_vs_zs"] = round(df_perds["acc_final"] - df_perds["zs_baseline"], 4)

    # tidy order & dtypes
    for c in ["method","tasks","n_samples","dataset","role"]:
        if c in df_perds.columns: df_perds[c] = df_perds[c].astype(str)
    df_perds = df_perds.sort_values(["method","tasks","n_samples","role","dataset"])

    df_perds.to_csv(args.out_perds, index=False)
    print(f"[OK] wrote {args.out_perds} with {len(df_perds)} rows")


if __name__ == "__main__":
    main()
