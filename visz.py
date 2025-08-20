"""
ipo_step3_viz.py — Step 3: Visualizations for IPO gains classification

Creates:
- reports/target_count.png                # class balance of Listing_Gains_Profit
- reports/<feature>_boxplot.png           # outliers per numeric feature
- reports/<feature>_hist_by_class.png     # feature distribution by target (0 vs 1)
- reports/corr_heatmap.png                # correlations among predictors
- reports/step3_visual_summary.md         # short, auto-generated summary paragraph

Usage:
  python ipo_step3_viz.py
  # optional:
  python ipo_step3_viz.py --csv data/Indian_IPO_Market_Data.csv --clean data/ipo_clean.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Canonical feature set we modeled in step 2
PREDICTORS = ["Issue_Size", "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Issue_Price"]
TARGET = "Listing_Gains_Profit"
RAW_TARGET = "Listing_Gains_Percent"

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "issue size": "Issue_Size", "issue_size": "Issue_Size",
        "subscription qib": "Subscription_QIB", "subscription_qib": "Subscription_QIB",
        "subscription hni": "Subscription_HNI", "subscription_hni": "Subscription_HNI",
        "subscription rii": "Subscription_RII", "subscription_rii": "Subscription_RII",
        "subscription total": "Subscription_Total", "subscription_total": "Subscription_Total",
        "issue price": "Issue_Price", "issue_price": "Issue_Price",
        "listing gains percent": RAW_TARGET, "listing_gains_percent": RAW_TARGET,
        "listing gains %": RAW_TARGET, "listing gain %": RAW_TARGET, "gains_on_listing_%": RAW_TARGET,
    }
    lower_map = {k.lower(): v for k, v in rename_map.items()}
    ren = {}
    for c in df.columns:
        key = c.strip().lower().replace("-", " ").replace("/", " ")
        if key in lower_map:
            ren[c] = lower_map[key]
    if ren:
        df = df.rename(columns=ren)
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in set(PREDICTORS + [RAW_TARGET]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_clean_df(raw_csv: Path, clean_csv: Path) -> pd.DataFrame:
    """
    Load cleaned data if present; otherwise build it from the raw CSV.
    Cleaned columns: PREDICTORS + TARGET
    """
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        # If target is not present, rebuild
        if TARGET not in df.columns:
            raise SystemExit(f"{clean_csv} exists but missing {TARGET}. Recreate with step 2.")
        return df

    # Build from raw
    if not raw_csv.exists():
        raise SystemExit(f"CSV not found. Provide --csv or ensure {raw_csv} exists.")
    try:
        df_raw = pd.read_csv(raw_csv, parse_dates=["Date"])
    except Exception:
        df_raw = pd.read_csv(raw_csv)

    df_raw = normalize(df_raw)
    if RAW_TARGET not in df_raw.columns:
        raise SystemExit(f"'{RAW_TARGET}' not found in {raw_csv}.")
    df_raw[TARGET] = (df_raw[RAW_TARGET] > 0).astype(int)

    # Select predictors (drop Subscription_Total to avoid redundancy)
    cols = [c for c in PREDICTORS if c in df_raw.columns] + [TARGET]
    df = df_raw[cols].dropna().copy()
    return df

def save_target_countplot(df: pd.DataFrame, outdir: Path):
    counts = df[TARGET].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Distribution of Target: Listing_Gains_Profit (1=Profit, 0=No Profit)")
    plt.xlabel("Listing_Gains_Profit")
    plt.ylabel("Count")
    out = outdir / "target_count.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()

def save_boxplots(df: pd.DataFrame, outdir: Path):
    for col in PREDICTORS:
        if col not in df.columns:
            continue
        plt.figure()
        plt.boxplot(df[col].dropna().values, vert=True)
        plt.title(f"Outliers Check — {col}")
        plt.ylabel(col)
        out = outdir / f"{col}_boxplot.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()

def save_hist_by_class(df: pd.DataFrame, outdir: Path):
    for col in PREDICTORS:
        if col not in df.columns:
            continue
        plt.figure()
        a = df[df[TARGET] == 0][col].dropna().values
        b = df[df[TARGET] == 1][col].dropna().values
        # Overlaid histograms; no explicit colors per instruction
        plt.hist(a, bins=30, alpha=0.6, label="0: No Profit")
        plt.hist(b, bins=30, alpha=0.6, label="1: Profit")
        plt.title(f"{col} — Distribution by {TARGET}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        out = outdir / f"{col}_hist_by_class.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()

def save_corr_heatmap(df: pd.DataFrame, outdir: Path):
    cols = [c for c in PREDICTORS if c in df.columns]
    if not cols:
        return
    corr = df[cols].corr()
    plt.figure()
    plt.imshow(corr, interpolation="nearest")
    plt.title("Correlation Heatmap — Predictors")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    out = outdir / "corr_heatmap.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return corr

def write_summary(df: pd.DataFrame, corr: pd.DataFrame | None, outdir: Path):
    # Profit percentage
    pct_profit = df[TARGET].mean() * 100.0

    # Simple class-wise means to guide interpretation
    means = df.groupby(TARGET)[[c for c in PREDICTORS if c in df.columns]].mean(numeric_only=True)

    # High-corr pairs
    high_pairs = []
    if corr is not None:
        for i, a in enumerate(corr.columns):
            for j, b in enumerate(corr.columns):
                if j <= i:
                    continue
                r = corr.loc[a, b]
                if abs(r) >= 0.7:
                    high_pairs.append((a, b, float(r)))

    text = []
    text.append("# Step 3 — Visualization Summary\n")
    text.append(f"- **Class balance:** ~{pct_profit:.1f}% of IPOs listed at a profit (target=1).\n")

    # Compare means by class to hint at relationships
    lines = []
    for c in means.columns:
        m0 = means.loc[0, c] if 0 in means.index else np.nan
        m1 = means.loc[1, c] if 1 in means.index else np.nan
        delta = m1 - m0 if (pd.notna(m1) and pd.notna(m0)) else np.nan
        lines.append(f"  - {c}: mean(1)={m1:.2f}, mean(0)={m0:.2f}, Δ={delta:.2f}")
    if lines:
        text.append("- **Feature vs target (class-wise means):**\n" + "\n".join(lines) + "\n")

    if high_pairs:
        bullets = "\n".join([f"  - {a} vs {b}: r={r:.2f}" for a, b, r in high_pairs])
        text.append("- **Strong predictor correlations (|r| ≥ 0.70):**\n" + bullets + "\n")
    else:
        text.append("- **Strong predictor correlations:** None detected above |r| ≥ 0.70.\n")

    text.append(
        "In short: the **count plot** shows overall class balance; **boxplots** reveal potential outliers in the "
        "numeric predictors; **overlaid histograms** suggest how each predictor’s distribution differs between profit "
        "and non-profit classes; and the **correlation heatmap** highlights redundant signals you might drop before modeling.\n"
    )

    out = outdir / "step3_visual_summary.md"
    out.write_text("\n".join(text), encoding="utf-8")
    print(f"Wrote summary: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("data/Indian_IPO_Market_Data.csv"))
    ap.add_argument("--clean", type=Path, default=Path("data/ipo_clean.csv"))
    args = ap.parse_args()

    reports = Path("reports"); reports.mkdir(parents=True, exist_ok=True)

    df = ensure_clean_df(args.csv, args.clean)

    save_target_countplot(df, reports)
    save_boxplots(df, reports)
    save_hist_by_class(df, reports)
    corr = save_corr_heatmap(df, reports)
    write_summary(df, corr, reports)

    print("Saved figures to:", reports.resolve())

if __name__ == "__main__":
    main()
