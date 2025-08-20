"""
ipo_step2.py — Step 2: Preprocess & Explore for IPO gains classification

What it does:
1) Creates binary target Listing_Gains_Profit = 1 if Listing_Gains_Percent > 0 else 0
2) Reports missing values (count & %)
3) Prints summary stats and selects modeling features
4) Shows target distribution and % of profitable IPOs
5) Saves a cleaned dataset to data/ipo_clean.csv

Usage:
  python ipo_step2.py                         # assumes data/Indian_IPO_Market_Data.csv
  python ipo_step2.py --csv path/to/file.csv  # custom file
  python ipo_step2.py --out data/ipo_clean.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Accept common header variations and normalize to canonical names
RENAME_MAP = {
    "issue size": "Issue_Size", "issue_size": "Issue_Size",
    "subscription qib": "Subscription_QIB", "subscription_qib": "Subscription_QIB",
    "subscription hni": "Subscription_HNI", "subscription_hni": "Subscription_HNI",
    "subscription rii": "Subscription_RII", "subscription_rii": "Subscription_RII",
    "subscription total": "Subscription_Total", "subscription_total": "Subscription_Total",
    "issue price": "Issue_Price", "issue_price": "Issue_Price",
    "listing gains percent": "Listing_Gains_Percent", "listing_gains_percent": "Listing_Gains_Percent",
    "listing gains %": "Listing_Gains_Percent", "listing gain %": "Listing_Gains_Percent",
    "gains_on_listing_%": "Listing_Gains_Percent",
}

CANON_NUMERIC = [
    "Issue_Size",
    "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Subscription_Total",
    "Issue_Price", "Listing_Gains_Percent",
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    renames = {}
    for col in df.columns:
        key = col.strip().lower().replace("-", " ").replace("/", " ")
        if key in lower_map:
            renames[col] = lower_map[key]
    if renames:
        df = df.rename(columns=renames)
    # Parse Date if present
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in CANON_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    p = argparse.ArgumentParser(description="Step 2: Build binary target + explore & clean.")
    p.add_argument("--csv", type=Path, default=Path("data/Indian_IPO_Market_Data.csv"))
    p.add_argument("--out", type=Path, default=Path("data/ipo_clean.csv"))
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"[ERROR] CSV not found at: {args.csv}")

    # Load and normalize
    try:
        df = pd.read_csv(args.csv, parse_dates=["Date"])
    except Exception:
        df = pd.read_csv(args.csv)
    df = normalize_columns(df)
    df = coerce_numeric(df)
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    # 1) Binary target
    if "Listing_Gains_Percent" not in df.columns:
        sys.exit("[ERROR] 'Listing_Gains_Percent' not found in the CSV. Please ensure the column exists.")
    df["Listing_Gains_Profit"] = (df["Listing_Gains_Percent"] > 0).astype(int)

    # 2) Missing values report
    miss_ct = df.isna().sum()
    miss_pct = (miss_ct / len(df) * 100).round(2)
    miss = pd.DataFrame({"missing_count": miss_ct, "missing_pct": miss_pct}).sort_values("missing_count", ascending=False)

    print("\n=== Missing values by column ===")
    print(miss[miss["missing_count"] > 0] if miss["missing_count"].any() else "No missing values 🎉")

    # 3) Pick modeling variables (keep numeric signals, drop ID/text/redundant)
    # We’ll use: Issue_Size, Subscription_QIB/HNI/RII, Issue_Price
    # Note: Subscription_Total is roughly QIB+HNI+RII; drop it to reduce redundancy.
    candidate_feats = ["Issue_Size", "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Issue_Price"]
    features = [c for c in candidate_feats if c in df.columns]

    print("\n=== Candidate modeling features ===")
    print(features if features else "[WARN] None of the candidate features found.")

    print("\n=== Summary stats (selected features + Listing_Gains_Percent) ===")
    sel_for_stats = list(dict.fromkeys(features + ["Listing_Gains_Percent"]))  # preserve order, unique
    print(df[sel_for_stats].describe().T if sel_for_stats else "[WARN] No features to describe.")

    # 4) Target distribution: Does an average IPO list at a profit?
    vc = df["Listing_Gains_Profit"].value_counts(dropna=False)
    vc_norm = df["Listing_Gains_Profit"].value_counts(normalize=True, dropna=False) * 100

    print("\n=== Target distribution (Listing_Gains_Profit: 1=profit, 0=non-profit) ===")
    print(vc.sort_index())
    print("\n=== Target distribution % ===")
    print(vc_norm.sort_index().round(2))

    pct_profit = vc_norm.get(1, 0.0)
    if pct_profit > 50:
        tendency = "On average, IPOs tend to list at a profit."
    elif pct_profit < 50:
        tendency = "On average, IPOs tend not to list at a profit."
    else:
        tendency = "On average, IPOs are evenly split between profit and non-profit."
    print(f"\n=> % of IPOs that listed at a profit: {pct_profit:.2f}%")
    print(f"=> {tendency}")

    # 5) Save cleaned dataset used for modeling
    needed_cols = features + ["Listing_Gains_Profit"]
    clean = df[needed_cols].dropna().copy()

    print(f"\n=== Save cleaned data ===")
    print(f"Original rows: {len(df)}, Clean rows (after dropping NA in selected features/target): {len(clean)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
