"""
ipo_step5_scale.py — Step 5: Build X (predictors) / y (target) and scale X to [0, 1]

What it does
------------
1) Loads a cleaned dataset (prefers data/ipo_clean_winsor.csv, else data/ipo_clean.csv, else builds from raw CSV).
2) Selects predictors and target:
   - X: Issue_Size, Subscription_QIB, Subscription_HNI, Subscription_RII, Issue_Price
   - y: Listing_Gains_Profit (1 if Listing_Gains_Percent > 0 else 0)
3) Drops rows with missing values in X or y.
4) Scales X to [0, 1] using MinMaxScaler, handling constant columns safely.
5) Prints summary statistics to confirm scaling worked.
6) Saves:
   - data/ipo_scaled.csv (scaled X + y)
   - data/X.npy , data/y.npy
   - reports/step5_scaling_summary.md (short summary paragraph)

Usage
-----
python ipo_step5_scale.py
# or custom paths:
python ipo_step5_scale.py --raw data/Indian_IPO_Market_Data.csv --in-clean data/ipo_clean.csv --in-winsor data/ipo_clean_winsor.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PREDICTORS = ["Issue_Size", "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Issue_Price"]
TARGET = "Listing_Gains_Profit"
RAW_TARGET = "Listing_Gains_Percent"

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
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
    # coerce numeric
    for c in set(PREDICTORS + [RAW_TARGET, TARGET]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_source_df(raw_csv: Path, clean_csv: Path, winsor_csv: Path) -> pd.DataFrame:
    """Prefer winsorized -> clean -> raw."""
    if winsor_csv.exists():
        df = pd.read_csv(winsor_csv)
        return df
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        return df

    if not raw_csv.exists():
        sys.exit(f"[ERROR] No input data found. Expected one of: {winsor_csv}, {clean_csv}, or {raw_csv}")
    try:
        df = pd.read_csv(raw_csv, parse_dates=["Date"])
    except Exception:
        df = pd.read_csv(raw_csv)
    df = normalize_headers(df)
    # create target if needed
    if TARGET not in df.columns:
        if RAW_TARGET not in df.columns:
            sys.exit(f"[ERROR] Raw data missing '{RAW_TARGET}' to create target.")
        df[TARGET] = (df[RAW_TARGET] > 0).astype(int)
    # select predictors + target
    cols = [c for c in PREDICTORS if c in df.columns] + [TARGET]
    df = df[cols].copy()
    return df

def main():
    ap = argparse.ArgumentParser(description="Create X/y and scale X to [0, 1].")
    ap.add_argument("--raw", type=Path, default=Path("data/Indian_IPO_Market_Data.csv"))
    ap.add_argument("--in-clean", type=Path, default=Path("data/ipo_clean.csv"))
    ap.add_argument("--in-winsor", type=Path, default=Path("data/ipo_clean_winsor.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/ipo_scaled.csv"))
    ap.add_argument("--out-X", type=Path, default=Path("data/X.npy"))
    ap.add_argument("--out-y", type=Path, default=Path("data/y.npy"))
    args = ap.parse_args()

    reports = Path("reports"); reports.mkdir(parents=True, exist_ok=True)

    df = load_source_df(args.raw, args.in_clean, args.in_winsor)
    df = normalize_headers(df)

    # Ensure required columns present
    missing_predictors = [c for c in PREDICTORS if c not in df.columns]
    if missing_predictors:
        sys.exit(f"[ERROR] Missing predictors in data: {missing_predictors}")
    if TARGET not in df.columns:
        sys.exit(f"[ERROR] Target column '{TARGET}' not found.")

    # Drop rows with NA in X or y
    keep_cols = PREDICTORS + [TARGET]
    df = df[keep_cols].dropna().reset_index(drop=True)

    # Build arrays
    X = df[PREDICTORS].values.astype(float)
    y = df[TARGET].values.astype(int)

    # Min–Max scale X to [0,1]
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    # Verify scaling via summary stats
    scaled_df = pd.DataFrame(X_scaled, columns=PREDICTORS)
    stats = scaled_df.agg(["min", "max", "mean", "std"]).T

    print("\n=== Shapes ===")
    print(f"X: {X.shape}  X_scaled: {X_scaled.shape}  y: {y.shape}")

    print("\n=== Scaled feature stats (expect min≈0, max≈1) ===")
    print(stats)

    # Save outputs
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = scaled_df.copy()
    out_df[TARGET] = y
    out_df.to_csv(args.out_csv, index=False)

    np.save(args.out_X, X_scaled)
    np.save(args.out_y, y)

    # Write short markdown summary
    pct_profit = 100.0 * (y.sum() / len(y)) if len(y) else 0.0
    lines = []
    lines.append("# Step 5 — Scaling Summary\n")
    lines.append(f"- Built **y** = `{TARGET}` and **X** = {PREDICTORS}.\n")
    lines.append("- Dropped rows with missing values in X or y prior to scaling.\n")
    lines.append("- Applied **Min–Max scaling** to map each predictor to **[0, 1]**.\n")
    lines.append("- Post-scaling checks (per feature): `min`≈0, `max`≈1 (minor deviations are expected if constants exist).\n")
    lines.append(f"- Class balance after filtering: **{pct_profit:.1f}%** positive (target=1).\n")
    lines.append(f"- Saved: `{args.out_csv}` (scaled X + y), `{args.out_X.name}`, `{args.out_y.name}`.\n")
    (reports / "step5_scaling_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote: {args.out_csv}")
    print(f"Wrote: {args.out_X}")
    print(f"Wrote: {args.out_y}")
    print("Wrote: reports/step5_scaling_summary.md")
    print("\nDone.")

if __name__ == "__main__":
    main()
