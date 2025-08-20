"""
ipo.py — Step 1: Load & Inspect Indian IPO dataset

Usage:
  python ipo.py                          # uses default: data/Indian_IPO_Market_Data.csv
  python ipo.py --csv path/to/your.csv   # custom path
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# Expected canonical column names for the guided project
EXPECTED = [
    "Date", "IPOName", "Issue_Size",
    "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Subscription_Total",
    "Issue_Price", "Listing_Gains_Percent",
]

# Common alternative names you might see (map -> canonical)
RENAME_MAP = {
    "issue size": "Issue_Size",
    "issue_size": "Issue_Size",
    "subscription qib": "Subscription_QIB",
    "subscription_qib": "Subscription_QIB",
    "subscription hni": "Subscription_HNI",
    "subscription_hni": "Subscription_HNI",
    "subscription rii": "Subscription_RII",
    "subscription_rii": "Subscription_RII",
    "subscription total": "Subscription_Total",
    "subscription_total": "Subscription_Total",
    "issue price": "Issue_Price",
    "issue_price": "Issue_Price",
    "listing gains percent": "Listing_Gains_Percent",
    "listing_gains_percent": "Listing_Gains_Percent",
    "listing gains %": "Listing_Gains_Percent",
    "listing_gains_%": "Listing_Gains_Percent",
    "listing gain %": "Listing_Gains_Percent",
    "listing_gain_%": "Listing_Gains_Percent",
    "gains_on_listing_%": "Listing_Gains_Percent",
}


def coerce_common_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns to numeric (safe even if missing)."""
    numeric_cols = [
        "Issue_Size",
        "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Subscription_Total",
        "Issue_Price", "Listing_Gains_Percent",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants to canonical names and ensure Date is datetime if present."""
    # Build a mapping based on lowercase keys
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    renames = {}
    for col in df.columns:
        key = col.strip().lower().replace("-", " ").replace("/", " ")
        if key in lower_map:
            renames[col] = lower_map[key]
    if renames:
        df = df.rename(columns=renames)

    # Parse Date if present and not already datetime
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False, infer_datetime_format=True)

    return df


def print_required_outputs(df: pd.DataFrame) -> None:
    """Print: head/tail, shape, info, Listing_Gains_Percent stats, and full describe."""
    # For nicer console printing
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 50)

    print("\n=== First five rows ===")
    print(df.head())

    print("\n=== Last five rows ===")
    print(df.tail())

    print("\n=== Shape (rows, columns) ===")
    print(df.shape)

    print("\n=== DataFrame info ===")
    # df.info() prints to stdout; capture via buf if you prefer, but printing is fine for this task.
    df.info()

    if "Listing_Gains_Percent" in df.columns:
        print("\n=== Summary stats: Listing_Gains_Percent ===")
        print(df["Listing_Gains_Percent"].describe())
    else:
        print("\n[WARN] Column 'Listing_Gains_Percent' not found — cannot print its stats.")

    # Describe all variables (include non-numeric; treat datetimes numerically for count/min/max)
    print("\n=== Summary stats: ALL variables ===")
    try:
        print(df.describe(include="all", datetime_is_numeric=True))
    except TypeError:
        # pandas <2.0 compatibility
        print(df.describe(include="all"))


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found at: {csv_path}  (Tip: put it at data/Indian_IPO_Market_Data.csv or use --csv)")
    # Try parsing Date during read; we'll normalize again after
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
    except Exception:
        df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    df = coerce_common_numeric(df)
    # Sort by date if Date exists
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Load and inspect Indian IPO dataset (Step 1).")
    parser.add_argument("--csv", type=Path, default=Path("data/Indian_IPO_Market_Data.csv"),
                        help="Path to the CSV file (default: data/Indian_IPO_Market_Data.csv)")
    args = parser.parse_args()

    df = load_dataframe(args.csv)
    print_required_outputs(df)


if __name__ == "__main__":
    main()
