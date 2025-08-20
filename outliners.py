"""
ipo_step4_outliers.py — Step 4: Detect & Treat Outliers (IQR Winsorization)

What it does
------------
1) Loads cleaned data (from step 2). If missing, builds it from the raw CSV.
2) Identifies outliers per numeric predictor using IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
3) Decides which variables to treat (default: abs(skewness) > 1 OR outlier_pct > 5%).
4) Caps values outside bounds to the bounds (winsorization).
5) Saves:
   - reports/outlier_report.csv  (per-feature metrics)
   - data/ipo_clean_winsor.csv   (post-treatment dataset)
   - reports/step4_outliers_summary.md (short rationale paragraph)

Usage
-----
python ipo_step4_outliers.py
# or customize paths/thresholds:
python ipo_step4_outliers.py --csv data/Indian_IPO_Market_Data.csv --clean data/ipo_clean.csv \
  --out data/ipo_clean_winsor.csv --skew 1.0 --pct 5.0
# dry-run (detect only, no capping):
python ipo_step4_outliers.py --dry-run
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PREDICTORS = ["Issue_Size", "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Issue_Price"]
TARGET = "Listing_Gains_Profit"
RAW_TARGET = "Listing_Gains_Percent"

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
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
    for c in set(PREDICTORS + [RAW_TARGET, TARGET]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_clean_df(raw_csv: Path, clean_csv: Path) -> pd.DataFrame:
    """Prefer the step-2 cleaned file; otherwise build it from raw CSV."""
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        if TARGET not in df.columns:  # safety
            raise SystemExit(f"{clean_csv} exists but is missing {TARGET}. Recreate with step 2.")
        return df

    if not raw_csv.exists():
        raise SystemExit(f"CSV not found. Provide --csv or ensure {raw_csv} exists.")
    try:
        df_raw = pd.read_csv(raw_csv, parse_dates=["Date"])
    except Exception:
        df_raw = pd.read_csv(raw_csv)
    df_raw = _normalize(df_raw)
    if RAW_TARGET not in df_raw.columns:
        raise SystemExit(f"'{RAW_TARGET}' not found in {raw_csv}.")
    df_raw[TARGET] = (df_raw[RAW_TARGET] > 0).astype(int)
    cols = [c for c in PREDICTORS if c in df_raw.columns] + [TARGET]
    df = df_raw[cols].dropna().copy()
    return df

def detect_outliers_iqr(s: pd.Series):
    """Return Q1, Q3, IQR, lower, upper, mask for outliers."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        lower = q1
        upper = q3
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return q1, q3, iqr, lower, upper, mask

def main():
    ap = argparse.ArgumentParser(description="Detect & treat outliers with IQR winsorization.")
    ap.add_argument("--csv", type=Path, default=Path("data/Indian_IPO_Market_Data.csv"))
    ap.add_argument("--clean", type=Path, default=Path("data/ipo_clean.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/ipo_clean_winsor.csv"))
    ap.add_argument("--skew", type=float, default=1.0, help="abs(skew) threshold to trigger treatment")
    ap.add_argument("--pct", type=float, default=5.0, help="outlier percentage threshold to trigger treatment")
    ap.add_argument("--dry-run", action="store_true", help="detect only; do not modify values")
    args = ap.parse_args()

    reports = Path("reports"); reports.mkdir(parents=True, exist_ok=True)

    df = _ensure_clean_df(args.csv, args.clean)
    cols = [c for c in PREDICTORS if c in df.columns]

    rows = []
    bounds = {}
    total_n = len(df)

    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3, iqr, lower, upper, mask = detect_outliers_iqr(s)
        out_ct = int(mask.sum())
        out_pct = (out_ct / total_n) * 100.0
        skew = float(s.skew())
        treat = (abs(skew) > args.skew) or (out_pct > args.pct)
        rows.append({
            "feature": col,
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower": lower, "upper": upper,
            "skew": skew,
            "outliers_count": out_ct,
            "outliers_pct": out_pct,
            "treat_flag": treat,
        })
        bounds[col] = (lower, upper, treat)

    report_df = pd.DataFrame(rows).sort_values("outliers_pct", ascending=False)
    report_path = reports / "outlier_report.csv"
    report_df.to_csv(report_path, index=False)

    # Apply winsorization
    df_wins = df.copy()
    changed_counts = []
    for col in cols:
        lower, upper, treat = bounds[col]
        if not treat:
            continue
        before_low = (df_wins[col] < lower).sum()
        before_high = (df_wins[col] > upper).sum()
        if not args.dry_run:
            df_wins[col] = df_wins[col].clip(lower, upper)
        changed_counts.append((col, int(before_low + before_high)))

    # Save output
    if args.dry_run:
        out_path = None
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df_wins.to_csv(args.out, index=False)
        out_path = args.out

    # Build summary paragraph
    treated = [c for c in cols if bounds[c][2]]
    untreated = [c for c in cols if not bounds[c][2]]

    def fmt_pct(x): return f"{x:.1f}%"
    lines = []
    lines.append("# Step 4 — Outlier Strategy & Rationale\n")
    lines.append(
        "We identified outliers using the **IQR rule** (lower bound = Q1 − 1.5×IQR, upper bound = Q3 + 1.5×IQR) on "
        "each numeric predictor. A variable was **selected for treatment** if either its absolute skewness exceeded "
        f"**{args.skew:.1f}** or more than **{args.pct:.1f}%** of rows fell outside the IQR bounds."
    )
    if treated:
        lines.append("\n**Variables treated (winsorized to the IQR bounds):**")
        for col in treated:
            row = report_df.loc[report_df["feature"] == col].iloc[0]
            ch = next((c for c in changed_counts if c[0] == col), (col, 0))[1]
            lines.append(
                f"- {col}: skew={row['skew']:.2f}, outliers={row['outliers_count']} ({fmt_pct(row['outliers_pct'])}), "
                f"bounds=[{row['lower']:.3g}, {row['upper']:.3g}], values capped={ch}"
            )
    else:
        lines.append("\n**No variables met the thresholds for treatment.**")

    if untreated:
        lines.append("\n**Variables inspected but left as-is:** " + ", ".join(untreated))

    if out_path:
        lines.append(f"\nThe post-treatment dataset was saved to **{out_path}**.")
    else:
        lines.append("\nThis was a **dry run**—no values were modified.")

    lines.append(
        "\n**Rationale:** Winsorization preserves the rank order and reduces the influence of extreme tails on model "
        "training without discarding data. If outliers are genuine signals (e.g., very large issues or exceptional "
        "subscription spikes), keeping them bounded avoids instability while retaining their relative magnitude. "
        "If your model is tree-based, you may choose to skip winsorization; for linear/DNN models, capping can "
        "improve training stability."
    )

    summary_path = reports / "step4_outliers_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote: {report_path}")
    print(f"Wrote: {summary_path}")
    if out_path:
        print(f"Wrote: {out_path}")
    print("\nDone.")
    
if __name__ == "__main__":
    main()
