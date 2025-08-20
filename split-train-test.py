"""
ipo_step6_split.py — Step 6: Train/Test Holdout Split

- Loads scaled data (default: data/ipo_scaled.csv) created in Step 5
- Splits into train/test using a chosen train proportion (default 0.80)
- Uses stratified split on Listing_Gains_Profit to preserve class balance
- Saves: data/X_train.npy, data/X_test.npy, data/y_train.npy, data/y_test.npy,
         data/train.csv, data/test.csv
- Writes a short summary to reports/step6_split_summary.md
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PREDICTORS = ["Issue_Size", "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Issue_Price"]
TARGET = "Listing_Gains_Profit"

def main():
    ap = argparse.ArgumentParser(description="Create train/test holdout split with stratification.")
    ap.add_argument("--in-csv", type=Path, default=Path("data/ipo_scaled.csv"),
                    help="Scaled dataset with predictors in [0,1] and target column.")
    ap.add_argument("--train", type=float, default=0.80,
                    help="Proportion for training set (default 0.80).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    # Load scaled data
    if not args.in_csv.exists():
        sys.exit(f"[ERROR] Missing {args.in_csv}. Run Step 5 to create it.")
    df = pd.read_csv(args.in_csv)

    # Basic checks
    missing_predictors = [c for c in PREDICTORS if c not in df.columns]
    if missing_predictors:
        sys.exit(f"[ERROR] Predictors not found in {args.in_csv}: {missing_predictors}")
    if TARGET not in df.columns:
        sys.exit(f"[ERROR] Target '{TARGET}' not found in {args.in_csv}")

    # Build arrays
    X = df[PREDICTORS].values.astype(float)
    y = df[TARGET].values.astype(int)

    # Stratified holdout
    test_size = 1.0 - args.train
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=args.seed,
        stratify=y,
        shuffle=True,
    )

    # Save arrays
    Path("data").mkdir(parents=True, exist_ok=True)
    np.save("data/X_train.npy", X_tr)
    np.save("data/X_test.npy",  X_te)
    np.save("data/y_train.npy", y_tr)
    np.save("data/y_test.npy",  y_te)

    # Save CSVs for inspection
    train_df = pd.DataFrame(X_tr, columns=PREDICTORS)
    train_df[TARGET] = y_tr
    test_df = pd.DataFrame(X_te, columns=PREDICTORS)
    test_df[TARGET] = y_te
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # Explore shapes & class balance
    def pct_pos(arr): 
        return (100.0 * arr.mean()) if len(arr) else 0.0
    overall_pct = pct_pos(y)
    train_pct = pct_pos(y_tr)
    test_pct  = pct_pos(y_te)

    print("\n=== Holdout summary ===")
    print(f"Chosen train proportion: {args.train:.2f}  (test={test_size:.2f})")
    print(f"Overall:  X={X.shape},  y={y.shape},  pos%={overall_pct:.1f}")
    print(f"Train:    X={X_tr.shape}, y={y_tr.shape}, pos%={train_pct:.1f}")
    print(f"Test:     X={X_te.shape}, y={y_te.shape}, pos%={test_pct:.1f}")
    print("\nFeature count check (should be 5):", X_tr.shape[1])

    # Write markdown summary
    Path("reports").mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Step 6 — Holdout Split Summary\n")
    md.append(f"- Train proportion: **{args.train:.2f}** (Test: **{test_size:.2f}**), stratified by `{TARGET}`.\n")
    md.append(f"- Shapes — Overall: `X={X.shape}`, `y={y.shape}`; Train: `X={X_tr.shape}`, Test: `X={X_te.shape}`.\n")
    md.append(f"- Class balance (%% target=1) — Overall: **{overall_pct:.1f}%**, Train: **{train_pct:.1f}%**, Test: **{test_pct:.1f}%**.\n")
    md.append(f"- Number of predictor features: **{X_tr.shape[1]}** (expected **{len(PREDICTORS)}**).\n")
    md.append("- Artifacts saved: `data/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, and human-readable `data/train.csv`, `data/test.csv`.\n")
    (Path("reports") / "step6_split_summary.md").write_text("".join(md), encoding="utf-8")

    print("\nSaved:")
    print("  data/X_train.npy  data/X_test.npy  data/y_train.npy  data/y_test.npy")
    print("  data/train.csv    data/test.csv")
    print("  reports/step6_split_summary.md")

if __name__ == "__main__":
    main()
