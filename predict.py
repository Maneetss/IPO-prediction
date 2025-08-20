import argparse, pandas as pd, numpy as np
from pathlib import Path
import joblib, tensorflow as tf

PREDICTORS = ["Issue_Size","Subscription_QIB","Subscription_HNI","Subscription_RII","Issue_Price"]

def main():
    ap = argparse.ArgumentParser(description="Predict IPO listing profit probability for new rows.")
    ap.add_argument("--csv", type=Path, required=True, help="CSV with predictor columns")
    ap.add_argument("--out", type=Path, default=Path("data/predictions.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # minimal safety: select & order predictors
    X = df[PREDICTORS].astype(float).values

    scaler = joblib.load("models/scaler.pkl")
    Xs = scaler.transform(X)

    model = tf.keras.models.load_model("models/ipo_mlp.keras")
    proba = model.predict(Xs, verbose=0).ravel()
    pred  = (proba >= 0.5).astype(int)

    out = df.copy()
    out["prob_profit"] = proba
    out["pred_profit"] = pred

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
