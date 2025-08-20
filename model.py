"""
ipo_step7_model.py — Step 7: Build & train a dense neural net for IPO listing gain classification

What it does
------------
- Loads train/test arrays from Step 6 (or falls back to CSV if needed)
- Builds a configurable MLP (1–4 hidden layers) with Keras Sequential
- Compiles with binary cross-entropy, tracks accuracy/AUC/precision/recall
- Trains with early stopping + reduce LR; uses a val split from the training set
- Evaluates on the test set; saves predictions, confusion matrix, ROC, history, and model

Usage
-----
python ipo_step7_model.py
# tweak architecture:
python ipo_step7_model.py --depth 3 --units 128 --activation relu --dropout 0.3 --batchnorm --l2 1e-4
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

PREDICTORS = ["Issue_Size", "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Issue_Price"]
TARGET = "Listing_Gains_Profit"

def load_split_arrays():
    """Load X_train, X_test, y_train, y_test from .npy produced in Step 6."""
    data = Path("data")
    paths = {k: data / f"{k}.npy" for k in ["X_train", "X_test", "y_train", "y_test"]}
    if all(p.exists() for p in paths.values()):
        X_train = np.load(paths["X_train"])
        X_test  = np.load(paths["X_test"])
        y_train = np.load(paths["y_train"])
        y_test  = np.load(paths["y_test"])
        return X_train, X_test, y_train, y_test
    # fallback: try from CSV (Step 6 saved these too)
    train_csv = data / "train.csv"
    test_csv  = data / "test.csv"
    if train_csv.exists() and test_csv.exists():
        tr = pd.read_csv(train_csv); te = pd.read_csv(test_csv)
        X_train = tr[PREDICTORS].values.astype(float)
        y_train = tr[TARGET].values.astype(int)
        X_test  = te[PREDICTORS].values.astype(float)
        y_test  = te[TARGET].values.astype(int)
        return X_train, X_test, y_train, y_test
    sys.exit("[ERROR] Could not find split arrays. Run Step 6 (ipo_step6_split.py) first.")

def build_mlp(input_dim:int, depth:int=2, units:int=64, activation:str="relu",
              dropout:float=0.2, l2:float=0.0, batchnorm:bool=False) -> keras.Model:
    """
    Build a configurable MLP:
    - input -> [Dense -> (BN) -> Activation -> (Dropout)] x depth -> Dense(1, sigmoid)
    """
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    m = keras.Sequential(name="ipo_mlp")
    m.add(layers.Input(shape=(input_dim,)))
    for i in range(depth):
        m.add(layers.Dense(units, kernel_regularizer=reg, name=f"dense_{i+1}"))
        if batchnorm:
            m.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        m.add(layers.Activation(activation, name=f"act_{i+1}"))
        if dropout and dropout > 0:
            m.add(layers.Dropout(dropout, name=f"drop_{i+1}"))
    m.add(layers.Dense(1, activation="sigmoid", name="output"))  # binary classification
    return m

def plot_training(history:keras.callbacks.History, outdir:Path):
    hist = pd.DataFrame(history.history)
    outdir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure()
    hist[["loss","val_loss"]].plot(ax=plt.gca())
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Binary Cross-Entropy")
    plt.savefig(outdir/"training_loss.png", bbox_inches="tight", dpi=150); plt.close()

    # AUC (if present)
    if "auc" in hist.columns and "val_auc" in hist.columns:
        plt.figure()
        hist[["auc","val_auc"]].plot(ax=plt.gca())
        plt.title("Training vs Validation AUC")
        plt.xlabel("Epoch"); plt.ylabel("AUC")
        plt.savefig(outdir/"training_auc.png", bbox_inches="tight", dpi=150); plt.close()

def save_confusion_and_roc(y_true, y_prob, outdir:Path):
    outdir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    for (i,j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    plt.savefig(outdir/"confusion_matrix.png", bbox_inches="tight", dpi=150); plt.close()

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
        plt.savefig(outdir/"roc_curve.png", bbox_inches="tight", dpi=150); plt.close()
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2, choices=[1,2,3,4], help="Number of hidden layers")
    ap.add_argument("--units", type=int, default=64, help="Units per hidden layer")
    ap.add_argument("--activation", type=str, default="relu", help="Hidden activation")
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (0 to disable)")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 weight decay (0 to disable)")
    ap.add_argument("--batchnorm", action="store_true", help="Enable BatchNorm after Dense")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    # Load data
    X_train, X_test, y_train, y_test = load_split_arrays()
    input_dim = X_train.shape[1]

    # Build model
    model = build_mlp(
        input_dim=input_dim,
        depth=args.depth,
        units=args.units,
        activation=args.activation,
        dropout=args.dropout,
        l2=args.l2,
        batchnorm=args.batchnorm,
    )

    # Compile
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    print("\nModel summary:")
    model.summary()

    # Callbacks
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=cbs,
        verbose=1,
    )

    # Evaluate
    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names
    print("\nTest metrics:")
    for name, val in zip(metric_names, results):
        print(f"- {name}: {val:.4f}")

    # Predictions & reports
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, digits=3)
    print("\nClassification report:\n", report)

    # Save artifacts
    reports = Path("reports"); models_dir = Path("models"); data_dir = Path("data")
    reports.mkdir(exist_ok=True); models_dir.mkdir(exist_ok=True); data_dir.mkdir(exist_ok=True)

    # History & plots
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(reports/"training_history.csv", index=False)
    plot_training(history, reports)

    # Confusion & ROC
    save_confusion_and_roc(y_test, y_prob, reports)

    # Predictions CSV
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }).to_csv(data_dir/"test_predictions.csv", index=False)

    # Save model
    model_path = models_dir/"ipo_mlp.keras"
    model.save(model_path)

    # Summary markdown
    md = []
    md.append("# Step 7 — Model Summary\n")
    md.append(f"- Architecture: depth={args.depth}, units={args.units}, activation={args.activation}, "
              f"dropout={args.dropout}, batchnorm={args.batchnorm}, l2={args.l2}\n")
    for name, val in zip(metric_names, results):
        md.append(f"- Test {name}: **{val:.4f}**\n")
    (reports/"step7_model_summary.md").write_text("".join(md), encoding="utf-8")

    print("\nSaved:")
    print(" ", reports/"training_history.csv")
    print(" ", reports/"training_loss.png", reports/"training_auc.png")
    print(" ", reports/"confusion_matrix.png", reports/"roc_curve.png")
    print(" ", data_dir/"test_predictions.csv")
    print(" ", model_path)
    print(" ", reports/"step7_model_summary.md")

if __name__ == "__main__":
    main()
