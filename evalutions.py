"""
ipo_step8_evaluate.py — Evaluate trained model on train & test and summarize.
- Loads arrays from data/ (X_train.npy, X_test.npy, y_train.npy, y_test.npy)
- Loads model from models/ipo_mlp.keras by default (or evaluates the last in-memory model if run after training)
"""

from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

def load_arrays():
    d = Path("data")
    paths = [d/"X_train.npy", d/"X_test.npy", d/"y_train.npy", d/"y_test.npy"]
    if not all(p.exists() for p in paths):
        sys.exit("[ERROR] Split arrays not found. Run the holdout split first.")
    return (np.load(paths[0]), np.load(paths[1]),
            np.load(paths[2]), np.load(paths[3]))

def main():
    X_train, X_test, y_train, y_test = load_arrays()

    # Load saved model
    mpath = Path("models/ipo_mlp.keras")
    if not mpath.exists():
        sys.exit("[ERROR] Model file not found at models/ipo_mlp.keras. Train and save first.")
    model = tf.keras.models.load_model(mpath)

    print("\nEvaluating on TRAIN set...")
    train_results = model.evaluate(X_train, y_train, verbose=0)
    print("TRAIN metrics:")
    for name, val in zip(model.metrics_names, train_results):
        print(f"  {name}: {val:.4f}")

    print("\nEvaluating on TEST set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print("TEST metrics:")
    for name, val in zip(model.metrics_names, test_results):
        print(f"  {name}: {val:.4f}")

    # Compare
    names = model.metrics_names
    def get(metric, res): return res[names.index(metric)] if metric in names else None
    train_acc, test_acc = get("accuracy", train_results), get("accuracy", test_results)
    train_auc, test_auc = get("auc", train_results), get("auc", test_results)

    print("\n=== Comparison ===")
    if train_acc is not None and test_acc is not None:
        print(f"Accuracy: train={train_acc:.4f}, test={test_acc:.4f}, gap={train_acc - test_acc:+.4f}")
    if train_auc is not None and test_auc is not None:
        print(f"AUC:      train={train_auc:.4f}, test={test_auc:.4f}, gap={train_auc - test_auc:+.4f}")

    # Simple heuristic note
    if (train_acc and test_acc and train_acc - test_acc > 0.05) or (train_auc and test_auc and train_auc - test_auc > 0.05):
        note = "Model likely overfitting (train >> test)."
    else:
        note = "Train/Test are aligned; overfitting not obvious."
    print("\nNote:", note)

if __name__ == "__main__":
    main()
