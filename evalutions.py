"""
ipo_step8_evaluate.py — Evaluate trained model on train & test and summarize.
- Loads arrays from data/ (X_train.npy, X_test.npy, y_train.npy, y_test.npy)
- Loads model from models/ipo_mlp.keras by default (or evaluates the last in-memory model if run after training)
"""

from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

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
    y_prob_tr = model.predict(X_train, verbose=0).ravel()
    y_pred_tr = (y_prob_tr >= 0.5).astype(int)
    train_acc = accuracy_score(y_train, y_pred_tr)
    try:
        train_auc = roc_auc_score(y_train, y_prob_tr)
    except Exception:
        train_auc = float("nan")
    print(f"TRAIN Accuracy: {train_acc:.4f}")
    print(f"TRAIN AUC:      {train_auc:.4f}")

    print("\nEvaluating on TEST set...")
    y_prob_te = model.predict(X_test, verbose=0).ravel()
    y_pred_te = (y_prob_te >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_pred_te)
    try:
        test_auc = roc_auc_score(y_test, y_prob_te)
    except Exception:
        test_auc = float("nan")
    print(f"TEST Accuracy:  {test_acc:.4f}")
    print(f"TEST AUC:       {test_auc:.4f}")

    print("\n=== Comparison ===")
    print(f"Accuracy gap (train - test): {train_acc - test_acc:+.4f}")
    print(f"AUC gap      (train - test): {train_auc - test_auc:+.4f}")
    # Simple heuristic note
    if (train_acc and test_acc and train_acc - test_acc > 0.05) or (train_auc and test_auc and train_auc - test_auc > 0.05):
        note = "Model likely overfitting (train >> test)."
    else:
        note = "Train/Test are aligned; overfitting not obvious."
    print("\nNote:", note)

if __name__ == "__main__":
    main()
