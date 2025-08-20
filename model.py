"""
model.py — Compile & Train a Dense NN for IPO Listing Gain Classification

- Loads train/test arrays created in Step 6 (X_train.npy, y_train.npy, etc.)
- Builds a small MLP (configurable depth/units/activation)
- Compiles with your chosen optimizer, loss, and metrics
- Trains with early stopping + LR reduction
- Prints model.summary() and test metrics

Run:
  python model.py --epochs 60 --optimizer adam --lr 1e-3
  # or try: --optimizer sgd --lr 1e-2
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report

PREDICTORS = ["Issue_Size", "Subscription_QIB", "Subscription_HNI", "Subscription_RII", "Issue_Price"]
TARGET = "Listing_Gains_Profit"

def load_split():
    data = Path("data")
    Xtr_p, Xte_p = data/"X_train.npy", data/"X_test.npy"
    ytr_p, yte_p = data/"y_train.npy", data/"y_test.npy"
    if all(p.exists() for p in [Xtr_p, Xte_p, ytr_p, yte_p]):
        X_train, X_test = np.load(Xtr_p), np.load(Xte_p)
        y_train, y_test = np.load(ytr_p), np.load(yte_p)
        return X_train, X_test, y_train, y_test
    # fallback to CSVs if npy not found
    train_csv, test_csv = data/"train.csv", data/"test.csv"
    if train_csv.exists() and test_csv.exists():
        tr, te = pd.read_csv(train_csv), pd.read_csv(test_csv)
        X_train, y_train = tr[PREDICTORS].values, tr[TARGET].values.astype(int)
        X_test,  y_test  = te[PREDICTORS].values, te[TARGET].values.astype(int)
        return X_train, X_test, y_train, y_test
    sys.exit("[ERROR] Missing split files. Run the holdout split step first.")

def build_model(input_dim:int, depth:int, units:int, activation:str,
                dropout:float, l2:float, batchnorm:bool) -> keras.Model:
    reg = regularizers.l2(l2) if l2 > 0 else None
    m = keras.Sequential(name="ipo_mlp")
    m.add(layers.Input(shape=(input_dim,)))
    for i in range(depth):
        m.add(layers.Dense(units, kernel_regularizer=reg, name=f"dense_{i+1}"))
        if batchnorm:
            m.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        m.add(layers.Activation(activation, name=f"act_{i+1}"))
        if dropout > 0:
            m.add(layers.Dropout(dropout, name=f"drop_{i+1}"))
    m.add(layers.Dense(1, activation="sigmoid", name="output"))  # binary classification
    return m

def make_optimizer(name:str, lr:float):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    sys.exit(f"[ERROR] Unsupported optimizer: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--optimizer", type=str, default="adam", help="adam | sgd | rmsprop")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--depth", type=int, default=2, choices=[1,2,3,4])
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--activation", type=str, default="relu")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--batchnorm", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    X_train, X_test, y_train, y_test = load_split()
    input_dim = X_train.shape[1]

    # 1) Instantiate model
    model = build_model(
        input_dim=input_dim,
        depth=args.depth,
        units=args.units,
        activation=args.activation,
        dropout=args.dropout,
        l2=args.l2,
        batchnorm=args.batchnorm,
    )

    # 2) Compile: optimizer + loss + metrics (+ learning rate where applicable)
    opt = make_optimizer(args.optimizer, args.lr)
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

    # 3) Print summary
    print("\nModel summary:")
    model.summary()

    # 4) Fit/train
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        callbacks=callbacks,
        verbose=1,
    )

    # 5) Evaluate on test set
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f"Test {name}: {val:.4f}")

    # Optional: quick classification report at 0.5 threshold
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    main()
