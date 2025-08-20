from pathlib import Path
import datetime as dt
import textwrap

ROOT = Path(".")
R = ROOT / "reports"
D = ROOT / "data"
M = ROOT / "models"

# Known artifacts we’ll try to include if present
FIGS = {
    "Target Count":        R / "target_count.png",
    "Correlation Heatmap": R / "corr_heatmap.png",
    "Training Loss":       R / "training_loss.png",
    "Validation AUC":      R / "training_auc.png",
    "Confusion Matrix":    R / "confusion_matrix.png",
    "ROC Curve":           R / "roc_curve.png",
}
STEP_MD = [
    R / "step3_visual_summary.md",
    R / "step4_outliers_summary.md",
    R / "step5_scaling_summary.md",
    R / "step6_split_summary.md",
    R / "step7_model_summary.md",
]

PREDICTORS = ["Issue_Size","Subscription_QIB","Subscription_HNI","Subscription_RII","Issue_Price"]
TARGET = "Listing_Gains_Profit"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""

def safe_md_image(path: Path, title: str, width_pct=48) -> str:
    if not path.exists():
        return f"> _(Missing figure: {path.name})_\n"
    # HTML img so we can control width on GitHub
    return f'<p><img src="{path.as_posix()}" alt="{title}" width="{width_pct}%"/></p>\n'

def model_summary_text() -> str:
    """Capture keras model.summary() if possible; otherwise return step7 summary."""
    out = ""
    try:
        import tensorflow as tf
        mp = M / "ipo_mlp.keras"
        if mp.exists():
            model = tf.keras.models.load_model(mp, compile=False)
            lines = []
            model.summary(print_fn=lambda s: lines.append(s))
            out = "```\n" + "\n".join(lines) + "\n```"
    except Exception as e:
        out = f"> _(Could not load model/summary: {e})_\n"
    return out

def latest_metrics_text() -> str:
    """Compute test Accuracy/AUC if arrays + model are present. Silent fallback."""
    try:
        import numpy as np, tensorflow as tf
        from sklearn.metrics import accuracy_score, roc_auc_score
        Xt, yt = (D/"X_test.npy"), (D/"y_test.npy")
        mp = M/"ipo_mlp.keras"
        if Xt.exists() and yt.exists() and mp.exists():
            Xt, yt = np.load(Xt), np.load(yt)
            model = tf.keras.models.load_model(mp, compile=False)
            y_prob = model.predict(Xt, verbose=0).ravel()
            acc = float(((y_prob >= 0.5).astype(int) == yt).mean())
            try:
                auc = float(roc_auc_score(yt, y_prob))
            except Exception:
                auc = float("nan")
            return f"**Latest metrics** — Test Accuracy **{acc:.4f}**, Test AUC **{auc:.4f}**\n"
    except Exception:
        pass
    # Fallback to step7 summary if exists
    s7 = read_text(R/"step7_model_summary.md")
    return s7 if s7 else "> _(Train a model and/or run evaluation to populate metrics.)_\n"

def main():
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# IPO Listing Gains (India) — Full Project Report\n")
    lines.append(f"_Generated: {ts}_\n")

    # TL;DR
    lines.append("## TL;DR\n")
    lines.append("- Binary classification: `Listing_Gains_Profit` = 1 if listing gain > 0%, else 0.\n")
    lines.append(f"- Predictors used: `{', '.join(PREDICTORS)}`.\n")
    lines.append("- Model: Dense Neural Network (Keras Sequential) with sigmoid output.\n")
    lines.append("- Scaling: Min–Max to [0,1]; Split: stratified train/test.\n")
    lines.append(latest_metrics_text() + "\n")

    # Key figures
    lines.append("## Key Figures\n")
    row = []
    for title, fig in FIGS.items():
        row.append(safe_md_image(fig, title, width_pct=48))
        if len(row) == 2:
            lines.extend(row); row = []
    if row: lines.extend(row)
    lines.append("\n")

    # Pipeline overview
    lines.append("## Pipeline Overview\n")
    lines.append(textwrap.dedent(f"""
    1. **EDA & Target** — Created `Listing_Gains_Profit` from `Listing_Gains_Percent`, checked missingness, selected features.
    2. **Visualizations** — Count plot, boxplots + histograms by class, correlation heatmap.
    3. **Outliers** — IQR bounds; winsorization applied to flagged features.
    4. **Scaling** — Min–Max scaling of predictors to [0,1].
    5. **Holdout Split** — Stratified train/test split.
    6. **Model** — Dense MLP (1–4 hidden layers configurable), sigmoid output; loss=Binary CE; metrics=Accuracy/AUC.
    7. **Training** — EarlyStopping (val AUC) + ReduceLROnPlateau; validation split from training.
    8. **Evaluation** — Train vs Test metrics; confusion matrix + ROC.
    """).strip() + "\n\n")

    # Include step summaries if available
    for md in STEP_MD:
        if md.exists():
            lines.append(f"## {md.stem.replace('_',' ').title()}\n")
            lines.append(read_text(md) + "\n\n")

    # Model architecture (captured)
    lines.append("## Model Architecture\n")
    ms = model_summary_text()
    if ms.strip():
        lines.append(ms + "\n")
    else:
        lines.append("> _(Model summary not available.)_\n\n")

    # Reproduction
    lines.append("## Reproducibility\n")
    lines.append(textwrap.dedent("""
    ```bash
    # create & activate venv
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    # (optional) place real data:
    # cp <your_csv> data/Indian_IPO_Market_Data.csv

    # pipeline
    python ipo_step2.py
    python ipo_step3_viz.py
    python ipo_step4_outliers.py
    python ipo_step5_scale.py
    python ipo_step6_split.py
    python model.py --epochs 60 --optimizer adam --lr 1e-3
    python ipo_step8_evaluate.py
    ```
    """).strip() + "\n")

    # Save
    out = ROOT / "PROJECT_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out.resolve()}")
    if (R/"gallery_all.png").exists():
        print("Tip: Embed the full gallery with:  ![](reports/gallery_all.png)")
    else:
        print("Tip: You can generate a gallery with the earlier gallery_all.png script.")
if __name__ == "__main__":
    main()
