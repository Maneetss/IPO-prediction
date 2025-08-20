# Step 6 — Holdout Split Summary
- Train proportion: **0.80** (Test: **0.20**), stratified by `Listing_Gains_Profit`.
- Shapes — Overall: `X=(319, 5)`, `y=(319,)`; Train: `X=(255, 5)`, Test: `X=(64, 5)`.
- Class balance (%% target=1) — Overall: **54.5%**, Train: **54.5%**, Test: **54.7%**.
- Number of predictor features: **5** (expected **5**).
- Artifacts saved: `data/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, and human-readable `data/train.csv`, `data/test.csv`.
