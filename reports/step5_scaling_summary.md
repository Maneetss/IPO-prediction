# Step 5 — Scaling Summary

- Built **y** = `Listing_Gains_Profit` and **X** = ['Issue_Size', 'Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Issue_Price'].

- Dropped rows with missing values in X or y prior to scaling.

- Applied **Min–Max scaling** to map each predictor to **[0, 1]**.

- Post-scaling checks (per feature): `min`≈0, `max`≈1 (minor deviations are expected if constants exist).

- Class balance after filtering: **54.5%** positive (target=1).

- Saved: `data/ipo_scaled.csv` (scaled X + y), `X.npy`, `y.npy`.
