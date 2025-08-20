# Step 4 — Outlier Strategy & Rationale

We identified outliers using the **IQR rule** (lower bound = Q1 − 1.5×IQR, upper bound = Q3 + 1.5×IQR) on each numeric predictor. A variable was **selected for treatment** if either its absolute skewness exceeded **1.0** or more than **5.0%** of rows fell outside the IQR bounds.

**Variables treated (winsorized to the IQR bounds):**
- Issue_Size: skew=4.85, outliers=34 (10.7%), bounds=[-1.23e+03, 2.5e+03], values capped=34
- Subscription_QIB: skew=2.14, outliers=35 (11.0%), bounds=[-49.1, 84.9], values capped=35
- Subscription_HNI: skew=3.08, outliers=48 (15.0%), bounds=[-90, 153], values capped=48
- Subscription_RII: skew=3.71, outliers=33 (10.3%), bounds=[-9.72, 19.6], values capped=33
- Issue_Price: skew=1.70, outliers=12 (3.8%), bounds=[-506, 1.16e+03], values capped=12

The post-treatment dataset was saved to **data/ipo_clean_winsor.csv**.

**Rationale:** Winsorization preserves the rank order and reduces the influence of extreme tails on model training without discarding data. If outliers are genuine signals (e.g., very large issues or exceptional subscription spikes), keeping them bounded avoids instability while retaining their relative magnitude. If your model is tree-based, you may choose to skip winsorization; for linear/DNN models, capping can improve training stability.