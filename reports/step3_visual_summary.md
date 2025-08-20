# Step 3 — Visualization Summary

- **Class balance:** ~54.5% of IPOs listed at a profit (target=1).

- **Feature vs target (class-wise means):**
  - Issue_Size: mean(1)=1203.11, mean(0)=1180.56, Δ=22.55
  - Subscription_QIB: mean(1)=37.69, mean(0)=11.28, Δ=26.41
  - Subscription_HNI: mean(1)=105.39, mean(0)=27.73, Δ=77.66
  - Subscription_RII: mean(1)=11.75, mean(0)=4.74, Δ=7.01
  - Issue_Price: mean(1)=387.89, mean(0)=359.82, Δ=28.06

- **Strong predictor correlations (|r| ≥ 0.70):**
  - Subscription_QIB vs Subscription_HNI: r=0.77

In short: the **count plot** shows overall class balance; **boxplots** reveal potential outliers in the numeric predictors; **overlaid histograms** suggest how each predictor’s distribution differs between profit and non-profit classes; and the **correlation heatmap** highlights redundant signals you might drop before modeling.
