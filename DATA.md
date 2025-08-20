# Data Instructions

This project expects a CSV at `data/Indian_IPO_Market_Data.csv` with the following columns:
- Date, IPOName, Issue_Size, Subscription_QIB, Subscription_HNI, Subscription_RII, Subscription_Total, Issue_Price, Listing_Gains_Percent

The pipeline creates a binary target `Listing_Gains_Profit` (1 if `Listing_Gains_Percent` > 0).

> If you cannot share the full dataset publicly, do not commit it. Place it locally in `data/` following the name above. Large files should use Git LFS if versioned.
