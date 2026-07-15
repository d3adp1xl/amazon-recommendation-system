# Amazon Product Recommendation System

Building and comparing recommender systems on **Amazon Electronics** ratings — from a simple popularity baseline through memory-based collaborative filtering to matrix factorization. The tuned **SVD model won with an RMSE of 0.8808**, the lowest error across every approach, while also handling data sparsity that broke the KNN models.

> **TL;DR:** Progressed through four recommender families and measured each one honestly (RMSE + precision@k / recall@k / F1@k). Matrix factorization (SVD) beat memory-based KNN not just on error but on *robustness* — it produced personalized predictions for user-item pairs where KNN had no neighbors to fall back on.

**Stack:** Python · [Surprise](http://surpriselib.com/) · scikit-learn · pandas · NumPy · Matplotlib / Seaborn · Google Colab

---

## The Problem

Online catalogs have millions of items and each user sees a tiny fraction. A recommender's job is to predict what a given user would rate highly and surface it. Here the task is to predict a user's rating (1–5) for Amazon electronics products they haven't seen, and to return the top-N products worth recommending to each user.

## The Data

Amazon electronics product reviews (user, product, rating, timestamp).

| Property | Value |
|---|---|
| Original size | 7,824,482 ratings |
| After filtering | 65,290 ratings (users with ≥ 50 ratings) |
| Unique users | 1,540 |
| Unique products | 5,689 |
| Rating scale | 1–5 (mean ≈ 4.29) |

Two facts shaped everything downstream:

- **Extreme sparsity.** 65,290 ratings out of a possible ~8.76M user-product cells — roughly **99.3% empty**. Most users and items barely overlap.
- **Heavy positive skew.** About 54% of all ratings are 5-star and ~82% are 4 or 5. Users mostly rate things they already liked, so the model has to work against a strong optimism bias.

## Approach

Four model families, each evaluated on a held-out test set. Ranking quality was measured with **precision@k, recall@k, and F1@k** (threshold = 3.5, k = 10); rating accuracy with **RMSE**.

1. **Rank-based (popularity)** — non-personalized baseline; top products by average rating with minimum-interaction thresholds (50 and 100) to avoid recommending items with too few reviews.
2. **User-user collaborative filtering** — KNNBasic (cosine / MSD similarity), baseline then GridSearchCV-tuned.
3. **Item-item collaborative filtering** — KNNBasic on item similarity, baseline then tuned.
4. **Matrix factorization (SVD)** — model-based CF learning latent user/item factors, baseline then tuned (n_epochs, lr_all, reg_all).

## Results

| Model | RMSE | Precision@10 | Recall@10 | F1@10 |
|---|:---:|:---:|:---:|:---:|
| **SVD — tuned** | **0.8808** | **0.855** | 0.877 | 0.866 |
| SVD — baseline | 0.8882 | — | 0.880 | 0.866 |
| Item-item KNN — tuned | 0.9615 | 0.835 | 0.878 | 0.856 |
| Item-item KNN — baseline | 0.9950 | 0.838 | 0.845 | 0.841 |
| User-user KNN — tuned | ~1.00 | 0.852 | 0.889 | ~0.87 |
| User-user KNN — baseline | ~1.00 | 0.855 | 0.858 | 0.856 |
| Rank-based | n/a | n/a | n/a | n/a |

*Lower RMSE is better; higher precision/recall/F1 is better. Rank-based popularity is non-personalized, so per-user ranking metrics don't apply.*

RMSE improved steadily as the models grew more sophisticated: user-user (~1.00) → item-item tuned (0.9615) → **SVD tuned (0.8808)**.

## The Key Insight: Sparsity Breaks Neighborhoods, Not Latent Factors

The most instructive result wasn't in the summary metrics — it showed up when predicting individual ratings.

For a user–product pair with few or no similar neighbors, the KNN models returned `was_impossible = True` and fell back to a global average, meaning **no real personalization**. On the exact same pair, SVD produced a confident, personalized estimate (e.g. ~4.08/5), because it represents every user and item as a dense vector of latent factors rather than depending on overlapping co-ratings.

**In a catalog that's ~99% empty, that robustness matters more than a few points of RMSE.** Matrix factorization degrades gracefully where memory-based methods simply fail — the kind of behavior that decides whether a recommender is usable in production, not just on a benchmark.

## What Tuning Bought

- **Item-item KNN:** GridSearchCV cut RMSE from 0.9950 → 0.9615 and lifted recall from 0.845 → 0.878.
- **User-user KNN:** tuning pushed recall from 0.858 → 0.889 (the highest recall of any model).
- **SVD:** tuning n_epochs / lr_all / reg_all edged RMSE to 0.8808 (best overall) and precision to 0.855 (best overall).

## Conclusion & Recommendations

- **Deploy tuned SVD as the core recommender** — best rating accuracy, best precision, and graceful handling of sparse users/items.
- **Keep the rank-based model as a cold-start fallback** for brand-new users with no history.
- The positive rating skew means precision/recall should be read alongside RMSE — a model can look "accurate" just by predicting high ratings, so ranking metrics are the honest check.
- Next steps worth exploring: hybrid models (blend content features with collaborative signals) and richer implicit feedback beyond explicit star ratings.

## Repository

```
├── README.md
├── notebook/
│   └── amazon_recommendation_system.ipynb   # EDA → rank-based → user/item KNN → SVD → tuning
└── assets/                                   # rating distribution, model comparison charts
```

The notebook runs top-to-bottom on Google Colab (the Surprise library installs cleanly there).

---

*Recommendation systems project — MIT Applied AI & Data Science. Built by Fuad Ahmadov.*
