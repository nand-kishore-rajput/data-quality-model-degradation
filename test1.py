# import pandas as pd
# import numpy as np
# from pathlib import Path

# RAW_DIR = Path("/workspace/datasets/raw")

# # -------------------------------------------------------
# # CHURN — check for multicollinearity in billing features
# # -------------------------------------------------------
# print("=" * 60)
# print("CHURN — billing feature correlation audit")
# print("=" * 60)

# churn = pd.read_parquet(RAW_DIR / "openml" / "churn.parquet")

# billing_pairs = [
#     ("total_day_minutes",   "total_day_charge"),
#     ("total_eve_minutes",   "total_eve_charge"),
#     ("total_night_minutes", "total_night_charge"),
#     ("total_intl_minutes",  "total_intl_charge"),
# ]

# for a, b in billing_pairs:
#     if a in churn.columns and b in churn.columns:
#         # Convert to numeric in case stored as string
#         col_a = pd.to_numeric(churn[a], errors="coerce")
#         col_b = pd.to_numeric(churn[b], errors="coerce")
#         corr  = col_a.corr(col_b)
#         ratio = (col_b / col_a).dropna()
#         print(f"\n  {a} vs {b}")
#         print(f"    Pearson r  : {corr:.6f}")
#         print(f"    Ratio mean : {ratio.mean():.6f}")
#         print(f"    Ratio std  : {ratio.std():.8f}")
#         print(f"    Verdict    : {'!! PERFECT LINEAR — one is computed from the other' if corr > 0.9999 else 'ok'}")

# # -------------------------------------------------------
# # ONLINE SHOPPERS — PageValues leakage audit
# # -------------------------------------------------------
# print("\n" + "=" * 60)
# print("ONLINE_SHOPPERS — PageValues audit")
# print("=" * 60)

# shop = pd.read_parquet(RAW_DIR / "uci" / "online_shoppers.parquet")

# target = "Revenue"
# y = shop[target].map({False: 0, True: 1})

# pv = pd.to_numeric(shop["PageValues"], errors="coerce")

# print(f"\n  PageValues distribution:")
# print(f"    Mean (Revenue=0): {pv[y==0].mean():.4f}")
# print(f"    Mean (Revenue=1): {pv[y==1].mean():.4f}")
# print(f"    % zero when Revenue=0: {(pv[y==0]==0).mean()*100:.1f}%")
# print(f"    % zero when Revenue=1: {(pv[y==1]==0).mean()*100:.1f}%")
# print(f"    Corr with target: {pv.corr(y):.4f}")

# # Feature importance proxy — AUC of PageValues alone
# from sklearn.metrics import roc_auc_score
# pv_clean = pv.fillna(0)
# auc_pv_alone = roc_auc_score(y, pv_clean)
# print(f"\n  AUC of PageValues alone (single feature): {auc_pv_alone:.4f}")
# print(f"  Verdict: {'!! SOFT LEAKAGE — PageValues alone explains most of target AUC' if auc_pv_alone > 0.80 else 'ok — PageValues is predictive but not a proxy'}")


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RAW_DIR = Path("/workspace/datasets/raw")
SEED = 42

XGB = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", tree_method="hist",
    random_state=SEED, n_jobs=-1, verbosity=0,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# --- CHURN re-screen (drop 4 charge cols) ---
churn = pd.read_parquet(RAW_DIR / "openml" / "churn.parquet")
churn = churn.drop(columns=[
    "phone_number",
    "total_day_charge", "total_eve_charge",
    "total_night_charge", "total_intl_charge"
], errors="ignore")
y_churn = churn["class"].astype(int)
X_churn = churn.drop(columns=["class"])
le = LabelEncoder()
for col in X_churn.select_dtypes(include=["object","string","bool"]).columns:
    X_churn[col] = le.fit_transform(X_churn[col].astype(str))
scores = cross_val_score(XGB, X_churn, y_churn, cv=cv, scoring="roc_auc")
print(f"CHURN (charge cols dropped): AUC={np.mean(scores):.4f} ± {np.std(scores):.4f}")

# --- ONLINE_SHOPPERS re-screen (drop PageValues) ---
shop = pd.read_parquet(RAW_DIR / "uci" / "online_shoppers.parquet")
shop = shop.drop(columns=["PageValues"])
y_shop = shop["Revenue"].map({False:0, True:1})
X_shop = shop.drop(columns=["Revenue"])
for col in X_shop.select_dtypes(include=["object","string","bool"]).columns:
    X_shop[col] = le.fit_transform(X_shop[col].astype(str))
scores = cross_val_score(XGB, X_shop, y_shop, cv=cv, scoring="roc_auc")
print(f"ONLINE_SHOPPERS (PageValues dropped): AUC={np.mean(scores):.4f} ± {np.std(scores):.4f}")