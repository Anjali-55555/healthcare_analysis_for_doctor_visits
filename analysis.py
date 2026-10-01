"""Healthcare Analytics for Doctor Visits_TIRTC - analysis script."""
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

df = pd.read_csv("data/doctor_visits.csv").drop(columns="Unnamed: 0")
df["any_visit"] = (df.visits > 0).astype(int)
bin_cols = ["private", "freepoor", "freerepat", "nchronic", "lchronic"]
X = df.drop(columns=["visits", "any_visit"]).copy()
X["gender"] = (X.gender == "female").astype(int)
for c in bin_cols: X[c] = (X[c] == "yes").astype(int)
y = df.any_visit
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)

res = {}
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr)
rf = RandomForestClassifier(300, max_depth=6, class_weight="balanced", random_state=42).fit(Xtr, ytr)
for n, m in [("Logistic Regression", lr), ("Random Forest", rf)]:
    p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
    res[n] = dict(acc=accuracy_score(yte, p), auc=roc_auc_score(yte, pr), f1=f1_score(yte, p))
coef = pd.Series(lr[-1].coef_[0], index=X.columns).sort_values()
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
res["coef"] = coef.round(3).to_dict(); res["imp"] = imp.round(3).to_dict()
pois = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=500)).fit(Xtr, df.visits[Xtr.index])
res["poisson_mean_pred"] = float(pois.predict(Xte).mean())
json.dump(res, open("outputs/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C = "#1F6FA8"
fig, ax = plt.subplots(figsize=(6, 3.6)); df.visits.value_counts().sort_index().plot.bar(ax=ax, color=C)
ax.set(title="Doctor visits in past 2 weeks", xlabel="Visits", ylabel="Patients"); plt.tight_layout(); plt.savefig("outputs/visits_dist.png", dpi=200)
fig, ax = plt.subplots(figsize=(6, 3.6)); coef.plot.barh(ax=ax, color=np.where(coef > 0, C, "#C0504D"))
ax.set(title="Logistic regression coefficients (standardised)"); plt.tight_layout(); plt.savefig("outputs/coef.png", dpi=200)
