import sys, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
D = sys.argv[1]
cv = StratifiedKFold(5, shuffle=True, random_state=0)

print("===== OTT")
o = pd.read_csv(f"{D}/ott/deceptive-opinion.csv")
print(o.shape, o.groupby(["polarity","deceptive"]).size().to_dict())
y = (o.deceptive=="deceptive").astype(int)
X = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True).fit_transform(o.text)
p = cross_val_predict(LogisticRegression(C=10, max_iter=2000), X, y, cv=cv, method="predict_proba")[:,1]
print("Ott TF-IDF LogReg AUC %.3f AP %.3f" % (roc_auc_score(y,p), average_precision_score(y,p)))

print("===== YELPZIP")
m = pd.read_csv(f"{D}/yelpzip/metadata", sep="\t", names=["user","prod","rating","label","date"])
t = pd.read_csv(f"{D}/yelpzip/reviewContent", sep="\t", names=["user","prod","date","text"], quoting=3, on_bad_lines="skip")
print("meta", m.shape, "text", t.shape)
m["fake"] = (m.label==-1).astype(int)
print("fake rate %.3f, users %d, prods %d, dates %s..%s" % (m.fake.mean(), m.user.nunique(), m["prod"].nunique(), m.date.min(), m.date.max()))
print("rating dist by label:\n", pd.crosstab(m.rating, m.fake, normalize="columns").round(3))
uc = m.groupby("user").size(); m["u_cnt"] = m.user.map(uc)
print("fake rate for users with 1 review: %.3f, >=5 reviews: %.3f; share of 1-review users among reviews %.3f" % (m[m.u_cnt==1].fake.mean(), m[m.u_cnt>=5].fake.mean(), (m.u_cnt==1).mean()))
# rating impact
g = m.groupby("prod").agg(n=("rating","size"), all_mean=("rating","mean"), fake_share=("fake","mean"))
g["real_mean"] = m[m.fake==0].groupby("prod").rating.mean()
g["delta"] = g.all_mean - g.real_mean
g = g[g.n>=20]
print("products n>=20: %d; delta (all - real) mean %.3f, median %.3f, |delta|>0.1: %.1f%%, |delta|>0.25: %.1f%%" % (len(g), g.delta.mean(), g.delta.median(), 100*(g.delta.abs()>0.1).mean(), 100*(g.delta.abs()>0.25).mean()))
print("delta>0 (inflated): %.1f%%, delta<0: %.1f%%; corr(fake_share, delta) %.2f" % (100*(g.delta>0).mean(), 100*(g.delta<0).mean(), g[["fake_share","delta"]].corr().iloc[0,1]))
# behavioural probe (note: u_cnt etc. from full data -> mild leakage, OK for probe)
m["date"] = pd.to_datetime(m.date)
pm = m.groupby("prod").rating.mean(); m["dev"] = (m.rating - m["prod"].map(pm)).abs()
m["u_maxday"] = m.groupby(["user","date"]).rating.transform("size").groupby(m.user).transform("max")
m["u_ext"] = m.rating.isin([1,5]).groupby(m.user).transform("mean")
m["u_dev"] = m.groupby("user").dev.transform("mean")
m["u_span"] = (m.groupby("user").date.transform("max")-m.groupby("user").date.transform("min")).dt.days
mt = m.merge(t[["user","prod","date"]].assign(date=lambda d: pd.to_datetime(d.date, errors="coerce"), length=t.text.str.len()), on=["user","prod","date"], how="left").drop_duplicates(["user","prod","date"])
mt["length"] = mt["length"].fillna(0)
feats = ["rating","dev","u_cnt","u_maxday","u_ext","u_dev","u_span","length"]
gk = GroupKFold(5)
p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), mt[feats], mt.fake, cv=gk, groups=mt.user, method="predict_proba")[:,1]
print("YelpZip behavioural+length HGB AUC %.3f AP %.3f (base AP %.3f)" % (roc_auc_score(mt.fake,p), average_precision_score(mt.fake,p), mt.fake.mean()))
s = t.merge(m[["user","prod","date","fake"]].assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")), on=["user","prod","date"]).sample(60000, random_state=0)
X = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=100000, sublinear_tf=True).fit_transform(s.text.fillna(""))
p = cross_val_predict(LogisticRegression(C=2, max_iter=2000, class_weight="balanced"), X, s.fake, cv=cv, method="predict_proba")[:,1]
print("YelpZip text-only TF-IDF (60k sample) AUC %.3f AP %.3f" % (roc_auc_score(s.fake,p), average_precision_score(s.fake,p)))

print("===== YELP CHICAGO (chiragdaryani loaded-reviews.csv)")
c = pd.read_csv(f"{D}/yelpchi_loaded.csv")
print(c.shape, c.flagged.value_counts().to_dict(), "users", c.reviewerID.nunique(), "restaurants", c.restaurantID.nunique())
