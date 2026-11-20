"""Run the SPWS machine-learning code (ridge_model.py, the same file used by the web app)
on the dataset the system was built with.

  python run_on_dataset.py                       # uses dataset_spws.csv
  python run_on_dataset.py my_data.csv           # any CSV/XLSX with the same column names

Features: every column of FEAT that exists in the file is used. The stored dataset only contains
midterm, total_lecture and total_section (the quiz columns are not stored), so those 3 are used here.
Three evaluations are reported: in-sample, 80/20 hold-out, and 5-fold cross-validation.
"""
import sys, json
import numpy as np
import pandas as pd
from ridge_model import FEAT, ridge_train, ridge_predict, risk


def load(path):
    return pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)


def fit_eval(Xtr, ytr, Xte, yte, tot_te, mid_i, lec_i, sec_i):
    m = ridge_train(Xtr, ytr, alpha=1.0, lo=0, hi=40)
    pf = ridge_predict(m, Xte)
    pt = np.clip(Xte[:, mid_i] + Xte[:, lec_i] + Xte[:, sec_i] + pf, 0, 100)
    fail, high = tot_te < 50, pt < 50
    tp, fn, fp = int((fail & high).sum()), int((fail & ~high).sum()), int((~fail & high).sum())
    err = yte - pf
    return dict(n=len(yte), MAE=np.abs(err).mean(), RMSE=np.sqrt((err ** 2).mean()),
                R2=1 - (err ** 2).sum() / ((yte - yte.mean()) ** 2).sum(),
                baseline_MAE=np.abs(yte - yte.mean()).mean(),
                risk_acc=100 * (len(yte) - fn - fp) / len(yte),
                fail_recall=100 * tp / (tp + fn) if tp + fn else float("nan"),
                fail_caught_med_or_high=100 * (fail & (pt < 65)).sum() / max(1, fail.sum()))


def show(tag, r):
    print(f"  {tag:<16} n={r['n']:<4} MAE={r['MAE']:.2f} (baseline {r['baseline_MAE']:.2f})  RMSE={r['RMSE']:.2f}  R2={r['R2']:.3f}"
          f"  risk-acc={r['risk_acc']:.1f}%  failing-recall(High)={r['fail_recall']:.1f}%  failing-flagged(Med+High)={r['fail_caught_med_or_high']:.1f}%")


def main(path):
    df = load(path)
    feats = [c for c in FEAT if c in df.columns]
    print("file:", path, "| rows:", len(df), "| features used:", feats)
    for subj, g in df.groupby("subject"):
        X = g[feats].astype(float).values; y = g["final"].astype(float).values; tot = g["final_total"].astype(float).values
        mi, li, si = (feats.index(c) for c in ("midterm", "total_lecture", "total_section"))
        print(f"\n== {subj} ({len(g)} students) ==")
        show("in-sample", fit_eval(X, y, X, y, tot, mi, li, si))
        idx = np.random.default_rng(42).permutation(len(g)); cut = int(len(g) * 0.8); tr, te = idx[:cut], idx[cut:]
        show("hold-out 80/20", fit_eval(X[tr], y[tr], X[te], y[te], tot[te], mi, li, si))
        folds = np.array_split(idx, 5); rs = []
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            rs.append(fit_eval(X[tr], y[tr], X[te], y[te], tot[te], mi, li, si))
        avg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; avg["n"] = len(g)
        show("5-fold CV (avg)", avg)
        m = ridge_train(X, y); print("  coefficients (standardised):", {f: round(float(c), 3) for f, c in zip(feats, m["coef"])}, "intercept:", round(m["intercept"], 2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset_spws.csv")
