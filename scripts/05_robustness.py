# -*- coding: utf-8 -*-
"""
05 robustness checks for the Panel SDM (year FE)
Vary: k (5 vs 10), DV (ln vs raw), sample (all vs good-only), FE (year / +field / +school), X set.
Report rho, AIC, and direct/indirect/total impacts (+p) for the focal covariates.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from libpysal.weights import full2W
import spreg
import config as C
OUT = C.OUT

def zstd(df, cols):
    Z = df[cols].astype(float).copy()
    for c in cols:
        s = Z[c]; sd = s.std(ddof=0); Z[c] = (s - s.mean()) / (sd if sd > 0 else 1.0)
    return Z

def rowstd(W):
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1.0; return W / rs

def run_sdm(panel, Wb, base_x, y_col="ln_enroll", add_field=False, add_school=False, R=1000, seed=C.SEED):
    panel = panel.reset_index(drop=True)
    y = panel[y_col].values.reshape(-1, 1)
    Xz = zstd(panel, base_x)
    parts = [Xz]
    yrs = sorted(panel["year"].unique())
    for yv in yrs[1:]:
        parts.append(pd.DataFrame({"yr%d" % yv: (panel["year"] == yv).astype(float)}))
    if add_field:
        fd = pd.get_dummies(panel["field_category"], prefix="fld", drop_first=True).astype(float)
        parts.append(fd.reset_index(drop=True))
    if add_school:
        sd = pd.get_dummies(panel["school_fe_id"].astype(str), prefix="sch", drop_first=True).astype(float)
        parts.append(sd.reset_index(drop=True))
    WX = pd.DataFrame(Wb @ Xz.values, columns=["W_" + c for c in base_x])
    X = pd.concat([p.reset_index(drop=True) for p in parts] + [WX], axis=1)
    names = list(X.columns)
    w = full2W(Wb); w.transform = "r"
    m = spreg.ML_Lag(y, X.values, w=w, name_y=y_col, name_x=names)

    betas = m.betas.flatten(); vm = m.vm
    full = ["CONSTANT"] + names + ["rho"]; rho_idx = len(betas) - 1
    col = {nm: full.index(nm) for nm in full}
    n = Wb.shape[0]; I = np.eye(n)
    def imp(vec):
        A = np.linalg.inv(I - vec[rho_idx] * Wb); r = {}
        for v in base_x:
            M = A @ (vec[col[v]] * I + vec[col["W_" + v]] * Wb)
            d = np.trace(M) / n; t = M.sum() / n; r[v] = (d, t - d, t)
        return r
    pt = imp(betas)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(betas, vm, size=R)
    sims = {v: {"direct": [], "indirect": [], "total": []} for v in base_x}
    for dv in draws:
        r = imp(dv)
        for v in base_x:
            sims[v]["direct"].append(r[v][0]); sims[v]["indirect"].append(r[v][1]); sims[v]["total"].append(r[v][2])
    out = {"rho": round(float(m.rho), 4), "aic": round(float(m.aic), 2), "n": n,
           "pseudo_r2": round(float(getattr(m, "pr2", np.nan)), 4)}
    for v in base_x:
        for eff in ["direct", "indirect", "total"]:
            arr = np.array(sims[v][eff]); p = 2 * min((arr > 0).mean(), (arr < 0).mean())
            out["%s_%s" % (v, eff)] = round(pt[v][["direct", "indirect", "total"].index(eff)], 4)
            out["%s_%s_p" % (v, eff)] = round(float(p), 4)
    return out

def main():
    panel = pd.read_csv(os.path.join(OUT, "analysis_panel.csv")).sort_values("panel_idx").reset_index(drop=True)
    W10 = np.load(os.path.join(OUT, "W_k10_block.npy"))
    W5  = np.load(os.path.join(OUT, "W_k5_block.npy"))
    focal = ["video", "materials", "bloom_mid", "bloom_high", "credit_est"]

    configs = []
    configs.append(("M0 main: k10, ln, all175, yearFE", dict(panel=panel, Wb=W10, base_x=focal)))
    configs.append(("M1 k5",                              dict(panel=panel, Wb=W5,  base_x=focal)))
    configs.append(("M2 raw DV (enroll)",                 dict(panel=panel, Wb=W10, base_x=focal, y_col="enroll")))
    configs.append(("M4 +field FE",                       dict(panel=panel, Wb=W10, base_x=focal, add_field=True)))
    configs.append(("M5 +school FE",                      dict(panel=panel, Wb=W10, base_x=focal, add_school=True)))
    configs.append(("M6 extended X (5 covars)",           dict(panel=panel, Wb=W10, base_x=["video","materials","hours","instructors","quiz"])))

    # good-only subsample: subset W and re-rowstd within blocks
    good = panel["data_quality_flag"].eq("good").values
    pg = panel[good].reset_index(drop=True)
    Wg = rowstd(W10[np.ix_(np.where(good)[0], np.where(good)[0])].copy())
    configs.append(("M3 good-only (n=%d)" % good.sum(), dict(panel=pg, Wb=Wg, base_x=focal)))

    rows = []
    for label, kw in configs:
        try:
            r = run_sdm(**kw); r = {"config": label, **r}
        except Exception as e:
            r = {"config": label, "rho": "ERR:%s" % type(e).__name__}
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "robustness_results.csv"), index=False, encoding="utf-8-sig")
    show = ["config","n","rho","aic","video_total","video_total_p","video_indirect","video_indirect_p",
            "materials_total","materials_total_p"]
    print(df[show].to_string(index=False))

if __name__ == "__main__":
    main()
