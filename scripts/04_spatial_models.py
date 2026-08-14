# -*- coding: utf-8 -*-
"""
04 spatial econometrics: Moran's I, LM tests, SAR/SEM/SDM + effect decomposition
Main spec: y=ln_enroll, X=[video,materials] (z-std) + year FE, W=k10 (within-year blocks).
Effect decomposition (direct/indirect/total) via LeSage-Pace with simulation inference.
NOTE: Bloom difficulty + credit/fee are reserved-empty; when filled, add to BASE_X and rerun.
"""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from libpysal.weights import full2W
import spreg, esda
import config as C

OUT = C.OUT
# A4：Bloom（低階為參照組）+ 學分數估算已到位，全部納入主模型
BASE_X = ["video", "materials", "bloom_mid", "bloom_high", "credit_est"]

def zstd(df, cols):
    Z = df[cols].copy()
    for c in cols:
        s = Z[c]; Z[c] = (s - s.mean()) / s.std(ddof=0)
    return Z

def build_w(k):
    Wb = np.load(os.path.join(OUT, "W_k%d_block.npy" % k))
    w = full2W(Wb); w.transform = "r"
    return Wb, w

def year_dummies(panel):
    yrs = sorted(panel["year"].unique())
    D = pd.DataFrame(index=panel.index)
    for y in yrs[1:]:                       # drop first year as base
        D["yr%d" % y] = (panel["year"] == y).astype(float)
    return D

def sdm_impacts(Wb, betas, vm, names_x, target_vars, R=2000, seed=C.SEED):
    """betas = [const, x..., rho] (flat); compute direct/indirect/total per target var."""
    n = Wb.shape[0]; I = np.eye(n)
    rho_idx = len(betas) - 1
    col = {nm: i + 1 for i, nm in enumerate(names_x)}   # +1 for CONST at index 0
    def impacts_from(vec):
        rho = vec[rho_idx]; A = np.linalg.inv(I - rho * Wb); res = {}
        for v in target_vars:
            b = vec[col[v]]; th = vec[col["W_" + v]]
            M = A @ (b * I + th * Wb)
            d = np.trace(M) / n; tot = M.sum() / n
            res[v] = (d, tot - d, tot)
        return res
    pt = impacts_from(betas)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(betas, vm, size=R)
    sim = {v: {"direct": [], "indirect": [], "total": []} for v in target_vars}
    for dvec in draws:
        r = impacts_from(dvec)
        for v in target_vars:
            sim[v]["direct"].append(r[v][0]); sim[v]["indirect"].append(r[v][1]); sim[v]["total"].append(r[v][2])
    recs = []
    for v in target_vars:
        for eff, pe in zip(["direct", "indirect", "total"], pt[v]):
            arr = np.array(sim[v][eff]); se = arr.std(ddof=1)
            p = 2 * min((arr > 0).mean(), (arr < 0).mean())
            recs.append({"variable": v, "effect": eff, "estimate": round(pe, 4),
                         "sim_se": round(se, 4), "p_value": round(float(p), 4),
                         "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
                         "ci_hi": round(float(np.percentile(arr, 97.5)), 4)})
    return pd.DataFrame(recs)

def main():
    panel = pd.read_csv(os.path.join(OUT, "analysis_panel.csv")).sort_values("panel_idx").reset_index(drop=True)
    Wb, w = build_w(C.KNN_MAIN)
    y = panel["ln_enroll"].values.reshape(-1, 1)

    # ---------- Moran's I on ln_enroll ----------
    moran_recs = []
    mo = esda.Moran(panel["ln_enroll"].values, w, permutations=999)
    moran_recs.append({"scope": "overall_k%d" % C.KNN_MAIN, "n": len(panel),
                       "Moran_I": round(mo.I, 4), "E_I": round(mo.EI, 4),
                       "z_sim": round(mo.z_sim, 3), "p_sim": round(mo.p_sim, 4)})
    for yr in sorted(panel["year"].unique()):
        idx = panel.index[panel["year"] == yr].values
        sub = panel.loc[idx]
        Wsub = Wb[np.ix_(idx, idx)]
        rs = Wsub.sum(1, keepdims=True); rs[rs == 0] = 1; Wsub = Wsub / rs
        wsub = full2W(Wsub); wsub.transform = "r"
        moy = esda.Moran(sub["ln_enroll"].values, wsub, permutations=999)
        moran_recs.append({"scope": "year_%d" % yr, "n": len(sub),
                           "Moran_I": round(moy.I, 4), "E_I": round(moy.EI, 4),
                           "z_sim": round(moy.z_sim, 3), "p_sim": round(moy.p_sim, 4)})
    moran = pd.DataFrame(moran_recs)
    moran.to_csv(os.path.join(OUT, "moran_results.csv"), index=False, encoding="utf-8-sig")

    # ---------- design ----------
    D = year_dummies(panel)
    Xz = zstd(panel, BASE_X)
    Xbase = pd.concat([Xz, D], axis=1)
    base_names = list(Xbase.columns)

    # WX (lag only the substantive covariates, not dummies/const)
    WXz = pd.DataFrame(Wb @ Xz.values, columns=["W_" + c for c in BASE_X], index=panel.index)
    Xsdm = pd.concat([Xbase, WXz], axis=1)
    sdm_names = list(Xsdm.columns)

    # ---------- OLS + LM tests ----------
    ols = spreg.OLS(y, Xbase.values, w=w, spat_diag=True, moran=True, name_y="ln_enroll", name_x=base_names)
    lm = {"moran_resid_I": float(ols.moran_res[0]), "moran_resid_p": float(ols.moran_res[2]),
          "LM_lag": float(ols.lm_lag[0]), "LM_lag_p": float(ols.lm_lag[1]),
          "LM_error": float(ols.lm_error[0]), "LM_error_p": float(ols.lm_error[1]),
          "RLM_lag": float(ols.rlm_lag[0]), "RLM_lag_p": float(ols.rlm_lag[1]),
          "RLM_error": float(ols.rlm_error[0]), "RLM_error_p": float(ols.rlm_error[1]),
          "r2": float(ols.r2)}
    with open(os.path.join(OUT, "lm_tests.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)

    # ---------- SAR / SEM / SDM ----------
    sar = spreg.ML_Lag(y, Xbase.values, w=w, name_y="ln_enroll", name_x=base_names)
    sem = spreg.ML_Error(y, Xbase.values, w=w, name_y="ln_enroll", name_x=base_names)
    sdm = spreg.ML_Lag(y, Xsdm.values, w=w, name_y="ln_enroll", name_x=sdm_names)

    def betas_dict(m, names):
        full = ["CONSTANT"] + names + (["rho"] if hasattr(m, "rho") else (["lambda"] if hasattr(m, "lam") else []))
        b = m.betas.flatten()
        return {full[i]: round(float(b[i]), 4) for i in range(len(b))}

    comp = []
    for nm, m, names, sp in [("OLS", ols, base_names, None),
                             ("SAR", sar, base_names, "rho"),
                             ("SEM", sem, base_names, "lambda"),
                             ("SDM", sdm, sdm_names, "rho")]:
        row = {"model": nm, "k_neighbors": C.KNN_MAIN, "n": len(panel),
               "loglik": round(float(getattr(m, "logll", np.nan)), 2) if hasattr(m, "logll") else np.nan,
               "aic": round(float(getattr(m, "aic", np.nan)), 2) if hasattr(m, "aic") else np.nan,
               "pseudo_r2": round(float(getattr(m, "pr2", getattr(m, "r2", np.nan))), 4)}
        if sp == "rho": row["spatial_rho"] = round(float(m.rho), 4)
        if sp == "lambda": row["spatial_lambda"] = round(float(m.betas.flatten()[-1]), 4)
        bd = betas_dict(m, names)
        row["beta_video"] = bd.get("video"); row["beta_materials"] = bd.get("materials")
        row["theta_W_video"] = bd.get("W_video"); row["theta_W_materials"] = bd.get("W_materials")
        comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False, encoding="utf-8-sig")

    # ---------- SDM impacts ----------
    imp = sdm_impacts(Wb, sdm.betas.flatten(), sdm.vm, sdm_names, BASE_X)
    imp.insert(0, "k_neighbors", C.KNN_MAIN)
    imp.to_csv(os.path.join(OUT, "sdm_impacts.csv"), index=False, encoding="utf-8-sig")

    # ---------- SDM->SAR Wald (theta jointly = 0) ----------
    names_full = ["CONSTANT"] + sdm_names + ["rho"]
    th_idx = [names_full.index("W_" + v) for v in BASE_X]
    th = sdm.betas.flatten()[th_idx]
    Vth = sdm.vm[np.ix_(th_idx, th_idx)]
    wald = float(th.T @ np.linalg.inv(Vth) @ th)
    from scipy.stats import chi2
    wald_p = float(chi2.sf(wald, len(th_idx)))

    print("=== Moran's I (ln_enroll) ===")
    print(moran.to_string(index=False))
    print("\n=== LM tests (OLS resid, with year FE) ===")
    print(json.dumps(lm, ensure_ascii=False, indent=2))
    print("\n=== model comparison ===")
    print(comp.to_string(index=False))
    print("\n=== SDM impacts (z-std X; in log-enrollment units per 1 SD) ===")
    print(imp.to_string(index=False))
    print("\n=== SDM->SAR Wald (H0: all theta=0): W=%.3f, df=%d, p=%.4f ===" % (wald, len(th_idx), wald_p))

if __name__ == "__main__":
    main()
