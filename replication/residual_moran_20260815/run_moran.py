# -*- coding: utf-8 -*-
"""
重跑：十個候選 k 之 OLS 殘差 Moran's I（主要樣本 n = 153）
邏輯完全沿用 analysis_pipeline/scripts/03_build_weights.py 之 knn_rowstd
與 k4_rerun_20260806/01_reproduce_table43_44.py 之建構方式。
OLS 設定依論文顯 154：影片數、教材數、中階認知比例、高階認知比例 ＋ 年度虛擬變項。
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from libpysal.weights import full2W
import spreg, esda

SEED = 20260626
K_LIST = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
X_COLS = ["video", "materials", "bloom_mid", "bloom_high"]

panel_all = pd.read_csv("analysis_panel.csv").sort_values("panel_idx").reset_index(drop=True)
emb_all = np.load("course_embeddings.npy")

# 主要樣本：data_quality_flag == good（153 筆）
mask = (panel_all["data_quality_flag"] == "good").values
panel = panel_all.loc[mask].reset_index(drop=True)
emb = emb_all[mask]
N = len(panel)
assert N == 153, N


def knn_rowstd(S, k):
    n = S.shape[0]; W = np.zeros_like(S); kk = min(k, n - 1)
    for i in range(n):
        order = np.argsort(-S[i]); nbr = [j for j in order if j != i][:kk]
        W[i, nbr] = S[i, nbr]
    W[W < 0] = 0.0
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1.0
    return W / rs


def build_W(k):
    tid = panel["term_id"].values
    Wb = np.zeros((N, N))
    for t in pd.unique(tid):
        idx = np.where(tid == t)[0]
        E = emb[idx]; S = E @ E.T; np.fill_diagonal(S, 0.0)
        Wb[np.ix_(idx, idx)] = knn_rowstd(S, k)
    return Wb


def zstd(df, cols):
    Z = df[cols].astype(float).copy()
    for c in cols:
        s = Z[c]; sd = s.std(ddof=0)
        Z[c] = (s - s.mean()) / (sd if sd > 0 else 1.0)
    return Z


# 設計矩陣（不隨 k 改變）
D = pd.DataFrame(index=panel.index)
for y in sorted(panel["year"].unique())[1:]:
    D["yr%d" % y] = (panel["year"] == y).astype(float)
X = pd.concat([zstd(panel, X_COLS), D], axis=1)
yv = panel["ln_enroll"].values.reshape(-1, 1)

rows = []
for k in K_LIST:
    Wb = build_W(k)
    w = full2W(Wb); w.transform = "r"
    ols = spreg.OLS(yv, X.values, w=w, spat_diag=True, moran=True)
    resid = np.asarray(ols.u).flatten()
    np.random.seed(SEED)
    m = esda.Moran(resid, w, permutations=999)
    rows.append({
        "k": k,
        "I": round(float(m.I), 4),
        "E_I": round(float(m.EI), 4),
        "z_norm": round(float(m.z_norm), 4),
        "p_norm": round(float(m.p_norm), 4),
        "p_sim": round(float(m.p_sim), 4),
        "p_z_sim": round(float(m.p_z_sim), 4),
        "moran_res_spreg_z": round(float(ols.moran_res[1]), 4),
        "moran_res_spreg_p": round(float(ols.moran_res[2]), 4),
    })
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("residual_moran_by_k_n153.csv", index=False, encoding="utf-8-sig")
print()
print(df.to_string(index=False))
print()
print("I 全距: %.4f ~ %.4f" % (df["I"].min(), df["I"].max()))
print("k=4 :", df.loc[df.k == 4].to_dict("records"))
print("最小 p_norm 之 k :", int(df.loc[df["p_norm"].idxmin(), "k"]), df["p_norm"].min())
print("最小 p_sim  之 k :", int(df.loc[df["p_sim"].idxmin(), "k"]), df["p_sim"].min())
