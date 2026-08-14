# -*- coding: utf-8 -*-
"""
03 build per-term (per-year) semantic spatial weight matrices W
- cosine similarity within each term (year), diagonal = 0
- kNN filter (k in config), similarity-weighted, then row-standardized
- assemble 175x175 block-diagonal W (no cross-term links)
Outputs: W_k{k}_block.npy (row-standardized dense), W_cosine_block.npy (raw cosine),
         weights_summary.csv, neighbors_k{k}.csv
Extensible: grouping is by term_id, so adding spring/fall terms just adds blocks.
"""
import os, numpy as np, pandas as pd
import config as C

def knn_rowstd(S, k):
    """similarity-weighted kNN + row standardization on a single block S (n x n)."""
    n = S.shape[0]
    W = np.zeros_like(S)
    kk = min(k, n - 1)
    for i in range(n):
        order = np.argsort(-S[i])           # descending similarity
        nbr = [j for j in order if j != i][:kk]
        W[i, nbr] = S[i, nbr]
    W[W < 0] = 0.0                          # negative cosine -> no link
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs

def main():
    panel = pd.read_csv(os.path.join(C.OUT, "analysis_panel.csv")).sort_values("panel_idx")
    emb = np.load(os.path.join(C.OUT, "course_embeddings.npy"))
    N = len(panel)
    assert emb.shape[0] == N

    term_ids = panel["term_id"].values
    idx_by_term = {t: np.where(term_ids == t)[0] for t in pd.unique(term_ids)}

    # raw cosine block (within-term only)
    cos_block = np.zeros((N, N), dtype=np.float32)
    for t, idx in idx_by_term.items():
        E = emb[idx]
        S = E @ E.T
        np.fill_diagonal(S, 0.0)
        cos_block[np.ix_(idx, idx)] = S
    np.save(os.path.join(C.OUT, "W_cosine_block.npy"), cos_block)

    summ = []
    for k in C.KNN_LIST:
        Wb = np.zeros((N, N), dtype=np.float64)
        for t, idx in idx_by_term.items():
            E = emb[idx]; S = E @ E.T; np.fill_diagonal(S, 0.0)
            Wt = knn_rowstd(S, k)
            Wb[np.ix_(idx, idx)] = Wt
            # per-term diagnostics
            kk = min(k, len(idx) - 1)
            kept = S[np.argsort(-S, axis=1)[:, 1:kk+1].flatten()] if len(idx) > 1 else np.array([0.0])
            summ.append({"k": k, "term_id": t, "n_courses": len(idx),
                         "neighbors_per_course": kk,
                         "mean_sim_kept": float(np.mean([np.sort(S[i])[::-1][:kk].mean() for i in range(len(idx))])) if len(idx)>1 else 0.0,
                         "row_sum_min": float(Wt.sum(1).min()), "row_sum_max": float(Wt.sum(1).max())})
        np.save(os.path.join(C.OUT, "W_k%d_block.npy" % k), Wb)
        # neighbor list for transparency (k_main only -> full, others summary)
        if k == C.KNN_MAIN:
            recs = []
            names = panel["course_name"].values; cids = panel["course_id"].values
            for t, idx in idx_by_term.items():
                E = emb[idx]; S = E @ E.T; np.fill_diagonal(S, -1)
                kk = min(k, len(idx) - 1)
                for ii, i in enumerate(idx):
                    order = np.argsort(-S[ii])[:kk]
                    for rank, jj in enumerate(order, 1):
                        recs.append({"term_id": t, "course_id": cids[i], "course_name": names[i],
                                     "rank": rank, "neighbor_course": names[idx[jj]],
                                     "cosine": round(float(S[ii, jj]), 4)})
            pd.DataFrame(recs).to_csv(os.path.join(C.OUT, "neighbors_k%d.csv" % k), index=False, encoding="utf-8-sig")

    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(C.OUT, "weights_summary.csv"), index=False, encoding="utf-8-sig")
    print("[03] built W for k =", C.KNN_LIST, "| N =", N, "| terms =", len(idx_by_term))
    print(sdf.round(3).to_string(index=False))
    # connectivity sanity: each block is fully connected within term, none across
    for k in C.KNN_LIST:
        Wb = np.load(os.path.join(C.OUT, "W_k%d_block.npy" % k))
        cross = 0
        for t1, i1 in idx_by_term.items():
            for t2, i2 in idx_by_term.items():
                if t1 != t2:
                    cross += Wb[np.ix_(i1, i2)].sum()
        print("k=%d cross-term weight (must be 0): %.6f | all rows sum~1: %s" %
              (k, cross, np.allclose(Wb.sum(1), 1.0)))

if __name__ == "__main__":
    main()
