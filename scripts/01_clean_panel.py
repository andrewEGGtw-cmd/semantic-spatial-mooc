# -*- coding: utf-8 -*-
"""
01 build analysis-ready panel + descriptive stats
- pull analysis columns, build DV / controls / term_id / grouping cols
- credit_est / credit_fee 由 hours 與常數推算（A1、A2）
- reserve empty placeholder cols for Bloom (A3 待標)
NOTE: course_id is NOT unique across db hosts; use src_row (raw row index) as stable key.
"""
import numpy as np, pandas as pd, json, os
import config as C


def main():
    df = pd.read_csv(C.RAW_CSV)
    n0 = len(df)

    out = pd.DataFrame()
    out["src_row"] = df.index.values
    out["mnet_course_id"] = df["mnet_course_id"]
    out["course_id"] = df[C.ID_COL]
    out["db"] = df["db"]
    out["course_name"] = df["course_name_clean"]
    out["course_key"] = df["course_key"]
    out["open_sequence"] = df["open_sequence"]
    out["repeat_course_count"] = df["repeat_course_count"]

    out["year"] = df[C.YEAR_COL].astype(int)
    out["term"] = df[C.TERM_COL].astype(str)
    out["term_id"] = out["year"].astype(str) + "_" + out["term"]

    out["enroll"] = pd.to_numeric(df[C.ENROLL_RAW], errors="coerce")
    out["ln_enroll"] = np.log(out["enroll"] + 1.0)

    out["video"] = pd.to_numeric(df[C.VIDEO_COL], errors="coerce")
    out["materials"] = pd.to_numeric(df[C.MAT_COL], errors="coerce")
    fill_log = {}
    for col in ["video", "materials"]:
        miss = int(out[col].isna().sum())
        if miss:
            out[col] = out.groupby("year")[col].transform(lambda s: s.fillna(s.median()))
            fill_log[col] = "filled %d missing with within-year median" % miss

    for name, src in C.EXTRA_COVARS.items():
        out[name] = pd.to_numeric(df[src], errors="coerce")

    # A1｜依教育部 MOOCs 規範由學習時數推算學分數（1 學分 = 18 hr）
    # A2｜依 ewant SOS 學生指南手冊每學分固定學分費 NT$750（2021–2025）
    #     hours 缺失者 credit_* 為 NaN；hours=0 者 credit_* 為 0（未成班或資料異常）。
    out["credit_est"] = (out["hours"] / C.CREDIT_HOURS_PER).round(2)
    out["credit_fee"] = out["credit_est"] * C.CREDIT_FEE_TWD

    for c in C.RESERVED_EMPTY:
        out[c] = np.nan

    out["field_category"] = df[C.FIELD_COL]
    out["school_fe_id"] = df[C.SCHOOL_COL]
    out["school_name"] = df[C.SCHOOL_NAME]
    out["data_quality_flag"] = df[C.QUALITY_COL]

    out = out.sort_values(["year", "term", "course_id"]).reset_index(drop=True)
    out.insert(0, "panel_idx", np.arange(len(out)))
    out.to_csv(os.path.join(C.OUT, "analysis_panel.csv"), index=False, encoding="utf-8-sig")

    desc_vars = ["enroll", "ln_enroll", "video", "materials", "hours",
                 "instructors", "quiz", "credit_est", "credit_fee"]
    desc = out[desc_vars].describe().T
    desc["n_zero"] = [int((out[v] == 0).sum()) for v in desc_vars]
    desc["n_miss"] = [int(out[v].isna().sum()) for v in desc_vars]
    desc["skew"] = [float(out[v].skew()) for v in desc_vars]
    desc = desc.round(3)
    desc.to_csv(os.path.join(C.OUT, "descriptive_stats.csv"), encoding="utf-8-sig")

    out.groupby("year")["enroll"].agg(["count", "mean", "median", "min", "max"]).round(2)\
       .to_csv(os.path.join(C.OUT, "enroll_by_year.csv"), encoding="utf-8-sig")

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    reg_vars = ["video", "materials", "hours", "instructors", "quiz"]
    out[reg_vars].corr().round(3).to_csv(
        os.path.join(C.OUT, "correlation_matrix.csv"), encoding="utf-8-sig")
    Xc = sm.add_constant(out[reg_vars].dropna())
    vif = pd.DataFrame({
        "var": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    }).round(3)
    vif.to_csv(os.path.join(C.OUT, "vif.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "n_rows_raw": int(n0),
        "n_rows_panel": int(len(out)),
        "years": sorted(out["year"].unique().tolist()),
        "terms": sorted(out["term"].unique().tolist()),
        "n_by_term_id": out["term_id"].value_counts().sort_index().to_dict(),
        "enroll_zeros": int((out["enroll"] == 0).sum()),
        "quality_counts": out["data_quality_flag"].value_counts().to_dict(),
        "fill_log": fill_log,
        "reserved_empty_cols": C.RESERVED_EMPTY,
        "credit_hours_per": C.CREDIT_HOURS_PER,
        "credit_fee_twd_per_credit": C.CREDIT_FEE_TWD,
        "credit_est_nonzero_count": int((out["credit_est"] > 0).sum()),
        "credit_est_zero_or_nan_count": int((out["credit_est"].fillna(0) == 0).sum()),
    }
    with open(os.path.join(C.OUT, "clean_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[01] analysis_panel rows:", len(out), "cols:", out.shape[1])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("--- descriptive_stats ---"); print(desc.to_string())
    print("--- VIF ---"); print(vif.to_string(index=False))


if __name__ == "__main__":
    main()
