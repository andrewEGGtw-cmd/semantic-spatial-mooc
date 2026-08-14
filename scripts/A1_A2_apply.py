# -*- coding: utf-8 -*-
"""
A1 + A2 只補 credit_est / credit_fee 兩欄到 analysis_panel.csv。
單獨執行不依賴 statsmodels，供 01 完整重跑前先落地 credit 欄。
"""
import os
import pandas as pd
import config as C

PANEL = os.path.join(C.OUT, "analysis_panel.csv")
df = pd.read_csv(PANEL)

df["credit_est"] = (pd.to_numeric(df["hours"], errors="coerce") / C.CREDIT_HOURS_PER).round(2)
df["credit_fee"] = df["credit_est"] * C.CREDIT_FEE_TWD

df.to_csv(PANEL, index=False, encoding="utf-8-sig")

print("[A1+A2] rows:", len(df))
print("credit_est stats:")
print(df["credit_est"].describe().round(3))
print("credit_fee stats:")
print(df["credit_fee"].describe().round(0))
print("hours=0 or NaN 樣本數:", int((df["hours"].fillna(0) == 0).sum()))
print("credit_est=0 or NaN 樣本數:", int((df["credit_est"].fillna(0) == 0).sum()))
print("credit_fee=0 or NaN 樣本數:", int((df["credit_fee"].fillna(0) == 0).sum()))
