# -*- coding: utf-8 -*-
"""
06 thesis-ready figures (300 dpi, Traditional-Chinese labels, Noto Sans CJK TC)
Outputs to analysis_pipeline/figures/*.png
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import config as C

OUT = C.OUT
FIG = os.path.join(C.BASE, "figures"); os.makedirs(FIG, exist_ok=True)

# ---- CJK font: prefer bundled Traditional-Chinese face ----
FONTDIR = os.path.join(C.BASE, "fonts"); os.makedirs(FONTDIR, exist_ok=True)
TC_OTF = os.path.join(FONTDIR, "NotoSansCJKtc-Regular.otf")
SYS_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.path.exists(TC_OTF) and os.path.exists(SYS_TTC):
    try:
        from fontTools.ttLib import TTCollection
        for _f in TTCollection(SYS_TTC).fonts:
            if _f["name"].getDebugName(1) == "Noto Sans CJK TC":
                _f.save(TC_OTF); break
    except Exception: pass
if os.path.exists(TC_OTF):
    fm.fontManager.addfont(TC_OTF); FONT = fm.FontProperties(fname=TC_OTF).get_name()
elif os.path.exists(SYS_TTC):
    fm.fontManager.addfont(SYS_TTC); FONT = fm.FontProperties(fname=SYS_TTC).get_name()
else:
    FONT = "sans-serif"
plt.rcParams.update({
    "font.family": FONT, "axes.unicode_minus": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "medium",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E5E2", "grid.linewidth": 0.8,
    "axes.edgecolor": "#888780", "axes.linewidth": 0.8,
})
BLUE, TEAL, CORAL, GRAY, AMBER = "#2C6FB3", "#1D9E75", "#D8602F", "#6F6E69", "#C8881A"

panel = pd.read_csv(os.path.join(OUT, "analysis_panel.csv")).sort_values("panel_idx").reset_index(drop=True)

def save(fig, name):
    p = os.path.join(FIG, name); fig.savefig(p, facecolor="white"); plt.close(fig); print("saved", name)

# ---- Fig 1: DV distribution (raw vs log) ----
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
ax[0].hist(panel["enroll"], bins=20, color=BLUE, edgecolor="white", linewidth=0.6)
ax[0].axvline(panel["enroll"].mean(), color=CORAL, ls="--", lw=1.5, label="平均 %.0f" % panel["enroll"].mean())
ax[0].axvline(panel["enroll"].median(), color=TEAL, ls=":", lw=1.8, label="中位 %.0f" % panel["enroll"].median())
ax[0].set_xlabel("修課人數"); ax[0].set_ylabel("課程數"); ax[0].set_title("(a) 修課人數（原始）")
ax[0].legend(frameon=False, fontsize=10)
ax[0].annotate("偏態=%.2f；7 門 0 人" % panel["enroll"].skew(), xy=(0.97, 0.78),
               xycoords="axes fraction", ha="right", fontsize=10, color=GRAY)
ax[1].hist(panel["ln_enroll"], bins=20, color=TEAL, edgecolor="white", linewidth=0.6)
ax[1].set_xlabel("ln(修課人數+1)"); ax[1].set_ylabel("課程數"); ax[1].set_title("(b) 對數轉換後")
save(fig, "fig1_dv_distribution.png")

# ---- Fig 2: enrollment by year (boxplot + mean) ----
fig, ax = plt.subplots(figsize=(6.8, 4.2))
years = sorted(panel["year"].unique())
data = [panel.loc[panel["year"] == y, "enroll"].values for y in years]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                medianprops=dict(color=CORAL, lw=1.6),
                meanprops=dict(marker="D", markerfacecolor=AMBER, markeredgecolor="white", markersize=6),
                flierprops=dict(marker="o", markersize=3, markerfacecolor=GRAY, markeredgecolor="none", alpha=0.5))
for b in bp["boxes"]: b.set(facecolor="#D6E4F2", edgecolor=BLUE, linewidth=1.0)
for w in bp["whiskers"]+bp["caps"]: w.set(color=BLUE, linewidth=1.0)
ax.set_xticklabels([str(y) for y in years]); ax.set_xlabel("年度（暑期班）"); ax.set_ylabel("修課人數")
ax.set_title("圖 4-2　各年度修課人數分布")
ax.plot([], [], color=CORAL, lw=1.6, label="中位數"); ax.scatter([], [], marker="D", c=AMBER, edgecolor="white", label="平均數")
ax.legend(frameon=False, fontsize=10, loc="upper left")
save(fig, "fig2_enroll_by_year.png")

# ---- Fig 3: field distribution ----
fields = ["管理類","資訊類","心理類","基礎科學類","語言文學類","人文社會類","醫療類","藝術創作類","史地類","法政類","工程類"]
cnts = [int((panel["field_category"] == f).sum()) for f in fields]
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ypos = np.arange(len(fields))[::-1]
ax.barh(ypos, cnts, color=GRAY, edgecolor="white", height=0.72)
ax.set_yticks(ypos); ax.set_yticklabels(fields)
for yp, c in zip(ypos, cnts): ax.text(c+0.3, yp, str(c), va="center", fontsize=10, color="#444441")
ax.set_xlabel("課程數"); ax.set_title("圖 4-3　課程學門分布（n=175）"); ax.grid(axis="y", visible=False)
save(fig, "fig3_field_distribution.png")

# ---- Fig 4: correlation heatmap ----
rv = ["video","materials","hours","instructors","quiz"]
lab = ["影片數","教材數","學習時數","主授教師數","測驗數"]
Cm = panel[rv].corr().values
fig, ax = plt.subplots(figsize=(5.8, 5.0))
im = ax.imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
ax.set_xticklabels(lab, rotation=35, ha="right"); ax.set_yticklabels(lab)
for i in range(len(lab)):
    for j in range(len(lab)):
        v = Cm[i, j]; ax.text(j, i, "%.2f" % v, ha="center", va="center",
                              color="white" if abs(v) > 0.6 else "#2C2C2A", fontsize=11)
ax.set_title("圖 4-4　教學設計變項相關係數")
cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.outline.set_visible(False)
ax.grid(False)
save(fig, "fig4_correlation_heatmap.png")

# ---- Fig 5: Moran scatterplot (ln_enroll, k=10) ----
W = np.load(os.path.join(OUT, "W_k10_block.npy"))
y = panel["ln_enroll"].values; z = (y - y.mean()) / y.std(); lag = W @ z
slope = np.polyfit(z, lag, 1)[0]
fig, ax = plt.subplots(figsize=(6.0, 5.2))
ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(0, color=GRAY, lw=0.8)
ax.scatter(z, lag, s=26, color=BLUE, alpha=0.65, edgecolor="white", linewidth=0.4)
xs = np.linspace(z.min(), z.max(), 50); ax.plot(xs, slope*xs, color=CORAL, lw=1.8)
ax.set_xlabel("ln(修課人數+1)（標準化）"); ax.set_ylabel("語意鄰近課程之空間滯後")
ax.set_title("")
ax.annotate("Moran's I = 0.047\np = 0.055（邊際）", xy=(0.04, 0.90), xycoords="axes fraction",
            fontsize=11, color="#444441", va="top")
save(fig, "fig5_moran_scatter.png")

# ---- Fig 6: SDM impacts (direct/indirect/total) ----
imp = pd.read_csv(os.path.join(OUT, "sdm_impacts.csv"))
varlab = {"video":"影片數", "materials":"教材數", "bloom_mid":"中階認知比例",
          "bloom_high":"高階認知比例", "credit_est":"學分數"}
VARS = ["credit_est", "bloom_high", "materials", "video", "bloom_mid"]
effs = ["direct","indirect","total"]; efflab = {"direct":"直接效果","indirect":"間接效果（外溢）","total":"總效果"}
effcol = {"direct":BLUE, "indirect":AMBER, "total":TEAL}
fig, ax = plt.subplots(figsize=(7.6, 8.2))
yt, ytl = [], []
base = 0
for vi, v in enumerate(VARS):
    for ei, eff in enumerate(effs):
        row = imp[(imp["variable"]==v) & (imp["effect"]==eff)].iloc[0]
        ypos = base; base += 1
        est = row["estimate"]; lo = row["ci_lo"]; hi = row["ci_hi"]
        sig = row["p_value"] < 0.05
        ax.errorbar(est, ypos, xerr=[[est-lo],[hi-est]], fmt="o", ms=7, color=effcol[eff],
                    ecolor=effcol[eff], elinewidth=1.6, capsize=3,
                    markerfacecolor=effcol[eff] if sig else "white", markeredgecolor=effcol[eff], markeredgewidth=1.4)
        yt.append(ypos); ytl.append("%s — %s%s" % (varlab[v], efflab[eff], "" if sig else "（n.s.）"))
    base += 0.6
ax.axvline(0, color=GRAY, lw=1.0, ls="--")
ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=10); ax.invert_yaxis()
ax.set_xlabel("對 ln(修課人數+1) 之效果（每 1 標準差，95% 模擬信賴區間）")
ax.set_title(""); ax.grid(axis="y", visible=False)
ax.annotate("實心=p<.05；空心=不顯著", xy=(0.98,0.015), xycoords="axes fraction", ha="right", fontsize=9, color=GRAY)
save(fig, "fig6_sdm_impacts.png")

# ---- Fig 7: robustness of rho ----
rob = pd.read_csv(os.path.join(OUT, "robustness_results.csv"))
rob = rob[pd.to_numeric(rob["rho"], errors="coerce").notna()].copy()
rob["rho"] = rob["rho"].astype(float)
short = {"M0 main: k10, ln, all175, yearFE":"主模型 (k10/ln/年FE)","M1 k5":"k=5",
         "M2 raw DV (enroll)":"原始 DV","M4 +field FE":"+學門 FE","M5 +school FE":"+學校 FE",
         "M6 extended X (5 covars)":"延伸 X（5 變項）"}
rob["lab"] = rob["config"].map(lambda s: short.get(s, s.replace("M3 good-only (n=153)","good-only (n=153)")))
order = rob.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ypos = np.arange(len(order))
ax.axvline(0, color=GRAY, lw=1.2, ls="--")
colors = [CORAL if r < 0 else BLUE for r in order["rho"]]
ax.scatter(order["rho"], ypos, s=70, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
for yp, r in zip(ypos, order["rho"]): ax.text(r, yp+0.18, "%.2f" % r, ha="center", fontsize=9, color="#444441")
ax.set_yticks(ypos); ax.set_yticklabels(order["lab"], fontsize=10)
ax.set_xlabel("空間自迴歸係數 ρ"); ax.set_title("")
ax.grid(axis="y", visible=False); ax.set_xlim(min(-0.75, order["rho"].min()-0.1), 0.15)
save(fig, "fig7_robustness_rho.png")

print("\nALL FIGURES IN:", FIG)
print("font used:", FONT)
