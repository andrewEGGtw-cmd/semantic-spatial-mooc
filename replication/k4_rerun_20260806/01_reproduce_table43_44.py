# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from libpysal.weights import full2W
import spreg, esda
SEED=20260626

panel=pd.read_csv("analysis_panel.csv").sort_values("panel_idx").reset_index(drop=True)
emb=np.load("course_embeddings.npy"); N=len(panel)

def knn_rowstd(S,k):
    n=S.shape[0]; W=np.zeros_like(S); kk=min(k,n-1)
    for i in range(n):
        order=np.argsort(-S[i]); nbr=[j for j in order if j!=i][:kk]; W[i,nbr]=S[i,nbr]
    W[W<0]=0.0; rs=W.sum(1,keepdims=True); rs[rs==0]=1.0; return W/rs

def build_W(k):
    tid=panel["term_id"].values; Wb=np.zeros((N,N))
    for t in pd.unique(tid):
        idx=np.where(tid==t)[0]; E=emb[idx]; S=E@E.T; np.fill_diagonal(S,0.0)
        Wb[np.ix_(idx,idx)]=knn_rowstd(S,k)
    return Wb

def zstd(df,cols):
    Z=df[cols].astype(float).copy()
    for c in cols:
        s=Z[c]; sd=s.std(ddof=0); Z[c]=(s-s.mean())/(sd if sd>0 else 1.0)
    return Z

BASE_X=["video","materials","bloom_mid","bloom_high","credit_est"]
Wb=build_W(4); np.save("W_k4_block.npy",Wb)
w=full2W(Wb); w.transform="r"
print("cross-term weight:",0.0,"| rows sum 1:",np.allclose(Wb.sum(1),1.0))

# ---- Moran's I ----
np.random.seed(SEED)
mo=esda.Moran(panel["ln_enroll"].values,w,permutations=999)
print("\n=== 表4-3 重現 (k=4) ===")
print("全樣本  I=%.4f  E=%.4f  p_sim=%.4f   | 論文: 0.104, -0.006, .021"%(mo.I,mo.EI,mo.p_sim))
paper={2021:(0.141,.063),2022:(-0.059,.432),2023:(0.047,.204),2024:(-0.000,.341),2025:(-0.009,.387)}
for yr in sorted(panel["year"].unique()):
    idx=panel.index[panel["year"]==yr].values
    Ws=Wb[np.ix_(idx,idx)].copy(); rs=Ws.sum(1,keepdims=True); rs[rs==0]=1; Ws=Ws/rs
    ws=full2W(Ws); ws.transform="r"
    np.random.seed(SEED)
    m=esda.Moran(panel.loc[idx,"ln_enroll"].values,ws,permutations=999)
    print("%d    I=%+.4f  E=%.4f  p_sim=%.3f   | 論文: %+.3f, %.3f"%(yr,m.I,m.EI,m.p_sim,*paper[yr]))

# ---- 表4-4 ----
D=pd.DataFrame(index=panel.index)
for y in sorted(panel["year"].unique())[1:]: D["yr%d"%y]=(panel["year"]==y).astype(float)
Xz=zstd(panel,BASE_X); Xbase=pd.concat([Xz,D],axis=1)
WXz=pd.DataFrame(Wb@Xz.values,columns=["W_"+c for c in BASE_X],index=panel.index)
Xsdm=pd.concat([Xbase,WXz],axis=1)
yv=panel["ln_enroll"].values.reshape(-1,1)
ols=spreg.OLS(yv,Xbase.values,w=w,spat_diag=True,moran=True)
sar=spreg.ML_Lag(yv,Xbase.values,w=w); sem=spreg.ML_Error(yv,Xbase.values,w=w)
sdm=spreg.ML_Lag(yv,Xsdm.values,w=w)
print("\n=== 表4-4 重現 (k=4) ===")
print("%-6s %10s %10s %10s %12s | %s"%("model","logLik","AIC","pseudoR2","spatial","論文"))
paper44={"OLS":(-292.57,605.14,0.180,None),"SAR":(-292.03,606.06,0.187,+0.121),
         "SEM":(-291.96,603.91,0.178,+0.148),"SDM":(-280.45,592.90,0.286,+0.039)}
for nm,m in [("OLS",ols),("SAR",sar),("SEM",sem),("SDM",sdm)]:
    ll=float(getattr(m,"logll",np.nan)); aic=float(getattr(m,"aic",np.nan))
    pr2=float(getattr(m,"pr2",getattr(m,"r2",np.nan)))
    sp=float(m.rho) if hasattr(m,"rho") else (float(m.betas.flatten()[-1]) if nm=="SEM" else None)
    p=paper44[nm]
    print("%-6s %10.2f %10.2f %10.4f %12s | %s"%(nm,ll,aic,pr2,("%+.4f"%sp if sp is not None else "-"),str(p)))

# Wald
names_full=["CONSTANT"]+list(Xsdm.columns)+["rho"]
ti=[names_full.index("W_"+v) for v in BASE_X]
th=sdm.betas.flatten()[ti]; V=sdm.vm[np.ix_(ti,ti)]
from scipy.stats import chi2
wald=float(th.T@np.linalg.inv(V)@th)
print("\nWald(H0: all theta=0) = %.2f, df=5, p=%.4f   | 論文: 25.10, df=5, p=.0001"%(wald,chi2.sf(wald,5)))
