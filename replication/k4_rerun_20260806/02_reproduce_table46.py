# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from libpysal.weights import full2W
import spreg
SEED=20260626
panel=pd.read_csv("analysis_panel.csv").sort_values("panel_idx").reset_index(drop=True)
emb=np.load("course_embeddings.npy"); N=len(panel)
def knn_rowstd(S,k):
    n=S.shape[0]; W=np.zeros_like(S); kk=min(k,n-1)
    for i in range(n):
        o=np.argsort(-S[i]); nbr=[j for j in o if j!=i][:kk]; W[i,nbr]=S[i,nbr]
    W[W<0]=0.0; rs=W.sum(1,keepdims=True); rs[rs==0]=1.0; return W/rs
def build_W(k):
    tid=panel["term_id"].values; Wb=np.zeros((N,N))
    for t in pd.unique(tid):
        idx=np.where(tid==t)[0]; E=emb[idx]; S=E@E.T; np.fill_diagonal(S,0.0)
        Wb[np.ix_(idx,idx)]=knn_rowstd(S,k)
    return Wb
def rowstd(W):
    rs=W.sum(1,keepdims=True); rs[rs==0]=1.0; return W/rs
def zstd(df,cols):
    Z=df[cols].astype(float).copy()
    for c in cols:
        s=Z[c]; sd=s.std(ddof=0); Z[c]=(s-s.mean())/(sd if sd>0 else 1.0)
    return Z
def run(panel,Wb,base_x,y_col="ln_enroll",add_field=False,add_school=False,R=2000):
    panel=panel.reset_index(drop=True); y=panel[y_col].values.reshape(-1,1)
    Xz=zstd(panel,base_x); parts=[Xz]
    for yv in sorted(panel["year"].unique())[1:]:
        parts.append(pd.DataFrame({"yr%d"%yv:(panel["year"]==yv).astype(float)}))
    if add_field: parts.append(pd.get_dummies(panel["field_category"],prefix="fld",drop_first=True).astype(float).reset_index(drop=True))
    if add_school: parts.append(pd.get_dummies(panel["school_fe_id"].astype(str),prefix="sch",drop_first=True).astype(float).reset_index(drop=True))
    WX=pd.DataFrame(Wb@Xz.values,columns=["W_"+c for c in base_x])
    X=pd.concat([p.reset_index(drop=True) for p in parts]+[WX],axis=1)
    names=list(X.columns); w=full2W(Wb); w.transform="r"
    m=spreg.ML_Lag(y,X.values,w=w,name_y=y_col,name_x=names)
    b=m.betas.flatten(); vm=m.vm; full=["CONSTANT"]+names+["rho"]; ri=len(b)-1
    col={nm:full.index(nm) for nm in full}; n=Wb.shape[0]; I=np.eye(n)
    def imp(v):
        A=np.linalg.inv(I-v[ri]*Wb); r={}
        for x in base_x:
            M=A@(v[col[x]]*I+v[col["W_"+x]]*Wb); d=np.trace(M)/n; t=M.sum()/n; r[x]=(d,t-d,t)
        return r
    pt=imp(b); rng=np.random.default_rng(SEED)
    draws=rng.multivariate_normal(b,vm,size=R)
    sims={x:{"direct":[],"indirect":[],"total":[]} for x in base_x}
    for d in draws:
        r=imp(d)
        for x in base_x:
            for i,e in enumerate(["direct","indirect","total"]): sims[x][e].append(r[x][i])
    out={"rho":round(float(m.rho),4),"aic":round(float(m.aic),2),"n":n,
         "logLik":round(float(m.logll),2),"pseudo_r2":round(float(m.pr2),4)}
    for x in base_x:
        for i,e in enumerate(["direct","indirect","total"]):
            arr=np.array(sims[x][e]); p=2*min((arr>0).mean(),(arr<0).mean())
            out["%s_%s"%(x,e)]=round(pt[x][i],4); out["%s_%s_p"%(x,e)]=round(float(p),4)
    return out

focal=["video","materials","bloom_mid","bloom_high","credit_est"]
W4,W3,W5=build_W(4),build_W(3),build_W(5)
good=panel["data_quality_flag"].eq("good").values
pg=panel[good].reset_index(drop=True)
Wg=rowstd(W4[np.ix_(np.where(good)[0],np.where(good)[0])].copy())
p7=panel.copy(); p7["credit_int"]=p7["credit_est"].round()
focal7=["video","materials","bloom_mid","bloom_high","credit_int"]

cfgs=[("M0 主模型 k=4",dict(panel=panel,Wb=W4,base_x=focal)),
      ("M1 k=3",       dict(panel=panel,Wb=W3,base_x=focal)),
      ("M1b k=5",      dict(panel=panel,Wb=W5,base_x=focal)),
      ("M2 原始依變項", dict(panel=panel,Wb=W4,base_x=focal,y_col="enroll")),
      ("M3 品質良好樣本",dict(panel=pg,Wb=Wg,base_x=focal)),
      ("M4 ＋學門 FE",  dict(panel=panel,Wb=W4,base_x=focal,add_field=True)),
      ("M5 ＋學校 FE",  dict(panel=panel,Wb=W4,base_x=focal,add_school=True)),
      ("M6 延伸控制變項",dict(panel=panel,Wb=W4,base_x=["video","materials","hours","instructors","quiz"])),
      ("M7 整數學分",   dict(panel=p7,Wb=W4,base_x=focal7))]
paper={"M0 主模型 k=4":(+0.04,592.9),"M1 k=3":(+0.05,601.5),"M1b k=5":(-0.05,601.8),
       "M2 原始依變項":(-0.25,1905),"M3 品質良好樣本":(-0.26,265.3),"M4 ＋學門 FE":(+0.08,594.7),
       "M5 ＋學校 FE":(+0.05,588.0),"M6 延伸控制變項":(-0.03,596.9),"M7 整數學分":(+0.01,597.9)}
rows=[]
print("%-16s %5s %8s %10s %10s | %-16s %s"%("設定","n","ρ","AIC","pseudoR²","論文ρ/AIC","符合"))
for lab,kw in cfgs:
    r=run(**kw); pr,pa=paper[lab]
    ok = (abs(r["rho"]-pr)<0.006) and (abs(r["aic"]-pa)<max(0.06,abs(pa)*0.0005))
    print("%-16s %5d %+8.4f %10.2f %10.4f | %+.2f / %-8.1f %s"%(lab,r["n"],r["rho"],r["aic"],r["pseudo_r2"],pr,pa,"✔" if ok else "✘"))
    rows.append({"設定":lab,**r,"match":ok})
pd.DataFrame(rows).to_csv("robustness_k4.csv",index=False,encoding="utf-8-sig")
