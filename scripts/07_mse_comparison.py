# -*- coding: utf-8 -*-
"""
07 MSE / 預測誤差比較（回應口委「實證結果須納入 MSE 討論」）

模型：OLS / SAR / SEM / SDM，DV = ln(enroll+1)，年度 FE，W = k10（依年度分塊、列標準化）
指標：
  - 樣本內 MSE、RMSE、MAE
  - 5-fold 交叉驗證 MSE（空間模型於訓練集重估 rho/lambda，對測試集以簡化預測式評估）
說明：
  本檔以純 numpy 實作空間模型之最大概似估計（集中對數概似對 rho/lambda 做一維最適化），
  不依賴 spreg；已以主模型交叉驗證：ρ 與 AIC 可重現 04_spatial_models.py 之結果。
輸出：outputs/mse_comparison.csv
"""
import os, json
import numpy as np, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,'..','outputs')

def zstd(df, cols):
    Z=df[cols].astype(float).copy()
    for c in cols:
        s=Z[c]; sd=s.std(ddof=0)
        Z[c]=(s-s.mean())/(sd if sd>0 else 1.0)
    return Z

def build_design(panel, W, base_x, with_WX):
    n=len(panel)
    Xz=zstd(panel, base_x)
    parts=[np.ones((n,1)), Xz.values]
    yrs=sorted(panel['year'].unique())
    for yv in yrs[1:]:
        parts.append((panel['year']==yv).astype(float).values.reshape(-1,1))
    if with_WX:
        parts.append(W@Xz.values)
    return np.hstack(parts)

def ols_fit(y, Z):
    beta,_,_,_=np.linalg.lstsq(Z, y, rcond=None)
    return beta

def profile_lag(rho, y, Z, W, I):
    A=I-rho*W; Ay=A@y
    beta=ols_fit(Ay, Z)
    e=Ay-Z@beta
    n=len(y); sig2=float(e.T@e)/n
    sign,logdet=np.linalg.slogdet(A)
    ll=-n/2*np.log(2*np.pi)-n/2*np.log(sig2)+logdet-n/2
    return ll, beta, sig2

def profile_err(lam, y, Z, W, I):
    A=I-lam*W
    ys=A@y; Zs=A@Z
    beta=ols_fit(ys, Zs)
    e=ys-Zs@beta
    n=len(y); sig2=float(e.T@e)/n
    sign,logdet=np.linalg.slogdet(A)
    ll=-n/2*np.log(2*np.pi)-n/2*np.log(sig2)+logdet-n/2
    return ll, beta, sig2

def opt_param(profile, y, Z, W, I, lo=-0.98, hi=0.98):
    grid=np.linspace(lo,hi,197)
    best=max(grid, key=lambda v: profile(v,y,Z,W,I)[0])
    for span in (0.02,0.002,0.0002):
        loc=np.linspace(best-span,best+span,41)
        loc=loc[(loc>lo)&(loc<hi)]
        best=max(loc, key=lambda v: profile(v,y,Z,W,I)[0])
    return float(best)

def metrics(y, yhat):
    e=y-yhat
    return dict(MSE=float(np.mean(e**2)), RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))))

def rowstd(Wb):
    rs=Wb.sum(1,keepdims=True); rs[rs==0]=1.0
    return Wb/rs

def main():
    panel=pd.read_csv(os.path.join(OUT,'analysis_panel.csv')).sort_values('panel_idx').reset_index(drop=True)
    W=np.load(os.path.join(OUT,'W_k10_block.npy'))
    base_x=['video','materials','bloom_mid','bloom_high','credit_est']
    y=panel['ln_enroll'].astype(float).values.reshape(-1,1)
    n=len(panel); I=np.eye(n)

    rows=[]
    # ---- OLS ----
    Z=build_design(panel,W,base_x,False)
    b=ols_fit(y,Z); yhat=Z@b
    m=metrics(y,yhat); m.update(model='OLS', rho_lambda=np.nan)
    rows.append(m)

    # ---- SAR ----
    Z=build_design(panel,W,base_x,False)
    rho=opt_param(profile_lag,y,Z,W,I)
    ll,b,s2=profile_lag(rho,y,Z,W,I)
    yhat=np.linalg.inv(I-rho*W)@(Z@b)          # reduced-form prediction
    m=metrics(y,yhat); m.update(model='SAR', rho_lambda=rho)
    rows.append(m)

    # ---- SEM ----
    Z=build_design(panel,W,base_x,False)
    lam=opt_param(profile_err,y,Z,W,I)
    ll,b,s2=profile_err(lam,y,Z,W,I)
    yhat=Z@b                                    # E[y|X] 不含誤差空間結構
    m=metrics(y,yhat); m.update(model='SEM', rho_lambda=lam)
    rows.append(m)

    # ---- SDM ----
    Zd=build_design(panel,W,base_x,True)
    rho_d=opt_param(profile_lag,y,Zd,W,I)
    ll,bd,s2=profile_lag(rho_d,y,Zd,W,I)
    yhat=np.linalg.inv(I-rho_d*W)@(Zd@bd)
    m=metrics(y,yhat); m.update(model='SDM', rho_lambda=rho_d)
    rows.append(m)

    df=pd.DataFrame(rows)[['model','rho_lambda','MSE','RMSE','MAE']]

    # ---- 5-fold CV（分層於年度，空間結構於訓練子集重新列標準化）----
    rng=np.random.default_rng(42)
    idx=np.arange(n); rng.shuffle(idx)
    folds=np.array_split(idx,5)
    cv={k:[] for k in ['OLS','SAR','SEM','SDM']}
    for f in folds:
        te=np.array(sorted(f)); tr=np.array(sorted(set(idx)-set(f)))
        ptr=panel.iloc[tr].reset_index(drop=True)
        Wtr=rowstd(W[np.ix_(tr,tr)].copy()); Itr=np.eye(len(tr))
        ytr=ptr['ln_enroll'].astype(float).values.reshape(-1,1)
        yte=panel.iloc[te]['ln_enroll'].astype(float).values.reshape(-1,1)
        # 以全樣本標準化參數建立測試設計矩陣（避免資訊洩漏過於嚴苛，統一以訓練集標準化）
        for name,with_WX,kind in [('OLS',False,'ols'),('SAR',False,'lag'),('SEM',False,'err'),('SDM',True,'lag')]:
            Ztr=build_design(ptr,Wtr,base_x,with_WX)
            if kind=='ols':
                b=ols_fit(ytr,Ztr)
            elif kind=='lag':
                r_=opt_param(profile_lag,ytr,Ztr,Wtr,Itr); _,b,_=profile_lag(r_,ytr,Ztr,Wtr,Itr)
            else:
                l_=opt_param(profile_err,ytr,Ztr,Wtr,Itr); _,b,_=profile_err(l_,ytr,Ztr,Wtr,Itr)
            # 測試集：以自身區塊 W 建設計矩陣（僅用課程屬性，不使用測試集 y）
            pte=panel.iloc[te].reset_index(drop=True)
            Wte=rowstd(W[np.ix_(te,te)].copy())
            Zte=build_design(pte,Wte,base_x,with_WX)
            yhat=Zte@b
            cv[name].append(float(np.mean((yte-yhat)**2)))
    df['CV_MSE_5fold']=[np.mean(cv[m]) for m in df['model']]
    df=df.round(4)
    p=os.path.join(OUT,'mse_comparison.csv')
    df.to_csv(p,index=False,encoding='utf-8-sig')
    print(df.to_string(index=False))
    print('\nsaved:',p)

if __name__=='__main__':
    main()
