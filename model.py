from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, curve_fit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import SplineTransformer
from statsmodels.robust.norms import HuberT
from statsmodels.robust.robust_linear_model import RLM

EPS = 1e-9

@dataclass
class FitResult:
    data: pd.DataFrame
    features: List[str]
    feature_names: List[str]
    coefficients: Dict[str, float]
    refs: Dict[str, float]
    nonlinear_hr: bool
    hr_knots: Optional[np.ndarray]
    rlm_result: object
    trend_kind: str
    trend_values: np.ndarray
    trend_unconstrained: np.ndarray
    trend_isotonic: np.ndarray
    residuals: np.ndarray
    robust_z: np.ndarray
    anomalies: np.ndarray
    weights: np.ndarray
    diagnostics: Dict[str, float]
    parsimony: Dict[str, float]


def _mad(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return med, 1.4826 * mad if mad > 0 else np.nanstd(x) + EPS


def _trend_basis(t: np.ndarray, df: int = 5) -> Tuple[np.ndarray, SplineTransformer]:
    n_knots = max(3, min(df, max(3, len(t)//4)))
    trans = SplineTransformer(n_knots=n_knots, degree=2, include_bias=False)
    B = trans.fit_transform(np.asarray(t).reshape(-1,1))
    return B, trans


def _hr_basis(hr: np.ndarray, nonlinear: bool) -> Tuple[np.ndarray, Optional[SplineTransformer], List[str]]:
    h = np.asarray(hr).reshape(-1,1)
    if not nonlinear:
        return h, None, ["hr"]
    n_knots = max(3, min(4, max(3, len(hr)//8)))
    trans = SplineTransformer(n_knots=n_knots, degree=2, include_bias=False)
    B = trans.fit_transform(h)
    return B, trans, [f"hr_spline_{i+1}" for i in range(B.shape[1])]


def _fit_rlm(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    return RLM(y, X1, M=HuberT()).fit(maxiter=200)


def _aic_like(resid, k):
    n = len(resid)
    rss = np.sum(np.square(resid)) + EPS
    return n*np.log(rss/n) + 2*k


def _select_hr_nonlinearity(df: pd.DataFrame, base_cols: List[str]) -> bool:
    # Parsimonious: only consider HR spline with enough observations and HR spread.
    if len(df) < 18 or df["avg_hr_bpm"].nunique() < 8:
        return False
    y = df["speed_kmh"].to_numpy()
    tB, _ = _trend_basis(df["t_days"].to_numpy(), df=4)
    extras = [df[c].fillna(df[c].median()).to_numpy()[:,None] for c in base_cols]
    Xextra = np.column_stack(extras + [tB]) if extras else tB
    lin_hr, _, _ = _hr_basis(df["avg_hr_bpm"].to_numpy(), False)
    spl_hr, _, _ = _hr_basis(df["avg_hr_bpm"].to_numpy(), True)
    r1 = _fit_rlm(y, np.column_stack([lin_hr, Xextra]))
    r2 = _fit_rlm(y, np.column_stack([spl_hr, Xextra]))
    a1 = _aic_like(y-r1.predict(), len(r1.params))
    a2 = _aic_like(y-r2.predict(), len(r2.params))
    return a2 + 2 < a1  # meaningful improvement, not marginal complexity.


def _temporal_validation(df: pd.DataFrame, cols: List[str], nonlinear_hr: bool) -> float:
    n = len(df)
    if n < 12:
        return np.nan
    split = max(8, int(n*0.7))
    train, test = df.iloc[:split], df.iloc[split:]
    # Simple linear time for validation stability; purpose is comparing nuisance-variable sets.
    def design(z):
        blocks=[]
        hb, tr, _ = _hr_basis(z["avg_hr_bpm"].to_numpy(), False)  # use linear HR for split comparability
        blocks.append(hb)
        for c in cols:
            blocks.append(z[c].fillna(train[c].median()).to_numpy()[:,None])
        blocks.append(z["t_days"].to_numpy()[:,None])
        return np.column_stack(blocks)
    Xtr=design(train); Xte=design(test)
    r=_fit_rlm(train["speed_kmh"].to_numpy(), Xtr)
    pred=np.column_stack([np.ones(len(Xte)),Xte]) @ r.params
    return float(mean_absolute_error(test["speed_kmh"], pred))


def fit_efficiency_model(df: pd.DataFrame) -> FitResult:
    x = df.copy().sort_values("date").reset_index(drop=True)
    if len(x) < 6:
        raise ValueError("Servono almeno 6 corse complete (data, distanza, passo e FC media) per un primo fit.")

    candidate_cols=[]
    if x["distance_km"].notna().all() and x["distance_km"].nunique() >= 3:
        candidate_cols.append("distance_km")
    if x["elevation_m"].notna().sum() >= max(6, int(.7*len(x))) and x["elevation_m"].nunique() >= 3:
        x["elevation_m"] = x["elevation_m"].fillna(x["elevation_m"].median())
        candidate_cols.append("elevation_m")

    # Parsimony stress test against HR + time only.
    cv_full = _temporal_validation(x, candidate_cols, False)
    cv_simple = _temporal_validation(x, [], False)
    if np.isfinite(cv_full) and np.isfinite(cv_simple) and cv_simple <= cv_full * 1.05:
        model_cols=[]
        parsimony_decision="simple"
    else:
        model_cols=candidate_cols
        parsimony_decision="full"

    nonlinear_hr=_select_hr_nonlinearity(x, model_cols)
    y=x["speed_kmh"].to_numpy()
    hrB, hrTrans, hrNames = _hr_basis(x["avg_hr_bpm"].to_numpy(), nonlinear_hr)
    trendB, trendTrans = _trend_basis(x["t_days"].to_numpy(), df=5)
    blocks=[hrB]
    names=list(hrNames)
    for c in model_cols:
        blocks.append(x[c].to_numpy()[:,None]); names.append(c)
    blocks.append(trendB); names += [f"trend_spline_{i+1}" for i in range(trendB.shape[1])]
    X=np.column_stack(blocks)
    rlm=_fit_rlm(y,X)
    fitted=rlm.predict()
    resid=y-fitted

    # Decompose nuisance and trend.
    p=np.asarray(rlm.params)
    idx=1
    hrcoef=p[idx:idx+hrB.shape[1]]; idx += hrB.shape[1]
    nuisance = p[0] + hrB@hrcoef
    coef={"intercept":float(p[0])}
    for j,nm in enumerate(hrNames): coef[nm]=float(hrcoef[j])
    for c in model_cols:
        coef[c]=float(p[idx]); nuisance += p[idx]*x[c].to_numpy(); idx+=1
    trendcoef=p[idx:]
    trend_unconstrained=trendB@trendcoef

    # Efficiency point normalized to medians of included nuisance variables.
    refs={"avg_hr_bpm":float(x["avg_hr_bpm"].median())}
    hrefB = hrTrans.transform([[refs["avg_hr_bpm"]]])[0] if hrTrans is not None else np.array([refs["avg_hr_bpm"]])
    hr_effect = hrB@hrcoef
    hr_ref_effect = float(hrefB@hrcoef)
    correction = hr_effect - hr_ref_effect
    for c in model_cols:
        refs[c]=float(x[c].median())
        correction += coef[c]*(x[c].to_numpy()-refs[c])
    v_eq=y-correction

    # Trend candidates on equivalent performance. Isotonic is increasing only as comparator, never forced.
    iso=IsotonicRegression(increasing=True, out_of_bounds="clip")
    trend_iso=iso.fit_transform(x["t_days"].to_numpy(), v_eq)
    # unconstrained smoother from RLM decomposition recentered to v_eq scale
    trend_unconstrained_abs = trend_unconstrained + np.median(v_eq-trend_unconstrained)
    mae_un=float(np.mean(np.abs(v_eq-trend_unconstrained_abs)))
    mae_iso=float(np.mean(np.abs(v_eq-trend_iso)))
    # Prefer unconstrained unless isotonic clearly helps; monotonicity is reported regardless.
    if mae_iso < mae_un*0.9:
        trend_kind="isotonic"
        trend_values=trend_iso
    else:
        trend_kind="unconstrained"
        trend_values=trend_unconstrained_abs

    eq_resid=v_eq-trend_values
    med, scale=_mad(eq_resid)
    rz=(eq_resid-med)/(scale+EPS)
    anomalies=np.abs(rz)>3.5
    weights=np.asarray(getattr(rlm, "weights", np.ones(len(x))))

    x["v_eq_kmh"]=v_eq
    x["model_fitted_kmh"]=fitted
    x["residual_kmh"]=resid
    x["eq_residual_kmh"]=eq_resid
    x["robust_z"]=rz
    x["anomaly"]=anomalies
    x["robust_weight"]=weights

    diagnostics={
        "mae_speed_kmh":float(np.mean(np.abs(resid))),
        "resid_mad_kmh":float(scale),
        "trend_mae_unconstrained":mae_un,
        "trend_mae_isotonic":mae_iso,
        "monotonicity_delta_mae":mae_iso-mae_un,
        "n_anomalies":int(anomalies.sum()),
        "n":len(x),
    }
    parsimony={
        "temporal_mae_full":cv_full,
        "temporal_mae_simple":cv_simple,
        "decision":parsimony_decision,
        "used_distance":float("distance_km" in model_cols),
        "used_elevation":float("elevation_m" in model_cols),
    }
    return FitResult(x, model_cols, names, coef, refs, nonlinear_hr, None, rlm, trend_kind,
                     trend_values, trend_unconstrained_abs, trend_iso, resid, rz, anomalies,
                     weights, diagnostics, parsimony)


def saturating(t, ginf, g0, k):
    return ginf - (ginf-g0)*np.exp(-np.maximum(k,1e-9)*t)


def fit_forecast_candidates(fit: FitResult) -> Dict:
    d=fit.data
    t=d["t_days"].to_numpy(float)
    y=d["v_eq_kmh"].to_numpy(float)
    if len(t)<8:
        return {"status":"insufficient"}
    span=max(t.max()-t.min(),1)
    g0_init=float(np.median(y[:max(2,len(y)//5)]))
    ginf_init=float(max(np.quantile(y,.8),g0_init+.1))
    k_init=1/max(span,30)
    sat_ok=True
    try:
        popt,_=curve_fit(saturating,t,y,p0=[ginf_init,g0_init,k_init],
                         bounds=([min(y)-3,min(y)-3,1e-7],[max(y)+8,max(y)+3,1.0]),maxfev=30000)
        sat_pred=saturating(t,*popt)
        sat_mae=float(mean_absolute_error(y,sat_pred))
    except Exception:
        sat_ok=False; popt=np.array([np.nan,np.nan,np.nan]); sat_mae=np.inf
    # linear alternative
    A=np.column_stack([np.ones(len(t)),t])
    lin=_fit_rlm(y,t[:,None])
    lin_pred=lin.predict(); lin_mae=float(mean_absolute_error(y,lin_pred))
    # sqrt-time alternative, gentle non-saturating slowdown
    sqrtX=np.sqrt(np.maximum(t,0)+1)[:,None]
    sq=_fit_rlm(y,sqrtX); sq_pred=sq.predict(); sq_mae=float(mean_absolute_error(y,sq_pred))
    candidates={"saturating":sat_mae,"linear":lin_mae,"sqrt_time":sq_mae}
    best=min(candidates,key=candidates.get)
    ginf,g0,k=map(float,popt)
    tau=1/k if sat_ok and k>0 else np.nan
    return {"status":"ok","saturating_params":{"g0":g0,"ginf":ginf,"k":k,"tau_days":tau},
            "candidate_mae":candidates,"best_in_sample":best,"linear_params":lin.params.tolist(),"sqrt_params":sq.params.tolist()}


def temporal_holdout_forecast_validation(fit: FitResult) -> Dict[str,float]:
    d=fit.data; n=len(d)
    if n<12: return {}
    split=max(8,int(.7*n)); tr=d.iloc[:split]; te=d.iloc[split:]
    ttr=tr.t_days.to_numpy(); ytr=tr.v_eq_kmh.to_numpy(); tte=te.t_days.to_numpy(); yte=te.v_eq_kmh.to_numpy()
    out={}
    try:
        p,_=curve_fit(saturating,ttr,ytr,p0=[np.quantile(ytr,.8),ytr[0],1/max(ttr.max(),30)],
                      bounds=([min(ytr)-3,min(ytr)-3,1e-7],[max(ytr)+8,max(ytr)+3,1.0]),maxfev=30000)
        out["saturating_mae"]=float(mean_absolute_error(yte,saturating(tte,*p)))
    except Exception: out["saturating_mae"]=np.nan
    lin=_fit_rlm(ytr,ttr[:,None]); out["linear_mae"]=float(mean_absolute_error(yte,np.column_stack([np.ones(len(tte)),tte])@lin.params))
    sq=_fit_rlm(ytr,np.sqrt(ttr+1)[:,None]); out["sqrt_time_mae"]=float(mean_absolute_error(yte,np.column_stack([np.ones(len(tte)),np.sqrt(tte+1)])@sq.params))
    return out


def bootstrap_forecast(fit: FitResult, future_days: np.ndarray, n_boot: int=150, seed: int=42) -> Dict:
    """Residual bootstrap with full pipeline refit on every replicate.

    Resamples robust-model residuals on the raw speed scale, rebuilds a pseudo dataset,
    refits nuisance terms + trend + anomaly weights from zero, then refits the saturating
    forecast. A horizon-dependent process term is added after the last observed date so
    uncertainty does not remain unrealistically flat years into the future.
    """
    rng=np.random.default_rng(seed)
    d=fit.data.copy()
    base=fit_forecast_candidates(fit)
    if base.get("status")!="ok" or not np.isfinite(base["saturating_params"]["k"]): return {}
    raw_fitted=d["model_fitted_kmh"].to_numpy(float)
    raw_resid=d["residual_kmh"].to_numpy(float)
    curves=[]; params=[]
    observed_span=max(float(d.t_days.max()-d.t_days.min()),30.0)
    for _ in range(n_boot):
        pseudo=d.copy()
        pseudo_speed=raw_fitted+rng.choice(raw_resid,size=len(raw_resid),replace=True)
        pseudo["speed_kmh"]=pseudo_speed
        pseudo["pace_min_km"]=60/np.maximum(pseudo_speed,EPS)
        try:
            refit=fit_efficiency_model(pseudo)
            fb=fit_forecast_candidates(refit)
            if fb.get("status")!="ok":
                continue
            pp=fb["saturating_params"]
            pb=np.array([pp["ginf"],pp["g0"],pp["k"]],float)
            curve=saturating(future_days,*pb)
            eq_resid=refit.data.v_eq_kmh.to_numpy()-saturating(refit.data.t_days.to_numpy(),*pb)
            sigma=float(np.std(eq_resid,ddof=1)) if len(eq_resid)>3 else 0.0
            horizon=np.maximum(future_days-float(d.t_days.max()),0.0)
            curve=curve+rng.normal(0,sigma*np.sqrt(1+horizon/observed_span),size=len(curve))
            curves.append(curve); params.append(pb)
        except Exception:
            continue
    if len(curves)<20: return {}
    arr=np.vstack(curves); par=np.vstack(params)
    return {"median":np.nanmedian(arr,axis=0),"lo":np.nanpercentile(arr,2.5,axis=0),"hi":np.nanpercentile(arr,97.5,axis=0),
            "param_lo":np.nanpercentile(par,2.5,axis=0),"param_hi":np.nanpercentile(par,97.5,axis=0),"n_success":len(curves)}

def predict_equivalent_on_date(fit: FitResult, date: pd.Timestamp, forecast: Dict) -> float:
    t=(pd.Timestamp(date)-fit.data.date.min()).total_seconds()/86400
    if forecast.get("status")!="ok": return np.nan
    p=forecast["saturating_params"]
    return float(saturating(np.array([t]),p["ginf"],p["g0"],p["k"])[0])


def predict_speed_conditions(fit: FitResult, v_eq: float, hr: float, distance: float, elevation: float|None) -> float:
    coef=fit.coefficients; ref=fit.refs
    # HR effect relative to reference.
    if fit.nonlinear_hr:
        # reconstruct transformer consistently from observed HR
        hb, trans, names=_hr_basis(fit.data.avg_hr_bpm.to_numpy(), True)
        a=trans.transform([[hr]])[0]; b=trans.transform([[ref["avg_hr_bpm"]]])[0]
        hc=np.array([coef[n] for n in names]); delta=float((a-b)@hc)
    else:
        delta=coef.get("hr",0)*(hr-ref["avg_hr_bpm"])
    if "distance_km" in fit.features:
        delta += coef["distance_km"]*(distance-ref["distance_km"])
    if "elevation_m" in fit.features and elevation is not None:
        delta += coef["elevation_m"]*(elevation-ref["elevation_m"])
    return float(v_eq+delta)


def target_date_for_speed(fit: FitResult, target_speed: float, hr: float, distance: float, elevation: float|None, forecast: Dict, max_years=5):
    if forecast.get("status")!="ok": return {"reachable":False,"reason":"forecast_non_disponibile"}
    # Convert requested-condition speed to equivalent target.
    dummy_eq=0.0
    delta=predict_speed_conditions(fit,dummy_eq,hr,distance,elevation)
    target_eq=target_speed-delta
    p=forecast["saturating_params"]; g0=p["g0"]; ginf=p["ginf"]; k=p["k"]
    if not np.isfinite(ginf) or k<=0: return {"reachable":False,"reason":"plateau_non_identificato"}
    if target_eq >= ginf:
        return {"reachable":False,"reason":"oltre_asintoto_modello","target_eq":target_eq,"ginf":ginf}
    ratio=(ginf-target_eq)/(ginf-g0)
    if ratio<=0: return {"reachable":False,"reason":"inversione_non_valida"}
    t=-np.log(ratio)/k
    if t<0: t=0
    date=fit.data.date.min()+pd.to_timedelta(t,unit="D")
    maxdate=fit.data.date.max()+pd.DateOffset(years=max_years)
    if date>maxdate: return {"reachable":False,"reason":f"oltre_orizzonte_{max_years}a","date":date}
    return {"reachable":True,"date":date,"t_days":t,"target_eq":target_eq}
