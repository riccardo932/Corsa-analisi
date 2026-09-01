from __future__ import annotations

import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from data_utils import CANONICAL_COLUMNS, add_engineered_columns, canonicalize_dataframe, detect_duplicates, pace_str_from_speed, read_uploaded_csv
from model import (bootstrap_forecast, fit_efficiency_model, fit_forecast_candidates,
                   predict_equivalent_on_date, predict_speed_conditions, target_date_for_speed,
                   temporal_holdout_forecast_validation)
from charts import raw_over_time, speed_hr, normalized_trend, residuals, forecast_chart
from vision_extractor import DEFAULT_VISION_MODEL, extract_runs_from_image

st.set_page_config(page_title="Running Efficiency Lab", page_icon="🏃", layout="wide")
st.title("Running Efficiency Lab")
st.caption("Modello statistico dinamico della velocità equivalente a condizioni comparabili. Non è una misura di running economy da laboratorio.")

if "runs" not in st.session_state:
    st.session_state.runs = pd.DataFrame(columns=CANONICAL_COLUMNS)
if "extraction_notes" not in st.session_state:
    st.session_state.extraction_notes = []

with st.sidebar:
    st.header("1 · Dati")

    st.subheader("Screenshot")
    screenshots = st.file_uploader(
        "Carica screenshot Strava / Garmin / Apple Fitness",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="I valori estratti vengono sempre mostrati nella tabella prima del fit.",
    )
    secret_key = None
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
        secret_model = st.secrets.get("OPENAI_VISION_MODEL", DEFAULT_VISION_MODEL)
    except Exception:
        secret_model = DEFAULT_VISION_MODEL
    api_key = secret_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "OpenAI API key (solo per estrazione screenshot)",
            type="password",
            help="Per il deploy online è preferibile salvarla nei Secrets di Streamlit, non nel codice.",
        )
    vision_model = secret_model or DEFAULT_VISION_MODEL
    if screenshots:
        for shot in screenshots:
            st.image(shot, caption=shot.name, use_container_width=True)
        if st.button("Estrai dati dagli screenshot", type="primary", use_container_width=True):
            if not api_key:
                st.error("Per l'estrazione automatica serve una OpenAI API key. Puoi comunque usare CSV o inserimento manuale.")
            else:
                extracted=[]
                notes=[]
                for shot in screenshots:
                    try:
                        df_img, img_notes = extract_runs_from_image(shot.getvalue(), shot.type or "image/png", api_key, vision_model)
                        if not df_img.empty:
                            extracted.append(df_img)
                        notes.extend([f"{shot.name}: {n}" for n in img_notes])
                    except Exception as e:
                        st.error(f"{shot.name}: {e}")
                if extracted:
                    st.session_state.runs = pd.concat([st.session_state.runs, *extracted], ignore_index=True)
                    st.session_state.extraction_notes.extend(notes)
                    st.session_state["fit_requested"] = False
                    st.success(f"Estratte {sum(len(x) for x in extracted)} corse. Controlla la tabella prima di analizzare.")
                    st.rerun()
                elif notes:
                    st.session_state.extraction_notes.extend(notes)

    st.divider()
    st.subheader("CSV")
    uploads=st.file_uploader("Importa CSV (Strava/Garmin/export)",type=["csv","txt"],accept_multiple_files=True)
    if uploads and st.button("Aggiungi file al dataset",use_container_width=True):
        parts=[]
        for f in uploads:
            try: parts.append(read_uploaded_csv(f.getvalue(),f.name))
            except Exception as e: st.error(str(e))
        if parts:
            st.session_state.runs=pd.concat([st.session_state.runs,*parts],ignore_index=True)
            st.session_state["fit_requested"] = False
            st.success(f"Aggiunte {sum(len(p) for p in parts)} righe")
            st.rerun()

    st.subheader("Inserimento manuale")
    with st.form("manual"):
        date=st.date_input("Data")
        distance=st.number_input("Distanza (km)",min_value=0.0,step=.1)
        pace=st.text_input("Passo medio (es. 5:20)")
        hr=st.number_input("FC media (bpm)",min_value=0,step=1)
        duration=st.text_input("Durata (hh:mm:ss, opz.)")
        elev=st.number_input("Dislivello (m, opz.)",min_value=0.0,step=10.0)
        power=st.number_input("Potenza (W, opz.)",min_value=0.0,step=1.0)
        cadence=st.number_input("Cadenza (spm, opz.)",min_value=0.0,step=1.0)
        rpe=st.number_input("RPE (opz.)",min_value=0.0,max_value=10.0,step=.5)
        submit=st.form_submit_button("Aggiungi corsa")
        if submit:
            row=pd.DataFrame([{"date":date,"duration_min":duration or np.nan,"distance_km":distance or np.nan,
                               "pace_min_km":pace or np.nan,"avg_hr_bpm":hr or np.nan,"elevation_m":elev or np.nan,
                               "power_w":power or np.nan,"cadence_spm":cadence or np.nan,"rpe":rpe or np.nan}])
            row=canonicalize_dataframe(row)
            st.session_state.runs=pd.concat([st.session_state.runs,row],ignore_index=True)
            st.session_state["fit_requested"] = False
            st.rerun()


st.header("Dataset estratto / revisione")
if st.session_state.extraction_notes:
    with st.expander("Note dell'estrazione screenshot", expanded=True):
        for note in st.session_state.extraction_notes:
            st.warning(note)
if st.session_state.runs.empty:
    st.info("Importa un CSV o aggiungi manualmente una corsa. L'analisi parte solo quando premi “Conferma dataset e rifitta da zero”.")
    st.stop()

edited=st.data_editor(st.session_state.runs, num_rows="dynamic", use_container_width=True,
                      column_config={"date":st.column_config.DateColumn("Data",format="DD/MM/YYYY")},key="editor")
st.session_state.runs=edited

dups=detect_duplicates(edited)
if dups.any(): st.warning(f"Possibili duplicati: {int(dups.sum())} righe evidenziabili tramite data+distanza+durata. Correggili prima del fit.")

c1,c2,c3=st.columns(3)
with c1:
    csv=edited.to_csv(index=False).encode()
    st.download_button("Scarica dataset revisionato",csv,"running_dataset.csv","text/csv",use_container_width=True)
with c2:
    fit_now=st.button("Conferma dataset e rifitta da zero",type="primary",use_container_width=True)
with c3:
    if st.button("Azzera dataset",use_container_width=True):
        st.session_state.runs=pd.DataFrame(columns=CANONICAL_COLUMNS)
        st.session_state.extraction_notes=[]
        st.session_state["fit_requested"] = False
        st.rerun()

if fit_now:
    st.session_state["fit_requested"]=True
if not st.session_state.get("fit_requested"):
    st.stop()

model_df=add_engineered_columns(edited)
if "pace_derived" in model_df.columns and model_df["pace_derived"].any():
    n_derived = int(model_df["pace_derived"].sum())
    st.info(f"Per {n_derived} corse senza passo esplicito, il motore ha derivato il passo come durata/distanza dopo la conferma del dataset. Il valore non viene presentato come estratto dallo screenshot/CSV.")
try:
    fit=fit_efficiency_model(model_df)
except Exception as e:
    st.error(f"Fit non eseguibile: {e}")
    st.stop()

forecast=fit_forecast_candidates(fit)
validation=temporal_holdout_forecast_validation(fit)

st.divider(); st.header("Modello rifittato da zero")
refs=fit.refs
ref_text=f"FC {refs['avg_hr_bpm']:.0f} bpm"
if "distance_km" in refs: ref_text+=f", distanza {refs['distance_km']:.2f} km"
if "elevation_m" in refs: ref_text+=f", dislivello {refs['elevation_m']:.0f} m"
st.info(f"**Definizione numerica dell'efficienza:** `v_eq` è la velocità osservata riportata statisticamente alle condizioni mediane del dataset ({ref_text}). È un punto di efficienza statistica, non running economy misurata in laboratorio.")

m1,m2,m3,m4=st.columns(4)
m1.metric("Corse usate",len(fit.data)); m2.metric("MAE fit",f"{fit.diagnostics['mae_speed_kmh']:.2f} km/h")
m3.metric("Anomalie robuste",int(fit.anomalies.sum())); m4.metric("Trend scelto",fit.trend_kind)

st.subheader("Coefficienti")
coef_df=pd.DataFrame({"termine":list(fit.coefficients.keys()),"coefficiente":list(fit.coefficients.values())})
st.dataframe(coef_df,use_container_width=True,hide_index=True)

st.subheader("Parsimonia e diagnostica")
p1,p2=st.columns(2)
with p1:
    st.json({"stress_test":fit.parsimony,"nonlinearita_FC":fit.nonlinear_hr})
with p2:
    st.json({"diagnostica":fit.diagnostics,"validazione_temporale_forecast":validation})

if fit.parsimony["decision"]=="simple":
    st.success("Stress test di parsimonia: distanza/dislivello non migliorano abbastanza la validazione temporale; è stato preferito il modello più semplice.")
else:
    st.write("Stress test di parsimonia: il modello con covariate disponibili è stato mantenuto perché la versione più semplice peggiorava la validazione.")

st.plotly_chart(raw_over_time(fit.data),use_container_width=True)
st.plotly_chart(speed_hr(fit.data),use_container_width=True)
st.plotly_chart(normalized_trend(fit),use_container_width=True)
st.plotly_chart(residuals(fit),use_container_width=True)

st.subheader("Punti pesati meno / anomalie")
flag=fit.data[(fit.data.anomaly) | (fit.data.robust_weight<0.8)][["date","distance_km","pace_min_km","avg_hr_bpm","v_eq_kmh","eq_residual_kmh","robust_z","robust_weight","anomaly"]]
if flag.empty: st.write("Nessun punto supera i criteri robusti correnti; nessuna osservazione eliminata automaticamente.")
else: st.dataframe(flag,use_container_width=True,hide_index=True)
st.caption("Le anomalie non vengono eliminate automaticamente: RLM le downweighta. Per dati sospetti, confronta sempre il risultato con e senza il punto solo se esiste anche una ragione indipendente per ritenerlo errato.")

st.header("Forecast")
if forecast.get("status")!="ok":
    st.warning("Campione insufficiente o forecast saturante non identificabile.")
else:
    pars=forecast["saturating_params"]
    f1,f2,f3,f4=st.columns(4)
    f1.metric("g₀",f"{pars['g0']:.2f} km/h"); f2.metric("g∞",f"{pars['ginf']:.2f} km/h"); f3.metric("k",f"{pars['k']:.5f}/giorno")
    f4.metric("τ = 1/k",f"{pars['tau_days']:.0f} giorni")
    st.caption("τ è la scala temporale del modello saturante: dopo circa τ giorni è stato percorso ~63% del cambiamento tra g₀ e g∞. g∞ è solo il plateau implicato dai dati/modello, non un limite genetico.")
    st.write("Confronto alternative (MAE in-sample, km/h):",forecast["candidate_mae"])

    horizon_years=st.slider("Orizzonte grafico (anni)",1,5,3)
    last=fit.data.date.max(); end=last+pd.DateOffset(years=horizon_years)
    dates=pd.date_range(fit.data.date.min(),end,periods=350)
    days=(dates-fit.data.date.min()).days.to_numpy()
    with st.spinner("Bootstrap + refit completo del modello predittivo..."):
        boot=bootstrap_forecast(fit,days,n_boot=150)
    st.plotly_chart(forecast_chart(fit,dates,days,boot,forecast),use_container_width=True)
    if boot:
        plo,phi=boot["param_lo"],boot["param_hi"]
        st.write(f"Bootstrap riusciti: {boot['n_success']}. IC95% parametri saturanti: g∞ [{plo[0]:.2f}, {phi[0]:.2f}] km/h; g₀ [{plo[1]:.2f}, {phi[1]:.2f}] km/h; k [{plo[2]:.5f}, {phi[2]:.5f}].")
        if phi[0]-plo[0] > 2:
            st.warning("Il plateau è poco identificato: l'intervallo di g∞ è ampio. Evita di interpretare una singola stima puntuale come precisa.")

    st.header("Simulatori")
    tab1,tab2=st.tabs(["Passo sostenibile in data X","Quando raggiungo un passo?"])
    with tab1:
        c1,c2,c3,c4=st.columns(4)
        qdate=c1.date_input("Data futura",value=(last+pd.Timedelta(days=90)).date(),key="qdate")
        qdist=c2.number_input("Distanza (km)",min_value=.1,value=float(refs.get("distance_km",10.0)),key="qdist")
        qhr=c3.number_input("FC media (bpm)",min_value=50,value=int(refs["avg_hr_bpm"]),key="qhr")
        qelev=c4.number_input("Dislivello (m)",min_value=0.0,value=float(refs.get("elevation_m",0.0)),key="qelev")
        veq=predict_equivalent_on_date(fit,pd.Timestamp(qdate),forecast)
        speed=predict_speed_conditions(fit,veq,qhr,qdist,qelev)
        st.metric("Passo stimato",pace_str_from_speed(speed)); st.caption(f"Velocità stimata {speed:.2f} km/h. È una previsione condizionata al modello, non una garanzia di prestazione.")
    with tab2:
        c1,c2,c3,c4=st.columns(4)
        pace_target=c1.text_input("Passo target (es. 4:30)",value="4:30")
        dist_target=c2.number_input("Distanza target (km)",min_value=.1,value=float(refs.get("distance_km",10.0)),key="td")
        hr_target=c3.number_input("FC scenario (bpm)",min_value=50,value=int(refs["avg_hr_bpm"]),key="th")
        elev_target=c4.number_input("Dislivello scenario (m)",min_value=0.0,value=float(refs.get("elevation_m",0.0)),key="te")
        from data_utils import parse_pace
        pp=parse_pace(pace_target)
        if np.isfinite(pp):
            target_speed=60/pp
            ans=target_date_for_speed(fit,target_speed,hr_target,dist_target,elev_target,forecast)
            if ans.get("reachable"):
                st.success(f"Data centrale implicata dal modello: **{pd.Timestamp(ans['date']).strftime('%d/%m/%Y')}**.")
            elif ans.get("reason")=="oltre_asintoto_modello":
                st.warning("Con il plateau stimato dal modello saturante, l'obiettivo non viene raggiunto nelle condizioni selezionate. Non viene inventata una data.")
            else: st.warning(f"Data non stimabile in modo affidabile: {ans.get('reason')}.")
        st.caption("Per gare o prestazioni massimali non usare automaticamente la FC delle easy/moderate run: inserisci una FC scenario plausibile oppure aggiungi dati di gara/test.")

st.divider()
st.subheader("Limiti")
st.write("Modello osservazionale: FC, distanza e dislivello non rendono perfettamente comparabili terreno, meteo, vento, fatica, percorso e qualità del sensore. Campioni piccoli limitano non-linearità e plateau. Potenza, cadenza, RPE e altre variabili restano nel dataset ma non entrano automaticamente nel modello per evitare overfitting. Ogni pressione del pulsante di conferma rifitta l'intera pipeline da zero.")
