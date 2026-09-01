from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.express import scatter


def raw_over_time(d):
    fig=go.Figure()
    fig.add_scatter(x=d.date,y=d.pace_min_km,mode="markers+lines",name="Passo grezzo",customdata=d[["speed_kmh"]],hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} min/km<extra></extra>")
    fig.update_yaxes(autorange="reversed",title="Passo (min/km)")
    fig.update_xaxes(title="Data")
    fig.update_layout(title="Prestazioni grezze nel tempo")
    return fig


def speed_hr(d):
    fig=scatter(d,x="avg_hr_bpm",y="speed_kmh",hover_data=["date","distance_km","pace_min_km"],trendline=None)
    fig.update_layout(title="Relazione velocità–FC",xaxis_title="FC media (bpm)",yaxis_title="Velocità (km/h)")
    return fig


def normalized_trend(fit):
    d=fit.data
    fig=go.Figure()
    normal=~d.anomaly
    fig.add_scatter(x=d.loc[normal,"date"],y=d.loc[normal,"v_eq_kmh"],mode="markers",name="Punti di efficienza")
    fig.add_scatter(x=d.loc[~normal,"date"],y=d.loc[~normal,"v_eq_kmh"],mode="markers",name="Anomalie",marker_symbol="x",marker_size=11)
    fig.add_scatter(x=d.date,y=fit.trend_unconstrained,mode="lines",name="Trend flessibile")
    fig.add_scatter(x=d.date,y=fit.trend_isotonic,mode="lines",name="Trend isotono",line_dash="dash")
    fig.update_layout(title="Efficienza normalizzata e trend",xaxis_title="Data",yaxis_title="Velocità equivalente (km/h)")
    return fig


def residuals(fit):
    d=fit.data
    fig=go.Figure()
    fig.add_scatter(x=d.date,y=d.eq_residual_kmh,mode="markers",name="Residui eq.",marker_symbol=np.where(d.anomaly,"x","circle"))
    fig.add_hline(y=0,line_dash="dash")
    fig.update_layout(title="Residui e anomalie",xaxis_title="Data",yaxis_title="Residuo (km/h)")
    return fig


def forecast_chart(fit, future_dates, future_days, boot, forecast):
    d=fit.data
    fig=go.Figure()
    fig.add_scatter(x=d.date,y=d.v_eq_kmh,mode="markers",name="Osservato")
    if boot:
        fig.add_scatter(x=future_dates,y=boot["hi"],mode="lines",line_width=0,showlegend=False,hoverinfo="skip")
        fig.add_scatter(x=future_dates,y=boot["lo"],mode="lines",fill="tonexty",line_width=0,name="IC bootstrap 95%")
        fig.add_scatter(x=future_dates,y=boot["median"],mode="lines",name="Forecast saturante")
    last=d.date.max()
    fig.add_vline(x=last.timestamp()*1000,line_dash="dash")
    fig.update_layout(title="Osservato vs previsione",xaxis_title="Data",yaxis_title="Velocità equivalente (km/h)")
    return fig
