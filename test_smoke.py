import pandas as pd
from data_utils import canonicalize_dataframe, add_engineered_columns
from model import fit_efficiency_model, fit_forecast_candidates, bootstrap_forecast

df=pd.read_csv('example_runs.csv')
df=canonicalize_dataframe(df)
x=add_engineered_columns(df)
fit=fit_efficiency_model(x)
fc=fit_forecast_candidates(fit)
assert len(fit.data)==14
assert 'avg_hr_bpm' in fit.refs
assert fc['status']=='ok'
print('OK', fit.trend_kind, fit.parsimony, fc['candidate_mae'])
