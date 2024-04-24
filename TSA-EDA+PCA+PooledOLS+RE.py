# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:31:57 2021

@author: ak000
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib import rcParams
from statsmodels.graphics import tsaplots
import statsmodels.tsa.stattools as stools
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.tsa.seasonal as seasonal

warnings.filterwarnings("ignore")
rcParams["figure.figsize"] = 24, 20

data = pd.read_csv("nifty500_5yrs_unclean/3MINDIA.csv")

print(data.shape)
#print(data.head())
data.loc[:, 'timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp', drop=True)
data = data.sort_index()

data['Ct-1'] = data['adjusted_close'].shift(1)
data['Ot-1'] = data['open'].shift(1)
data['Lt-1'] = data['low'].shift(1)
data['Ht-1'] = data['high'].shift(1)

#data['close_diff'] = data['adjusted_close'].diff()
data_r = pd.DataFrame()

data.reset_index(inplace=True)
data = data.drop(0)
data.reset_index(drop = True, inplace = True)

data.loc[:, 'Year'] = data["timestamp"].dt.year
data.loc[:, 'Month'] = data['timestamp'].dt.month

data_r['timestamp'] = data['timestamp']
data_r.loc[:, 'Close'] = np.log(data['adjusted_close']/data['Ct-1'])
data_r.loc[:, 'Open'] = np.log(data['open']/data['Ot-1'])
data_r.loc[:, 'High'] = np.log(data['high']/data['Ht-1'])
data_r.loc[:, 'Low'] = np.log(data['low']/data['Lt-1'])
data_r['VoI'] = data_r['Close'].shift(-1)
data_r['CtoO'] = np.log(data['close']/data['open'])

data_r = data_r.drop(data_r.shape[0]-1)

data_w = data_r.drop(['timestamp'], axis = 1)
data_i = data_r.set_index('timestamp', drop=True)

corrs_all = data_w.corr()

cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
mask = np.triu(np.ones_like(corrs_all, dtype=bool))

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrs_all, mask=mask, cmap=cmap, square=True, annot=True, ax=ax)
plt.show()

fig = tsaplots.plot_acf(data_r['Close'], lags=30)
plt.xlabel("Lag at k")
plt.ylabel("Correlation coefficient")
plt.show()
    
fig = tsaplots.plot_pacf(data_r['Close'], lags=30)
plt.xlabel("Lag at k")
plt.ylabel("Correlation coefficient")
plt.show()

def kpss_test(timeseries):
    kpsstest = stools.kpss(timeseries, regression="c",nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value

    if kpss_output["p-value"] >= 0.05:
        return "Yes"
    else:
        return "No"

def adf_test(timeseries):
    dftest = stools.adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    
    if dfoutput["p-value"] >= 0.05:
        return "No"
    else:
        return "Yes"
    
station_test = pd.DataFrame()

for name in data_w.columns.values.tolist():
    station_test.loc[name, "ADF"] = adf_test(data_r[name])
    station_test.loc[name, "KPSS"] = kpss_test(data_r[name])

print(station_test)

seasonality_data = {}
trend_data = {}
residual_data = {}
decomp_freq = 6

for name in data_i.columns.values.tolist():
    decomp = seasonal.seasonal_decompose(data_i[name], model = "additive", period= decomp_freq)
    seasonality_data[name] = decomp.seasonal
    trend_data[name] = decomp.trend
    residual_data[name] = decomp.resid

pd.DataFrame(seasonality_data)["2006-08-07":"2006-09-08"].plot(subplots = True, layout = (3 , 6), linewidth=1)
pd.DataFrame(trend_data).plot(subplots = True, layout = (3 , 6), linewidth=1)
#pd.DataFrame(residual_data)["2019-08-07":"2019-09-07"].plot(subplots = True, layout = (3 , 6), linewidth=1)
plt.show()

trend_data = pd.DataFrame(trend_data)
trend_data = trend_data.iloc[3:-3]
seasonality_data = pd.DataFrame(seasonality_data)

X_train, X_test, y_train, y_test = data_i.drop('VoI', axis=1).iloc[:int(0.7*(data.shape[0]))], data_i.drop('VoI', axis=1).iloc[int(0.7*(data.shape[0])):], data_i['VoI'].iloc[:int(0.7*(data.shape[0]))], data_i['VoI'].iloc[int(0.7*(data.shape[0])):]

std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

pca = PCA(n_components = X_train_std.shape[1])
pca_data = pca.fit_transform(X_train_std)

percent_var_explained = pca.explained_variance_/(np.sum(pca.explained_variance_))
cumm_var_explained = np.cumsum(percent_var_explained)

plt.plot(cumm_var_explained)
plt.grid()
plt.xlabel("n_components")
plt.ylabel("% variance explained")
plt.show()

pca = PCA(n_components=3)
pca_train_data = pca.fit_transform(X_train_std)
pca_test_data = pca.transform(X_test_std)

df_train_pca = pd.DataFrame(pca_train_data)
df_train_pca['Month'] = data['Month'].iloc[:int(0.7*(data.shape[0]))]
df_train_pca['Year'] = data['Year'].iloc[:int(0.7*(data.shape[0]))]

y_train = pd.DataFrame(y_train)
y_train.reset_index(drop = False, inplace = True)

y_train.loc[:, 'Year'] = y_train["timestamp"].dt.year
y_train.loc[:, 'Month'] = y_train['timestamp'].dt.month
timestamp = y_train['timestamp']

df_train_pca.set_index(['Year','Month'], drop=True, inplace=True)
y_train = y_train.drop('timestamp', axis = 1).set_index(['Year','Month'], drop=True)

y_test = pd.DataFrame(y_test)
y_test.reset_index(drop = False, inplace = True)

y_test.loc[:, 'Year'] = y_test["timestamp"].dt.year
y_test.loc[:, 'Month'] = y_test['timestamp'].dt.month

df_test_pca = pd.DataFrame(pca_test_data)
df_test_pca.loc[:, 'Month'] = y_test['Month']
df_test_pca.loc[:, 'Year'] = y_test['Year']
df_test_pca.set_index(['Year','Month'], drop=True, inplace=True)

y_test = y_test.drop('timestamp', axis = 1).set_index(['Year','Month'], drop=True)

corrs_pca = df_train_pca.corr()

cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
mask = np.triu(np.ones_like(corrs_pca, dtype=bool))

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrs_pca, mask=mask, cmap=cmap, square=True, annot=True, ax=ax)
plt.show()

station_test_pca = pd.DataFrame()

for name in df_train_pca.columns.values.tolist():
    station_test_pca.loc[name, "ADF"] = adf_test(df_train_pca[name])
    station_test_pca.loc[name, "KPSS"] = kpss_test(df_train_pca[name])

print(station_test_pca)

seasonality_pca = {}
trend_pca = {}
residual_pca = {}

for name in df_train_pca.columns.values.tolist():
    decomp = seasonal.seasonal_decompose(df_train_pca[name], model = "additive", period= decomp_freq)
    seasonality_pca[name] = decomp.seasonal
    trend_pca[name] = decomp.trend
    residual_pca[name] = decomp.resid

pd.DataFrame(seasonality_pca).iloc[0:50].plot(subplots = True, layout = (3 , 6), linewidth=1)
pd.DataFrame(trend_pca).plot(subplots = True, layout = (3 , 6), linewidth=1)
#pd.DataFrame(residual_diff)["2019-08-07":"2019-09-07"].plot(subplots = True, layout = (3 , 6), linewidth=1)
plt.show()

from linearmodels import PooledOLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

trend_data.reset_index(drop=False, inplace=True)
trend_data.loc[:, 'Year'] = trend_data["timestamp"].dt.year
trend_data.loc[:, 'Month'] = trend_data['timestamp'].dt.month
trend_data = trend_data.drop('timestamp', axis = 1).set_index(['Year','Month'], drop=True)

traini = trend_data.drop(['VoI', 'High'], axis=1)
testi = pca_test_data

exog = sm.tools.tools.add_constant(traini)
endog = pd.DataFrame(trend_data).loc[: , 'VoI']

mod = PooledOLS(endog, exog)
pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids
    
fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 15)
ax.set_ylabel('Residuals', fontsize = 15)
ax.set_title('Homoskedasticity Test', fontsize = 30)
plt.show()
    
#pooled_OLS_dataset = pd.concat([train, residuals_pooled_OLS], axis=1)
    
white_test_results = het_white(residuals_pooled_OLS, exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, white_test_results)))
 
breusch_pagan_test_results = het_breuschpagan(residuals_pooled_OLS, exog)
labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
print(dict(zip(labels, breusch_pagan_test_results)))
    
durbin_watson_test_results = durbin_watson(residuals_pooled_OLS) 
print(durbin_watson_test_results)

from linearmodels import PanelOLS
from linearmodels import RandomEffects
 
model_re = RandomEffects(endog, exog) 
re_res = model_re.fit() 

model_fe = PanelOLS(endog, exog, time_effects=True) 
fe_res = model_fe.fit() 

print(re_res.summary)
print(fe_res.summary)

print(re_res.predict(sm.tools.tools.add_constant(df_test_pca)))

# RE model predict() method does not work in Python. Its implementation is bugged. I will save the datframes from here to process them in R.

import feather
feather.write_dataframe(df_train_pca, 'pca_train_data.feather')
feather.write_dataframe(df_test_pca, 'pca_test_data.feather')
feather.write_dataframe(y_train, 'y_train.feather')
feather.write_dataframe(y_test, 'y_test.feather')

data_r.to_csv('pca_train_data.csv')
df_test_pca.to_csv('pca_test_data.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')
timestamp.to_csv('timestamp.csv')
data_r.to_csv('pca_train_data.csv')
trend_data.to_csv('trend_data.csv')
seasonality_data.to_csv('seas_data.csv')