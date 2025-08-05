#%% Importando bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#%% Carregando os dados

df = pd.read_csv('gold_price_data.csv')
df

#%% Estatísticas e informações básicas

df.shape

df.info()

df.describe()

#%% Mundando o nome das colunas e alterando o index para datetime

df.rename(columns={'Value' : 'Price'}, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df

#%% Visualização da base

plt.figure(figsize=(12,8))
df.plot()
plt.title('Gold Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.show()

#%% Verificando tendência e sazonalidade 

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['Price'], period=365)
decomposition.plot()
plt.show()

#%% feature engineering

# Target do modelo

df['target'] = df['Price'].shift(-1)

# Variáveis baseadas no período de tempo

df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['dayofyear'] = df.index.dayofyear
df['year'] = df.index.year
df

# Criação de lags, médias móveis, desvios padraões móveis 

lags_days = [1, 7 , 14, 30, 90, 180, 365, 730]

for lags in lags_days:
  df[f'Lag_{lags}'] = df['Price'].shift(lags)


rolling_means = [7, 14, 30, 90, 365]

for rm in rolling_means:
  df[f'rm_{rm}'] = df['Price'].shift(1).rolling(window=rm).mean()


rolling_stds = [30, 90]

for std in rolling_stds:
  df[f'r_std_{std}'] = df['Price'].shift(1).rolling(window=std).std()


df['return_lag_1'] = (df['Price'].pct_change() * 100).shift(1)

df.dropna(inplace=True)

df.shape

df = df.sort_index()

#%% Separando em treino e teste

X = df.drop(['Price', 'target'], axis=1)
y = df['target']

divisor = int(len(df) * 0.9)

X_train, X_test = X.iloc[:divisor], X.iloc[divisor:]
y_train, y_test = y.iloc[:divisor], y.iloc[divisor:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#%% Importando as bibliotecas de modelagem de MLs

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

#%% Fazendo o cross validation

tscv = TimeSeriesSplit(n_splits=5)

models = {'Random Forest' : RandomForestRegressor(n_estimators=100, random_state=42),
          'Xgboost' : XGBRegressor(n_estimators=100, random_state=42)}

for name, model in models.items():
  scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
  root_scores = np.sqrt(-scores)
  print(f'{name} RMSE: {np.mean(root_scores)}')

#%% Instanciando e treinando os modelos

rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

#%% Avaliando na base de treino

y_pred_train_rf = rf.predict(X_train)
y_pred_train_xgb = xgb.predict(X_train)


print('Random Forest')
print('MAE: ', mean_absolute_error(y_train, y_pred_train_rf))
print('RMSE: ', root_mean_squared_error(y_train, y_pred_train_rf))

print('Xgboost')
print('MAE: ', mean_absolute_error(y_train, y_pred_train_xgb))
print('RMSE: ', root_mean_squared_error(y_train, y_pred_train_xgb))

plt.figure(figsize=(12,8))
plt.plot(y_train.index, y_train, label='Actual')
plt.plot(y_train.index, y_pred_train_rf, label='Predicted RF')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(y_train.index, y_train, label='Actual')
plt.plot(y_train.index, y_pred_train_xgb, label='Predicted XGB')
plt.legend()
plt.show()

#%% Avaliando na base de teste
y_pred_test_rf = rf.predict(X_test)
y_pred_test_xgb = xgb.predict(X_test)


print('Random Forest')
print('MAE: ', mean_absolute_error(y_test, y_pred_test_rf))
print('RMSE: ', root_mean_squared_error(y_test, y_pred_test_rf))

print('Xgboost')
print('MAE: ', mean_absolute_error(y_test, y_pred_test_xgb))
print('RMSE: ', root_mean_squared_error(y_test, y_pred_test_xgb))

plt.figure(figsize=(12,8))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_test_rf, label='Predicted RF')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_test_xgb, label='Predicted XGB')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df.index, df['Price'], label='Actual')
plt.plot(y_test.index, y_pred_test_rf, label='Predicted RF')
plt.plot(y_test.index, y_pred_test_xgb, label='Predicted XGB')
plt.legend()
plt.show()

#%% Random Seachr Random forest 

params_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                               params_rf,
                               n_iter=30,
                               scoring='neg_mean_squared_error',
                               cv=tscv,
                               n_jobs=-1,
                               random_state=42)

rf_search.fit(X_train, y_train)

print('Melhores parâmetros: ', rf_search.best_params_)
print('Melhor score: ', np.sqrt(-rf_search.best_score_))

#%% Avaliando Random forest

rf_tunned = rf_search.best_estimator_

y_pred_test_rf_tunned = rf_tunned.predict(X_test)

print('Random Forest Tunned')
print('MAE: ', mean_absolute_error(y_test, y_pred_test_rf_tunned))
print('RMSE: ', root_mean_squared_error(y_test, y_pred_test_rf_tunned))

#%% Random search XGB

param_xgb = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}


xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42),
                               param_xgb,
                               n_iter=30,
                               scoring='neg_mean_squared_error',
                               cv=tscv,
                               n_jobs=-1,
                               random_state=42)

xgb_search.fit(X_train, y_train)

print('Melhores parâmetros: ', xgb_search.best_params_)
print('Melhor score: ', np.sqrt(-xgb_search.best_score_))

#%% Avaliando o XGB

xgb_tunned = xgb_search.best_estimator_

y_pred_test_xgb_tunned = xgb_tunned.predict(X_test)

print('XGB Tunned')
print('MAE: ', mean_absolute_error(y_test, y_pred_test_xgb_tunned))
print('RMSE: ', root_mean_squared_error(y_test, y_pred_test_xgb_tunned))

#%% Exportando os resultados

results_tunned = pd.DataFrame({
    'actual' : y_test,
    'rf_pred_tunned' : y_pred_test_rf_tunned,
    'xgb_pred_tunned' : y_pred_test_xgb_tunned},
    index=y_test.index)

results_tunned.to_csv('results_tunned.csv')