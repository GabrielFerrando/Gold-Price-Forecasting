# Gold-Price-Forecasting

# 📈 Previsão do Preço do Ouro com Machine Learning

Este projeto tem como objetivo prever o preço do ouro utilizando modelos de machine learning baseados em árvores: **Random Forest** e **XGBoost**.

O foco foi explorar o comportamento da série temporal e criar um modelo preditivo **sem o uso de variáveis externas**, utilizando apenas informações derivadas da própria série histórica.

---

## 🧠 Modelos Utilizados

- **Random Forest Regressor**
- **XGBoost Regressor**

Ambos os modelos foram comparados em sua forma padrão e após tuning de hiperparâmetros com **RandomizedSearchCV** e validação cruzada.

---

## 🛠️ Técnicas e Features

- **Lags temporais** (valores anteriores do preço)
- **Médias móveis** (curto e médio prazo)
- **Variáveis derivadas da data** (dia da semana, semana do ano, mês, etc.)
- **Tuning de hiperparâmetros**
- **Validação cruzada (TimeSeriesSplit)**
- Visualizações com **Plotly**
- Interface web interativa com **Streamlit**

---

## 📊 Resultados

### 🔹 Random Forest
| Métrica | Antes do Tuning | Após Tuning |
|--------|------------------|-------------|
| MAE    | 13.69               | 12.33          |
| RMSE   | 19.08               | 16.91          |

### 🔹 XGBoost
| Métrica | Antes do Tuning | Após Tuning |
|--------|------------------|-------------|
| MAE    | 19.97               | 14.06          |
| RMSE   | 31.64               | 20.21          |

Os modelos mostraram **melhorias significativas após o tuning**, principalmente na redução do erro médio.

---

## 🚀 Deploy

Você pode acessar a versão interativa do projeto no Streamlit:

👉 [Acessar o app no Streamlit](https://gold-price-forecasting-project.streamlit.app/)

---

