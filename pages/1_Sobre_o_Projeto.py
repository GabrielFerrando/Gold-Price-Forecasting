#%% Importando bibliotecas

import pandas as pd
import streamlit as st
import plotly.express as px

#%% Configuração da página

st.set_page_config('Previsão do Preço do Ouro', layout='wide')

#%% Título da página 

st.title('Previsão do Preço do Ouro com Ensemble Models')

st.subheader('📌 Sobre o Projeto')

st.markdown("""
    A previsão de preços de ativos financeiros é um desafio clássico da ciência de dados — e o ouro, por ser um ativo de proteção global, é particularmente interessante nesse contexto.

    Neste projeto, propus um experimento simples, mas poderoso: **prever o preço do ouro utilizando apenas informações históricas da própria série**, como:

    - **Lags temporais** (valores anteriores do preço)
    - **Médias móveis** de diferentes períodos (7, 14, 90 e 365 dias)

    O objetivo foi explorar o quanto é possível aprender **sem depender de variáveis externas** como inflação, juros ou câmbio.

    ---
    ### 🔍 Modelos utilizados
    Para esse experimento, utilizei dois modelos supervisionados:

    - **Random Forest Regressor**: modelo baseado em árvores de decisão em ensemble
    - **XGBoost Regressor**: algoritmo gradient boosting eficiente, muito usado em competições de machine learning

    Ambos foram treinados e avaliados sobre o mesmo conjunto de dados, e os resultados são apresentados em gráficos e métricas comparativas nas abas seguintes.

    ---
    ### 🎯 Por que isso importa?
    Este tipo de abordagem pode ser aplicada a outros ativos financeiros, commodities ou até séries operacionais (como demanda, preços agrícolas etc.).

    Mais do que prever com precisão absoluta, o foco está em **entender padrões**, **testar hipóteses** e **refinar a capacidade preditiva com poucos insumos**.

  
    """)
