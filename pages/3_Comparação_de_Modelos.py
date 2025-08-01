#%% Importando Bibliotecas 
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

#%% Configuração da página

st.set_page_config('Previsão do Preço do Ouro', layout='wide')

#%% Carregando os dados

@st.cache_data
def load_data2():
    df_results = pd.read_csv('results.csv', index_col='Date')
    df_tunned = pd.read_csv('results_tunned.csv', index_col='Date')
    return df_results, df_tunned

df_results, df_tunned = load_data2()

#%% Título e apresentação 

st.header('🤖 Comparação de Modelos de Machine Learning: Random Forest vs XGBoost')

st.markdown('---') 

st.markdown("""
Ambos os modelos foram treinados com 90% da base de dados e testados nos 10% restantes, utilizando **preço do ouro de amanhã** como target.

#### 🔍 Métricas de avaliação:
- **Gráfico Real vs Previsto**
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

Serão apresentados os resultados em duas etapas:
1. **Modelos base (sem tuning)**  
2. **Modelos otimizados com RandomizedSearchCV**

Para garantir uma comparação justa, os dois modelos foram otimizados utilizando **conjuntos de hiperparâmetros equivalentes em complexidade**, respeitando as características de cada algoritmo.

""")

#%% Comparação dos modelos originais 

st.markdown('---') 

st.subheader('🔹 Modelos Originais')


rf_mae  = 13.69
rf_rmse = 19.08

xgb_mae = 19.97
xgb_rmse = 31.64



def plots(real, pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real.index, y=real, mode='lines', name='Real'))
    fig.add_trace(go.Scatter(x=real.index, y=pred, mode='lines', name='Previsto'))
    fig.update_layout(title=title, yaxis_title='Preço do Ouro ($)')
    return fig


with st.expander('Visualizar comparação'):
    
    col1, col2 = st.columns(2)
    
    with col1: 
        
        fig_rf = plots(df_results['actual'], df_results['rf_pred'],"Valores Reais x Previsões (Random Forest)")
        st.plotly_chart(fig_rf, use_container_width=True)
        
        with st.container(border=True):
            st.markdown('**📊 Métricas Random Forest:**')
            m1, m2 = st.columns(2)
            m1.metric('📉 MAE', f'{rf_mae:.2f}')
            m2.metric('📐 RMSE', f'{rf_rmse:.2f}')
    
    with col2:
        
       fig_xgb = plots(df_results['actual'], df_results['xgb_pred'],"Valores Reais x Previsões (XGBoost)")
       st.plotly_chart(fig_xgb, use_container_width=True)
       
       with st.container(border=True):
           st.markdown('**📊 Métricas XGBoost:**')
           m3, m4 = st.columns(2)
           m3.metric('📉 MAE', f'{xgb_mae:.2f}')
           m4.metric('📐 RMSE', f'{xgb_rmse:.2f}')
   
#%% Comparação dos modelos Tunados 

rf_mae_tunned  = 12.33
rf_rmse_tunned = 16.91

xgb_mae_tunned = 14.06
xgb_rmse_tunned = 20.21
    
st.subheader('🔹 Modelos Tunados')

with st.expander('Visualizar comparação'):
    
    col3, col4 = st.columns(2)
    
    with col3: 
        
        fig_rf = plots(df_tunned['actual'], df_tunned['rf_pred_tunned'],"Valores Reais x Previsões (Random Forest Tunado)")
        st.plotly_chart(fig_rf, use_container_width=True)
        
        with st.container(border=True):
            st.markdown('**📊 Métricas Random Forest Tunado:**')
            m5, m6 = st.columns(2)
            m5.metric('📉 MAE', f'{rf_mae_tunned:.2f}')
            m6.metric('📐 RMSE', f'{rf_rmse_tunned:.2f}')
    
    with col4:
        
       fig_xgb = plots(df_tunned['actual'], df_tunned['xgb_pred_tunned'],"Valores Reais x Previsões (XGBoost Tunado)")
       st.plotly_chart(fig_xgb, use_container_width=True)
       
       with st.container(border=True):
           st.markdown('**📊 Métricas XGBoost Tunado:**')
           m7, m8 = st.columns(2)
           m7.metric('📉 MAE', f'{xgb_mae_tunned:.2f}')
           m8.metric('📐 RMSE', f'{xgb_rmse_tunned:.2f}')
           
#%% Modelos Orginais x Modelos Tunados 

st.subheader('🔹Modelos Originais x Modelos Tunados') 

with st.expander('Visualizar Evolução das Métricas'):
       
    col5, col6 = st.columns(2)
    
    with col5:
        
        with st.container(border=True):
            
            st.markdown('**📊 Variação das Métricas (Random Forest)**')
            
            m9, m10 = st.columns(2) 
        
            m9.metric('📉 MAE', f'{(rf_mae_tunned - rf_mae):.2f}', delta= f'{((rf_mae_tunned/rf_mae -1) * 100):.2f}%')
            m10.metric('📐 RMSE', f'{(rf_rmse_tunned - rf_rmse):.2f}', delta= f'{((rf_rmse_tunned/rf_rmse -1) * 100):.2f}%')
            
    with col6:
               
        with st.container(border=True):
                   
            st.markdown('**📊 Variação das Métricas (XGBoost)**')
                   
            m11, m12 = st.columns(2) 
               
            m11.metric('📉 MAE', f'{(xgb_mae_tunned - xgb_mae):.2f}', delta= f'{((xgb_mae_tunned/xgb_mae -1) * 100):.2f}%')
            m12.metric('📐 RMSE', f'{(xgb_rmse_tunned - xgb_rmse):.2f}', delta= f'{((xgb_rmse_tunned/xgb_rmse -1) * 100):.2f}%')  
            
    st.success("""
    🔍 **Insights da Comparação**:
    - O **Random Forest**, que já apresentava bons resultados, teve melhora leve mas consistente com o tuning.
    - O **XGBoost** teve um **ganho expressivo**, reduzindo significativamente os erros (especialmente RMSE).
    - A escolha do modelo depende do objetivo: RF pode ser mais estável, enquanto o XGB pode surpreender com o ajuste certo.
    """)
            
            # Gráficos dos resíduos dos modelos 
    st.markdown('---')        
    
    col7, col8 = st.columns(2)
    
    df_errors = df_results.copy()
    df_errors['erro_rf'] = df_errors['actual'] - df_errors['rf_pred']
    df_errors['erro_xgb'] = df_errors['actual'] - df_errors['xgb_pred']
    
    with col7:
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
        x=df_errors.index,
        y=df_errors['erro_rf'],
        mode='markers',
        name='Random Forest',
        marker=dict(color='royalblue', size=6),
        opacity=0.7))

        fig.add_trace(go.Scatter(
        x=df_errors.index,
        y=df_errors['erro_xgb'],
        mode='markers',
        name='XGBoost',
        marker=dict(color='firebrick', size=6),
        opacity=0.7
        ))

        fig.update_layout(
        title='Erros dos Modelos ao Longo do Tempo (Modelos Originais)',
        xaxis_title='',
        yaxis_title='Erro (Valor Real - Previsto)',
        legend=dict(x=0.01, y=0.99),
        height=450
        )

        st.plotly_chart(fig, use_container_width=True)
    
    df_errors_tunned = df_tunned.copy()
    df_errors_tunned['erro_rf_tunned'] = df_errors_tunned['actual'] - df_errors_tunned['rf_pred_tunned']
    df_errors_tunned['erro_xgb_tunned'] = df_errors_tunned['actual'] - df_errors_tunned['xgb_pred_tunned']   
    
    with col8: 
        fig = go.Figure()

        fig.add_trace(go.Scatter(
        x=df_errors_tunned.index,
        y=df_errors_tunned['erro_rf_tunned'],
        mode='markers',
        name='Random Forest',
        marker=dict(color='royalblue', size=6),
        opacity=0.7))

        fig.add_trace(go.Scatter(
        x=df_errors_tunned.index,
        y=df_errors_tunned['erro_xgb_tunned'],
        mode='markers',
        name='XGBoost',
        marker=dict(color='firebrick', size=6),
        opacity=0.7
        ))

        fig.update_layout(
        title='Erros dos Modelos ao Longo do Tempo (Modelos Tunados)',
        xaxis_title='',
        yaxis_title='Erro (Valor Real - Previsto)',
        legend=dict(x=0.01, y=0.99),
        height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info("""
            🔍 **Interpretação**:
            - Pontos mais distantes da linha zero indicam maiores erros.
            - Podemos identificar momentos em que os modelos se desviaram mais.
            - O tuning impactou positivamente na disperção dos erros.

            """)
