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
def load_data():
    df = pd.read_csv('C:/Users/ferra/Projetos Treino/Gold Price Forecasting Project/df_base.csv', index_col='Date')
    return df

df = load_data()

#%% 

st.header('📊 Visualização dos Dados Base')
st.markdown('Nesta seção, apresento os dados originais utilizados para treinar os modelos.')
st.markdown("""
            A base é composta por varáveis derivadas do próprio preço do ouro ao longo do tempo:
            
            - **Preço diário do ouro** (1981 - 2020)
            - **Variáveis Temporais** (ex: dia da semana, mês, etc...)
            - **Lags**: valores anteriores (ex: 1, 3, 7 dias antes)
            - **Médias móveis**: suavizações de curto, médio e longo prazo
            
            O objetivo foi capturar padrãoes sem utilização de variáveis externas.
                """)

#%% Exibindo o DataFrame

st.markdown('---')

st.subheader('🧾 Amostra dos Dados')

with st.expander('Visualizar Dados'):
    st.dataframe(df.drop('target', axis=1).head(10))
    
#%% Estatísticas Descritivas

st.markdown('---')

st.subheader('📌 Estatísticas Descritivas')

with st.expander('Visualizar Estatísticas'):
    st.dataframe(df.drop('target', axis=1).describe().T)
    
#%% Evolução do preço do ouro 

st.markdown('---')

st.subheader('📈 Evolução do Preço do Ouro (1981 - 2020)')

with st.expander('Visualizar Gráfico'):
    fig_line = px.line(df, x=df.index, y='Price', labels={'Date': "", 'Price': 'Preço do Ouro ($)'})
 
    st.plotly_chart(fig_line)

#%% Correlação entre vasriáveis

st.markdown('---')

st.subheader('🔗 Correlação entre Variáveis')

with st.expander('Visualizar Mapa'):
    fig_corr = px.imshow(df.corr(), text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto')
    st.plotly_chart(fig_corr, use_container_width=True)
    
#%%% Distribução do preço do ouro

st.markdown('---')

st.subheader('📊 Distribuição do preço do ouro')

with st.expander('Visualizar Histograma'):
    fig_hist = px.histogram(df, x='Price', nbins=50)
    st.plotly_chart(fig_hist, use_container_width=True)









