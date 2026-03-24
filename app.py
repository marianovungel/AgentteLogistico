import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Configuração da Página e Tema Escuro/Verde
st.set_page_config(page_title="Score Bidimensional", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #2e7d32;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #4CAF50 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Função para carregar dados (Simulando o df_classify dos seus arquivos)
@st.cache_data
def load_data(role):
    # Substitua pelos caminhos dos seus arquivos finais processados
    if role == 'Vendedor':
        # Simulando colunas baseadas no seu notebook Vendedor.ipynb
        return pd.read_csv('df_classify_vendedor.csv') 
    else:
        # Simulando colunas baseadas no seu notebook Comprador.ipynb
        return pd.read_csv('df_classify_comprador.csv')

# --- SIDEBAR NAVEGAÇÃO ---
#page = st.sidebar.selectbox("Navegação", ["Análise Gráfica", "Vendedor", "Comprador"])

page = st.sidebar.selectbox(
    "Navegação",
    ["Análise Gráfica", "Vendedor", "Comprador", "🤖 Chatbot"]
)


# --- PÁGINA: ANÁLISE GRÁFICA ---
if page == "Análise Gráfica":
    st.title("📊 Análise Gráfica de Scores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Score - Vendedor")
        df_v = load_data('Vendedor')
        fig_v = px.histogram(df_v, x="Score", color_discrete_sequence=['#4CAF50'])
        st.plotly_chart(fig_v, use_container_width=True)
        
    with col2:
        st.subheader("Distribuição de Score - Comprador")
        df_c = load_data('Comprador')
        fig_c = px.histogram(df_c, x="Score", color_discrete_sequence=['#81C784'])
        st.plotly_chart(fig_c, use_container_width=True)

    st.divider()
    
    st.subheader("💡 Variáveis Mais Importantes")
    v_feat = ['RFR_QTD', 'TPN_QTD', 'RI_QTD_TOT', 'RR_INST_QTD', 'RR_OP_QTD', 'CRC_NUMERICA', 'Days_to_Harvest', 'mean_valuation']
    c_feat = ["mean_financeira", "mean_pagamento", "mean_juridico", "mean_reputacao", "mean_logistica", "mean_relacionamento"]
    
    c1, c2 = st.columns(2)
    c1.info(f"**Vendedor:** {', '.join(v_feat)}")
    c2.success(f"**Comprador:** {', '.join(c_feat)}")

# --- PÁGINA: VENDEDOR ---
elif page == "Vendedor":
    st.title("👨‍🌾 Painel do Vendedor")
    df_v = load_data('Vendedor')
    
    st.subheader("1° Dataset Final (Classificado)")
    st.dataframe(df_v.head(10), use_container_width=True)
    
    st.subheader("2° Carregar CSV e Classificar Novo Dataset")
    css_file = st.file_uploader("Carregue o arquivo .csv com novas variáveis", type=["csv"], key="v_upload")
    if css_file:
        new_data = pd.read_csv(css_file)
        # Lógica de aplicação do modelo classificador_vendedor.pkl
        model_v = joblib.load('classificador_vendedor.pkl')
        # Predição (exemplo simplificado)
        new_data['Score'] = model_v.predict(new_data)
        st.write("Dados Processados:")
        st.dataframe(new_data, use_container_width=True)

    st.divider()
    st.subheader("🧪 Testar Modelo de Classificação")
    with st.form("test_vendedor"):
        
        RFR_QTD = st.number_input("RFR QTD")
        TPN_QTD = st.number_input("TPN_QTD")
        RI_QTD_TOT = st.number_input("RI_QTD_TOT")
        RR_INST_QTD = st.number_input("RR_INST_QTD")
        RR_OP_QTD = st.number_input("RR_OP_QTD")
        CRC_NUMERICA = st.number_input("CRC_NUMERICA")
        Days_to_Harvest = st.number_input("Days to Harvest")
        mean_valuation = st.number_input("Mean Valuation")

        submitted = st.form_submit_button("Testar Classificação")

        if submitted:
            model_v = joblib.load('classificador_vendedor.pkl')
            # Ajustar entrada para o formato do modelo
            input_data = np.array([[RFR_QTD, TPN_QTD, RI_QTD_TOT, RR_INST_QTD, RR_OP_QTD,
                                    CRC_NUMERICA, Days_to_Harvest, mean_valuation]])
            pred = model_v.predict(input_data)
            # st.success(f"Resultado da Classificação (Score): {pred[0]}")
            st.success(f"Resultado da Classificação (Score): {pred[0]}")

# --- PÁGINA: COMPRADOR ---
elif page == "Comprador":
    st.title("🏢 Painel do Comprador")
    df_c = load_data('Comprador')
    
    st.subheader("1° Dataset Final (Classificado)")
    st.dataframe(df_c.head(10), use_container_width=True)
    
    st.subheader("2° Carregar CSS e Classificar Novo Dataset")
    css_file_c = st.file_uploader("Carregue o arquivo .csv com novas variáveis", type=["csv"], key="c_upload")
    if css_file_c:
        new_data_c = pd.read_csv(css_file_c)
        model_c = joblib.load('classificador_comprador.pkl')
        new_data_c['Score'] = model_c.predict(new_data_c)
        st.dataframe(new_data_c, use_container_width=True)

    st.divider()
    st.subheader("🧪 Testar Modelo de Classificação")
    with st.form("test_comprador"):
        fin = st.number_input("Mean Financeira")
        pag = st.number_input("Mean Pagamento")
        jur = st.number_input("Mean Jurídico")
        rep = st.number_input("Mean Reputação")
        logist = st.number_input("Mean Logística")
        rel = st.number_input("Mean Relacionamento")
        submitted = st.form_submit_button("Testar Classificação")
        if submitted:
            model_c = joblib.load('classificador_comprador.pkl')
            input_data = np.array([[fin, pag, jur, rep, logist, rel]])
            pred = model_c.predict(input_data)
            st.success(f"Resultado da Classificação (Score): {pred[0]}")
# --- PÁGINA: CHATBOT ---
elif page == "🤖 Chatbot":
    from mainChat import ask_chatbot

    st.title("🤖 Fale com o Especialista")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input do usuário
    if prompt := st.chat_input("Pergunte sobre Score, regras, variáveis, modelos..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando documentos..."):
                response = ask_chatbot(prompt)
                st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
