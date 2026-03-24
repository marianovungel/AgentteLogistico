import pandas as pd
import numpy as np
import geopandas as gpd
#from posthog import page
from shapely.geometry import Point
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import os
import networkx as nx
import osmnx as ox 
import geopandas as gpd
import momepy
from scipy.spatial import KDTree
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from price import sacas




#Função para carregar os modais do IBGE
@st.cache_data
def carregar_malha_logistica():
    rod = gpd.read_file("ShapeFiles_RF/rod_trecho_rodoviario_l.shp")
    fer = gpd.read_file("ShapeFiles_RF/fer_trecho_ferroviario_l.shp")
    rod['modal'] = 'rodoviario'
    fer['modal'] = 'ferroviario'
    malha = pd.concat([rod[['geometry', 'modal']], fer[['geometry', 'modal']]], ignore_index=True)
    return malha

malha_logistica = carregar_malha_logistica()

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

#Interface de Preferência de Modal (Novo filtro na barra lateral)
st.sidebar.subheader("Preferência de Transporte")
modal_preferido = st.sidebar.selectbox("Modal Prioritário:", ["rodoviario", "ferroviario"])

#Construção do Grafo com Pesos Dinâmicos
@st.cache_resource
def construir_grafo_dinamico(_malha, preferido):
    malha_proj = _malha.to_crs(epsg=3857)
    
    G = momepy.gdf_to_nx(malha_proj, approach='primal')
    penalidade_alta = malha_proj.geometry.length.max() * 10
    for u, v, k, data in G.edges(data=True, keys=True):
        comprimento = data.get('mm_len', 0)
        modal_aresta = data.get('modal', 'rodoviario')
        
        if modal_aresta == preferido:
            data['weight'] = comprimento
        else:
            data['weight'] = comprimento + penalidade_alta
            
    return G

G_logistico = construir_grafo_dinamico(malha_logistica, modal_preferido)


#Função para encontrar o nó mais próximo no Grafo
def encontrar_no_proximo(G, ponto):
    nodes_list = list(G.nodes)
    nodes_coords = np.array(nodes_list)
    
    tree = KDTree(nodes_coords)
    dist, idx = tree.query([ponto.x, ponto.y])
    return nodes_list[idx]

#Função para traçar a rota e calcular a distância real
def calcular_rota_real_detalhada(G, origem_geom, destino_geom):
    origem_proj = gpd.GeoSeries([origem_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    destino_proj = gpd.GeoSeries([destino_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    
    node_origem = encontrar_no_proximo(G, origem_proj)
    node_destino = encontrar_no_proximo(G, destino_proj)
    
    try:
        rota_nos = nx.shortest_path(G, node_origem, node_destino, weight='weight')
        
        caminho_coords = []
        distancia_total_m = 0
        modais_na_rota = set()
        segmentos_rod = 0
        segmentos_fer = 0
        
        for i in range(len(rota_nos) - 1):
            u, v = rota_nos[i], rota_nos[i+1]
            dados_aresta = G[u][v][0]
            distancia_total_m += dados_aresta.get('mm_len', 0)
            modal = dados_aresta.get('modal', 'rodoviario')
            modais_na_rota.add(modal)
            
            if modal == 'rodoviario': segmentos_rod += 1
            else: segmentos_fer += 1
            
            p_latlon = gpd.GeoSeries([Point(u[0], u[1])], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
            caminho_coords.append((p_latlon.y, p_latlon.x))
            
        return {
            'coords': caminho_coords, 'dist_km': distancia_total_m / 1000,
            'tem_rodovia': 'rodoviario' in modais_na_rota,
            'tem_ferrovia': 'ferroviario' in modais_na_rota,
            'n_rodovia': segmentos_rod, 'n_ferrovia': segmentos_fer
        }
    except:

        return {
            'coords': [], 'dist_km': 0.0, 'tem_rodovia': False, 
            'tem_ferrovia': False, 'n_rodovia': 0, 'n_ferrovia': 0
        }

df_agr = pd.read_csv('df_agr.csv')

#Carregar o seu dataset
df = df_agr

# Criamos a coluna 'geometry' a partir da longitude e latitude
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

#Exportação para Shapefile
gdf.to_file("agricultores_agro_m2.shp")
gdf_teste = gpd.read_file("ShapeFiles/agricultores_agro_m2.shp")

# Configuração da Interface
st.title("Agro M2 - Análise Logística")

#Carregar os Shapefiles.
@st.cache_resource 
def carregar_modelo():
    return joblib.load('classificador_vendedor.pkl')

@st.cache_data
def carregar_dados():
    df_original = pd.read_csv('df_agr.csv')
    gdf = gpd.read_file("ShapeFiles/agricultores_agro_m2.shp")
    return df_original, gdf

modelo = carregar_modelo()
df_agr, gdf_agricultores = carregar_dados()

#Parâmetros de Busca (Barra Lateral)
st.sidebar.header("Parâmetros do Trader")
raio_km = st.sidebar.number_input("Raio de busca (km)", min_value=1, max_value=500, value=100)
lat = st.sidebar.number_input("Latitude do destino", min_value=-90.0, max_value=90.0, value=-25.0)
lon = st.sidebar.number_input("Longitude do destino", min_value=-180.0, max_value=180.0, value=-49.0)

# Lista de produtos disponíveis no seu dataset
lista_produtos = ['Soja', 'acucar', 'cafe', 'Milho', 'arroz', 'frutos', 'vegetais']

# Widget de seleção múltipla
produtos_selecionados = st.sidebar.multiselect(
    "Filtrar por Produtos:",
    options=lista_produtos,
    default=lista_produtos  
)

# ponto_x_coord = (-25.0, -49.0)
ponto_x_coord = (lat, lon)
st.sidebar.write(f"Destino (Ponto X): {ponto_x_coord}")

if 'ponto_x_manual' not in st.session_state:
    st.session_state.ponto_x_manual = ponto_x_coord

#Filtro por Produto.
def filtrar_por_produto(gdf, selecionados):
    if not selecionados:
        return gdf
    
    mask = gdf[selecionados].any(axis=1)
    return gdf[mask].copy()

# Aplicamos primeiro o filtro de produtos para reduzir o processamento espacial
gdf_filtrado_prod = filtrar_por_produto(gdf_agricultores, produtos_selecionados)

#Lógica de Filtragem Geográfica
def filtrar_por_raio(gdf, centro, raio_km):
    ponto_destino = Point(centro[1], centro[0]) 
    ponto_gs = gpd.GeoSeries([ponto_destino], crs="EPSG:4326")
    ponto_metros = ponto_gs.to_crs(epsg=3857)
    gdf_metros = gdf.to_crs(epsg=3857)
    distancias = gdf_metros.distance(ponto_metros.iloc[0])
    gdf['distancia_x_km'] = distancias / 1000
    return gdf[gdf['distancia_x_km'] <= raio_km].copy()

#Filtro Espacial (Aplicado sobre o resultado do filtro de produtos)
agricultores_no_raio = filtrar_por_raio(gdf_filtrado_prod, st.session_state.ponto_x_manual, raio_km)

# PROCESSAMENTO UNIFICADO: ML + LOGÍSTICA + MAPA 
if not agricultores_no_raio.empty:
    colunas_ml = ['RFR_QTD', 'TPN_QTD', 'RI_QTD_TOT', 'RR_INST_QTD', 
                  'RR_OP_QTD', 'CRC_NUMERICA', 'Days_to_Harvest', 'mean_valuation']

    top_10_distancia = agricultores_no_raio.sort_values(by='distancia_x_km').head(10).copy()
    
    lista_ids = top_10_distancia['ID'].tolist()
    dados_para_ml = df_agr[df_agr['ID'].isin(lista_ids)].set_index('ID').loc[lista_ids].reset_index()
    top_10_distancia['Score'] = modelo.predict(dados_para_ml[colunas_ml])

    top_10_distancia['Dist_Real_KM'] = 0.0
    top_10_distancia['ferrovia'] = False
    top_10_distancia['rodovia'] = False
    top_10_distancia['n_ferrovia'] = 0
    top_10_distancia['n_rodovia'] = 0


    #Criação do Objeto Mapa Único
    m = folium.Map(location=st.session_state.ponto_x_manual, zoom_start=7)
    folium.Marker(st.session_state.ponto_x_manual, tooltip="Destino", icon=folium.Icon(color='red')).add_to(m)
    folium.Circle(location=st.session_state.ponto_x_manual, radius=raio_km * 1000, color="blue", fill=True, fill_opacity=0.1).add_to(m)

    #Cálculo de Rota Real e Adição ao Mapa
    ponto_x_geom = Point(st.session_state.ponto_x_manual[1], st.session_state.ponto_x_manual[0])
    
    for idx, row in top_10_distancia.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5, color="green", fill=True,
            popup=f"ID: {row['ID']} - Dist Real: {row['distancia_x_km']:.2f}km"
        ).add_to(m)
        
        # Chama a função de roteamento detalhada
        resultado = calcular_rota_real_detalhada(G_logistico, row.geometry, ponto_x_geom)
        
        if resultado:
            top_10_distancia.at[idx, 'Dist_Real_KM'] = resultado['dist_km']
            top_10_distancia.at[idx, 'ferrovia'] = resultado['tem_ferrovia']
            top_10_distancia.at[idx, 'rodovia'] = resultado['tem_rodovia']
            top_10_distancia.at[idx, 'n_ferrovia'] = resultado['n_ferrovia']
            top_10_distancia.at[idx, 'n_rodovia'] = resultado['n_rodovia']
            
            folium.PolyLine(resultado['coords'], color="blue", weight=3, opacity=0.7).add_to(m)

    # Exibição da Tabela Final (O que o LLM vai ler)
    st.subheader(f"🎯 Top {len(top_10_distancia)} Agricultores (Análise Logística Multimodal)")
    st.dataframe(top_10_distancia[[
        'ID', 'distancia_x_km', 'Dist_Real_KM', 'Score', 
        'ferrovia', 'rodovia', 'n_ferrovia', 'n_rodovia'
    ]], use_container_width=True)

    

    # Renderização do Mapa e Captura de Clique
    mapa_output = st_folium(m, width=800, height=500)

    if mapa_output['last_clicked']:
        nova_lat = mapa_output['last_clicked']['lat']
        nova_lon = mapa_output['last_clicked']['lng']
        if (nova_lat, nova_lon) != st.session_state.ponto_x_manual:
            st.session_state.ponto_x_manual = (nova_lat, nova_lon)
            st.rerun()

    ##### Chat #####
    from mainChat import ask_chatbot
    # ADICIONE O BOTÃO
    if st.button("🪄 Gerar Insight de Compra (IA)"):
        insight_prompt = "Com base nos dados da tabela, faça uma análise comparativa e recomende o melhor agricultor para compra hoje."
        with st.spinner("O Analista está analisando os cenários..."):
            from mainChat import ask_chatbot
            insight = ask_chatbot(insight_prompt, df_data=top_10_distancia)
            st.info(insight)

    # INICIALIZAÇÃO 
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.divider()
    st.title("🤖 Consultor Estratégico Agro M2")

    # Exibir histórico existente
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input do usuário e Lógica de Envio
    if prompt := st.chat_input("Ex: Qual o melhor agricultor considerando os dados que tem?"):
        # Adiciona a mensagem do usuário ao estado
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("IA analisando dados logísticos e score..."):
                # Passamos o DataFrame atual para o Llama ter contexto
                response = ask_chatbot(prompt, df_data=top_10_distancia) 
                st.markdown(response)

        # Adiciona a resposta da IA ao estado
        st.session_state.messages.append({"role": "assistant", "content": response})