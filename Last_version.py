import os
import folium
import geopandas as gpd
import joblib
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px  # Atualmente não está sendo usado no fluxo principal
import streamlit as st
from scipy.spatial import KDTree
from shapely.geometry import Point
from streamlit_folium import st_folium

from price import sacas  # Atualmente não está sendo usado diretamente neste arquivo


# ============================================================
# 1. CARREGAMENTO DA MALHA LOGÍSTICA
# ============================================================
# Esta função lê os shapefiles de rodovia e ferrovia e os
# consolida em uma única base geoespacial, adicionando uma
# coluna "modal" para indicar o tipo de transporte.
@st.cache_data
def carregar_malha_logistica():
    rod = gpd.read_file("ShapeFiles_RF/rod_trecho_rodoviario_l.shp")
    fer = gpd.read_file("ShapeFiles_RF/fer_trecho_ferroviario_l.shp")

    rod["modal"] = "rodoviario"
    fer["modal"] = "ferroviario"

    malha = pd.concat(
        [rod[["geometry", "modal"]], fer[["geometry", "modal"]]],
        ignore_index=True
    )
    return malha


# Carrega a malha logística uma única vez e mantém em cache
malha_logistica = carregar_malha_logistica()


# ============================================================
# 2. ESTILIZAÇÃO DA INTERFACE STREAMLIT
# ============================================================
# Bloco CSS simples para personalizar aparência do dashboard.
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. CONFIGURAÇÃO DE PREFERÊNCIA DE MODAL
# ============================================================
# O usuário escolhe na barra lateral qual modal deseja priorizar
# no cálculo das rotas.
st.sidebar.subheader("Preferência de Transporte")
modal_preferido = st.sidebar.selectbox(
    "Modal Prioritário:",
    ["rodoviario", "ferroviario"]
)


# ============================================================
# 4. CONSTRUÇÃO DO GRAFO LOGÍSTICO
# ============================================================
# Converte a malha geoespacial em um grafo de rede, onde as
# arestas recebem pesos diferentes conforme o modal preferido.
# O modal não prioritário recebe uma penalização para tornar
# sua escolha menos provável no menor caminho.
@st.cache_resource
def construir_grafo_dinamico(_malha, preferido):
    # Reprojeta para um sistema métrico para cálculo de distâncias
    malha_proj = _malha.to_crs(epsg=3857)

    # Converte a malha em grafo no modo primal (linhas viram arestas)
    G = momepy.gdf_to_nx(malha_proj, approach="primal")

    # Penalização alta para desincentivar modal não prioritário
    penalidade_alta = malha_proj.geometry.length.max() * 10

    for u, v, k, data in G.edges(data=True, keys=True):
        comprimento = data.get("mm_len", 0)
        modal_aresta = data.get("modal", "rodoviario")

        if modal_aresta == preferido:
            data["weight"] = comprimento
        else:
            data["weight"] = comprimento + penalidade_alta

    return G


# Grafo logístico utilizado nas rotas
G_logistico = construir_grafo_dinamico(malha_logistica, modal_preferido)


# ============================================================
# 5. FUNÇÕES AUXILIARES DE ROTEAMENTO
# ============================================================
# Dado um ponto, encontra o nó mais próximo no grafo usando KDTree.
def encontrar_no_proximo(G, ponto):
    nodes_list = list(G.nodes)
    nodes_coords = np.array(nodes_list)

    tree = KDTree(nodes_coords)
    dist, idx = tree.query([ponto.x, ponto.y])
    return nodes_list[idx]


# Calcula a rota mais curta entre origem e destino no grafo,
# retornando:
# - coordenadas da rota para desenhar no mapa
# - distância total em km
# - presença de rodovia e/ou ferrovia
# - número de segmentos de cada modal
def calcular_rota_real_detalhada(G, origem_geom, destino_geom):
    # Reprojeta pontos para sistema métrico
    origem_proj = gpd.GeoSeries([origem_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    destino_proj = gpd.GeoSeries([destino_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    node_origem = encontrar_no_proximo(G, origem_proj)
    node_destino = encontrar_no_proximo(G, destino_proj)

    try:
        # Menor caminho ponderado pelo peso calculado no grafo
        rota_nos = nx.shortest_path(G, node_origem, node_destino, weight="weight")

        caminho_coords = []
        distancia_total_m = 0
        modais_na_rota = set()
        segmentos_rod = 0
        segmentos_fer = 0

        for i in range(len(rota_nos) - 1):
            u, v = rota_nos[i], rota_nos[i + 1]
            dados_aresta = G[u][v][0]

            distancia_total_m += dados_aresta.get("mm_len", 0)
            modal = dados_aresta.get("modal", "rodoviario")
            modais_na_rota.add(modal)

            if modal == "rodoviario":
                segmentos_rod += 1
            else:
                segmentos_fer += 1

            # Converte o ponto da rede para lat/lon para desenhar no mapa
            p_latlon = gpd.GeoSeries([Point(u[0], u[1])], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
            caminho_coords.append((p_latlon.y, p_latlon.x))

        return {
            "coords": caminho_coords,
            "dist_km": distancia_total_m / 1000,
            "tem_rodovia": "rodoviario" in modais_na_rota,
            "tem_ferrovia": "ferroviario" in modais_na_rota,
            "n_rodovia": segmentos_rod,
            "n_ferrovia": segmentos_fer,
        }

    except Exception:
        # Caso não seja possível calcular a rota, retorna estrutura vazia
        return {
            "coords": [],
            "dist_km": 0.0,
            "tem_rodovia": False,
            "tem_ferrovia": False,
            "n_rodovia": 0,
            "n_ferrovia": 0,
        }


# ============================================================
# 6. LEITURA DA BASE DE AGRICULTORES
# ============================================================
# Lê a base principal da aplicação.
df_agr = pd.read_csv("df_agr.csv")

# Cria GeoDataFrame a partir das colunas de longitude e latitude
gdf = gpd.GeoDataFrame(
    df_agr,
    geometry=gpd.points_from_xy(df_agr.longitude, df_agr.latitude),
    crs="EPSG:4326",
)

# Exporta para shapefile.
# Observação: a leitura posterior considera que o shapefile já está
# na pasta ShapeFiles. Aqui a exportação é mantida conforme o código original.
gdf.to_file("agricultores_agro_m2.shp")

# Teste de leitura da camada geográfica dos agricultores
gdf_teste = gpd.read_file("ShapeFiles/agricultores_agro_m2.shp")


# ============================================================
# 7. CONFIGURAÇÃO PRINCIPAL DA INTERFACE
# ============================================================
st.title("Agro M2 - Análise Logística")


# ============================================================
# 8. CARREGAMENTO DO MODELO E DOS DADOS
# ============================================================
# Carrega o modelo de classificação do vendedor.
@st.cache_resource
def carregar_modelo():
    return joblib.load("classificador_vendedor.pkl")


# Carrega novamente a base tabular e a camada geográfica.
@st.cache_data
def carregar_dados():
    df_original = pd.read_csv("df_agr.csv")
    gdf = gpd.read_file("ShapeFiles/agricultores_agro_m2.shp")
    return df_original, gdf


modelo = carregar_modelo()
df_agr, gdf_agricultores = carregar_dados()


# ============================================================
# 9. PARÂMETROS DE BUSCA NA SIDEBAR
# ============================================================
st.sidebar.header("Parâmetros do Trader")

# Raio de busca em quilômetros
raio_km = st.sidebar.number_input(
    "Raio de busca (km)",
    min_value=1,
    max_value=500,
    value=100
)

# Coordenadas do ponto de destino
lat = st.sidebar.number_input(
    "Latitude do destino",
    min_value=-90.0,
    max_value=90.0,
    value=-25.0
)
lon = st.sidebar.number_input(
    "Longitude do destino",
    min_value=-180.0,
    max_value=180.0,
    value=-49.0
)

# Lista fixa de produtos disponíveis na base
lista_produtos = ["Soja", "acucar", "cafe", "Milho", "arroz", "frutos", "vegetais"]

# Seleção múltipla de produtos
produtos_selecionados = st.sidebar.multiselect(
    "Filtrar por Produtos:",
    options=lista_produtos,
    default=lista_produtos
)

# Ponto inicial do destino
ponto_x_coord = (lat, lon)
st.sidebar.write(f"Destino (Ponto X): {ponto_x_coord}")

# Mantém o ponto em session_state para permitir atualização via clique no mapa
if "ponto_x_manual" not in st.session_state:
    st.session_state.ponto_x_manual = ponto_x_coord


# ============================================================
# 10. FILTROS DE PRODUTO E DISTÂNCIA
# ============================================================
# Filtra os agricultores com base nas colunas booleanas/indicadoras dos produtos.
def filtrar_por_produto(gdf, selecionados):
    if not selecionados:
        return gdf

    mask = gdf[selecionados].any(axis=1)
    return gdf[mask].copy()


# Aplica primeiro o filtro por produto para reduzir o volume do processamento espacial
gdf_filtrado_prod = filtrar_por_produto(gdf_agricultores, produtos_selecionados)


# Filtra os agricultores por raio de distância a partir do ponto de destino.
def filtrar_por_raio(gdf, centro, raio_km):
    ponto_destino = Point(centro[1], centro[0])

    ponto_gs = gpd.GeoSeries([ponto_destino], crs="EPSG:4326")
    ponto_metros = ponto_gs.to_crs(epsg=3857)
    gdf_metros = gdf.to_crs(epsg=3857)

    distancias = gdf_metros.distance(ponto_metros.iloc[0])

    # Adiciona distância em linha reta até o ponto X
    gdf["distancia_x_km"] = distancias / 1000

    return gdf[gdf["distancia_x_km"] <= raio_km].copy()


# Aplica o filtro espacial sobre o conjunto já filtrado por produto
agricultores_no_raio = filtrar_por_raio(
    gdf_filtrado_prod,
    st.session_state.ponto_x_manual,
    raio_km
)


# ============================================================
# 11. PROCESSAMENTO PRINCIPAL: SCORE + LOGÍSTICA + MAPA
# ============================================================
if not agricultores_no_raio.empty:
    # Variáveis esperadas pelo modelo preditivo do vendedor
    colunas_ml = [
        "RFR_QTD",
        "TPN_QTD",
        "RI_QTD_TOT",
        "RR_INST_QTD",
        "RR_OP_QTD",
        "CRC_NUMERICA",
        "Days_to_Harvest",
        "mean_valuation",
    ]

    # Seleciona os 10 agricultores mais próximos do ponto informado
    top_10_distancia = agricultores_no_raio.sort_values(by="distancia_x_km").head(10).copy()

    # Recupera os registros correspondentes para aplicar o modelo
    lista_ids = top_10_distancia["ID"].tolist()
    dados_para_ml = (
        df_agr[df_agr["ID"].isin(lista_ids)]
        .set_index("ID")
        .loc[lista_ids]
        .reset_index()
    )

    # Prediz o score dos agricultores selecionados
    top_10_distancia["Score"] = modelo.predict(dados_para_ml[colunas_ml])

    # Inicializa colunas logísticas que serão preenchidas pelo cálculo de rotas
    top_10_distancia["Dist_Real_KM"] = 0.0
    top_10_distancia["ferrovia"] = False
    top_10_distancia["rodovia"] = False
    top_10_distancia["n_ferrovia"] = 0
    top_10_distancia["n_rodovia"] = 0

    # ========================================================
    # 11.1 CONSTRUÇÃO DO MAPA
    # ========================================================
    m = folium.Map(location=st.session_state.ponto_x_manual, zoom_start=7)

    # Marca o destino
    folium.Marker(
        st.session_state.ponto_x_manual,
        tooltip="Destino",
        icon=folium.Icon(color="red")
    ).add_to(m)

    # Desenha o raio de busca
    folium.Circle(
        location=st.session_state.ponto_x_manual,
        radius=raio_km * 1000,
        color="blue",
        fill=True,
        fill_opacity=0.1
    ).add_to(m)

    # Geometria do ponto de destino
    ponto_x_geom = Point(
        st.session_state.ponto_x_manual[1],
        st.session_state.ponto_x_manual[0]
    )

    # ========================================================
    # 11.2 CÁLCULO DE ROTAS E INSERÇÃO NO MAPA
    # ========================================================
    for idx, row in top_10_distancia.iterrows():
        # Marca o agricultor no mapa
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="green",
            fill=True,
            popup=f"ID: {row['ID']} - Dist Real: {row['distancia_x_km']:.2f}km",
        ).add_to(m)

        # Calcula a rota logística entre agricultor e destino
        resultado = calcular_rota_real_detalhada(G_logistico, row.geometry, ponto_x_geom)

        if resultado:
            # Preenche as métricas logísticas no dataframe final
            top_10_distancia.at[idx, "Dist_Real_KM"] = resultado["dist_km"]
            top_10_distancia.at[idx, "ferrovia"] = resultado["tem_ferrovia"]
            top_10_distancia.at[idx, "rodovia"] = resultado["tem_rodovia"]
            top_10_distancia.at[idx, "n_ferrovia"] = resultado["n_ferrovia"]
            top_10_distancia.at[idx, "n_rodovia"] = resultado["n_rodovia"]

            # Desenha a rota no mapa
            folium.PolyLine(
                resultado["coords"],
                color="blue",
                weight=3,
                opacity=0.7
            ).add_to(m)

    # ========================================================
    # 11.3 TABELA FINAL DE RESULTADOS
    # ========================================================
    st.subheader(f"🎯 Top {len(top_10_distancia)} Agricultores (Análise Logística Multimodal)")

    st.dataframe(
        top_10_distancia[
            [
                "ID",
                "distancia_x_km",
                "Dist_Real_KM",
                "Score",
                "ferrovia",
                "rodovia",
                "n_ferrovia",
                "n_rodovia",
            ]
        ],
        use_container_width=True,
    )

    # ========================================================
    # 11.4 RENDERIZAÇÃO DO MAPA E CAPTURA DE CLIQUE
    # ========================================================
    mapa_output = st_folium(m, width=800, height=500)

    # Permite atualizar o ponto de destino clicando no mapa
    if mapa_output["last_clicked"]:
        nova_lat = mapa_output["last_clicked"]["lat"]
        nova_lon = mapa_output["last_clicked"]["lng"]

        if (nova_lat, nova_lon) != st.session_state.ponto_x_manual:
            st.session_state.ponto_x_manual = (nova_lat, nova_lon)
            st.rerun()

    # ========================================================
    # 11.5 BLOCO DE IA / CHAT
    # ========================================================
    # Importa a função que consulta o modelo de IA com contexto do dataframe atual
    from mainChat import ask_chatbot

    # Botão para gerar uma recomendação automática com IA
    if st.button("🪄 Gerar Insight de Compra (IA)"):
        insight_prompt = (
            "Com base nos dados da tabela, faça uma análise comparativa "
            "e recomende o melhor agricultor para compra hoje."
        )
        with st.spinner("O Analista está analisando os cenários..."):
            insight = ask_chatbot(insight_prompt, df_data=top_10_distancia)
            st.info(insight)

    # Inicializa histórico da conversa
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.divider()
    st.title("🤖 Consultor Estratégico Agro M2")

    # Exibe histórico existente do chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Campo de entrada do usuário no chat
    if prompt := st.chat_input("Ex: Qual o melhor agricultor considerando os dados que tem?"):
        # Registra mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera resposta da IA usando o dataframe atual como contexto
        with st.chat_message("assistant"):
            with st.spinner("IA analisando dados logísticos e score..."):
                response = ask_chatbot(prompt, df_data=top_10_distancia)
                st.markdown(response)

        # Armazena a resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": response})