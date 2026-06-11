import folium
import geopandas as gpd
import joblib
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.spatial import KDTree
from shapely.geometry import Point
from streamlit_folium import st_folium


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_CSV = "base_integrada_sprint4.csv"
MODEL_PATH = "classificador_vendedor.pkl"


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="AGRO M2 | Sprint 4",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# ESTILO GLOBAL
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
            color: #F9FAFB;
        }

        .block-container {
            padding-top: 4rem;
            padding-bottom: 2.5rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            max-width: 1500px;
        }

        h1, h2, h3 {
            color: green !important;
            margin-bottom: 0.35rem !important;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] .stNumberInput label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label {
            color: green !important;
            font-weight: 600;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: green !important;
        }

        .hero-box {
            background: linear-gradient(135deg, #123524 0%, #1f6f43 100%);
            padding: 1.55rem 1.65rem;
            border-radius: 22px;
            box-shadow: 0 12px 34px rgba(0,0,0,0.22);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
            color: #F9FAFB;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #D1FAE5;
            opacity: 0.95;
            line-height: 1.45;
        }

        .kpi-card {
            background: green;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 20px;
            padding: 1.15rem 1.1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.16);
            min-height: 132px;
        }

        .kpi-label {
            font-size: 0.92rem;
            color: #A5B4FC;
            margin-bottom: 0.55rem;
        }

        .kpi-value {
            font-size: 1.95rem;
            font-weight: 800;
            color: #F9FAFB;
            line-height: 1.1;
        }

        .kpi-help {
            font-size: 0.83rem;
            color: #CBD5E1;
            margin-top: 0.55rem;
            line-height: 1.35;
        }

        .best-card {
            background: linear-gradient(135deg, #1F2937 0%, #0F172A 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1.35rem 1.35rem 1.1rem 1.35rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.22);
            margin-bottom: 1.25rem;
            margin-top: 1.25rem;
        }

        .best-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #F9FAFB;
            margin-bottom: 0.9rem;
        }

        .best-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .best-item {
            background: rgba(255,255,255,0.035);
            border-radius: 16px;
            padding: 0.95rem 0.9rem;
            min-height: 86px;
        }

        .best-item-label {
            font-size: 0.82rem;
            color: #9CA3AF;
            margin-bottom: 0.28rem;
        }

        .best-item-value {
            font-size: 1.04rem;
            font-weight: 700;
            color: #F9FAFB;
            margin-top: 0.15rem;
            line-height: 1.25;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .badge-success {
            background: rgba(34,197,94,0.15);
            color: #86EFAC;
            border: 1px solid rgba(34,197,94,0.28);
        }

        .badge-warning {
            background: rgba(245,158,11,0.15);
            color: #FCD34D;
            border: 1px solid rgba(245,158,11,0.28);
        }

        .badge-danger {
            background: rgba(239,68,68,0.15);
            color: #FCA5A5;
            border: 1px solid rgba(239,68,68,0.28);
        }

        .section-card {
            background: #111827;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 20px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.16);
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-bottom: 0.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
            border-radius: 10px 10px 0 0;
        }

        .stButton > button {
            color: white;
            background: linear-gradient(135deg, #15803d 0%, #166534 100%);
            border-radius: 12px;
            border: none;
            font-weight: 700;
            padding: 0.65rem 1rem;
        }

        div[data-testid="stMetric"] {
            background: transparent;
        }

        @media (max-width: 1200px) {
            .best-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARREGAMENTO DA MALHA LOGÍSTICA
# ============================================================

@st.cache_data
def carregar_malha_logistica():
    """
    Carrega a malha logística.
    Se você possuir um shapefile hidroviário, pode descomentar o bloco
    indicado abaixo para incluir o modal hidroviário de fato no grafo.
    """
    rod = gpd.read_file("ShapeFiles_RF/rod_trecho_rodoviario_l.shp")
    fer = gpd.read_file("ShapeFiles_RF/fer_trecho_ferroviario_l.shp")

    rod["modal"] = "rodoviario"
    fer["modal"] = "ferroviario"

    malhas = [
        rod[["geometry", "modal"]],
        fer[["geometry", "modal"]],
    ]

    # Exemplo para usar hidrovia de verdade:
    # hid = gpd.read_file("ShapeFiles_RF/hid_trecho_hidroviario_l.shp")
    # hid["modal"] = "hidroviario"
    # malhas.append(hid[["geometry", "modal"]])

    malha = pd.concat(malhas, ignore_index=True)
    return malha


malha_logistica = carregar_malha_logistica()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.subheader("Preferência de Transporte")

opcoes_modal = {
    "Rodoviário": "rodoviario",
    "Ferroviário": "ferroviario",
    "Hidroviário": "hidroviario",
}

modal_label = st.sidebar.selectbox(
    "Modal prioritário",
    list(opcoes_modal.keys())
)

modal_preferido = opcoes_modal[modal_label]


# ============================================================
# GRAFO LOGÍSTICO
# ============================================================

@st.cache_resource
def construir_grafo_dinamico(_malha, preferido):
    malha_proj = _malha.to_crs(epsg=3857)
    G = momepy.gdf_to_nx(malha_proj, approach="primal")

    penalidade_alta = malha_proj.geometry.length.max() * 10

    for u, v, k, data in G.edges(data=True, keys=True):
        comprimento = data.get("mm_len", 0)
        modal_aresta = data.get("modal", "rodoviario")

        if modal_aresta == preferido:
            data["weight"] = comprimento
        else:
            data["weight"] = comprimento + penalidade_alta

    return G


G_logistico = construir_grafo_dinamico(malha_logistica, modal_preferido)


# ============================================================
# FUNÇÕES DE ROTA
# ============================================================

def encontrar_no_proximo(G, ponto):
    nodes_list = list(G.nodes)
    nodes_coords = np.array(nodes_list)
    tree = KDTree(nodes_coords)
    _, idx = tree.query([ponto.x, ponto.y])
    return nodes_list[idx]


def calcular_rota_real_detalhada(G, origem_geom, destino_geom):
    origem_proj = gpd.GeoSeries([origem_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    destino_proj = gpd.GeoSeries([destino_geom], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    node_origem = encontrar_no_proximo(G, origem_proj)
    node_destino = encontrar_no_proximo(G, destino_proj)

    try:
        rota_nos = nx.shortest_path(G, node_origem, node_destino, weight="weight")

        caminho_coords = []
        distancia_total_m = 0
        modais_na_rota = set()
        segmentos_rod = 0
        segmentos_fer = 0
        segmentos_hid = 0

        for i in range(len(rota_nos) - 1):
            u, v = rota_nos[i], rota_nos[i + 1]
            dados_aresta = G[u][v][0]

            distancia_total_m += dados_aresta.get("mm_len", 0)
            modal = dados_aresta.get("modal", "rodoviario")
            modais_na_rota.add(modal)

            if modal == "rodoviario":
                segmentos_rod += 1
            elif modal == "ferroviario":
                segmentos_fer += 1
            elif modal == "hidroviario":
                segmentos_hid += 1

            p_latlon = gpd.GeoSeries([Point(u[0], u[1])], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
            caminho_coords.append((p_latlon.y, p_latlon.x))

        return {
            "coords": caminho_coords,
            "dist_km": distancia_total_m / 1000,
            "tem_rodovia": "rodoviario" in modais_na_rota,
            "tem_ferrovia": "ferroviario" in modais_na_rota,
            "tem_hidrovia": "hidroviario" in modais_na_rota,
            "n_rodovia": segmentos_rod,
            "n_ferrovia": segmentos_fer,
            "n_hidrovia": segmentos_hid,
        }

    except Exception:
        return {
            "coords": [],
            "dist_km": 0.0,
            "tem_rodovia": False,
            "tem_ferrovia": False,
            "tem_hidrovia": False,
            "n_rodovia": 0,
            "n_ferrovia": 0,
            "n_hidrovia": 0,
        }


# ============================================================
# DADOS E MODELO
# ============================================================

@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def carregar_dados():
    df_original = pd.read_csv(INPUT_CSV)
    gdf = gpd.GeoDataFrame(
        df_original.copy(),
        geometry=gpd.points_from_xy(df_original.longitude, df_original.latitude),
        crs="EPSG:4326",
    )
    return df_original, gdf


modelo = carregar_modelo()
df_agr, gdf_agricultores = carregar_dados()


# ============================================================
# UI HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🌾 AGRO M2 — Sprint 4</div>
        <div class="hero-subtitle">
            Integração executiva de <b>Score</b>, <b>Logística</b>, <b>Preço</b> e <b>Clima</b> para apoio à tomada de decisão do Trader.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PARÂMETROS
# ============================================================

st.sidebar.header("Parâmetros do Trader")

raio_km = st.sidebar.number_input("Raio de busca (km)", min_value=1, max_value=500, value=100)
lat = st.sidebar.number_input("Latitude do destino", min_value=-90.0, max_value=90.0, value=-25.0)
lon = st.sidebar.number_input("Longitude do destino", min_value=-180.0, max_value=180.0, value=-49.0)

lista_produtos = ["Soja", "acucar", "cafe", "Milho", "arroz", "frutos", "vegetais"]
produtos_selecionados = st.sidebar.multiselect(
    "Filtrar por produtos",
    options=lista_produtos,
    default=lista_produtos
)

ponto_x_coord = (lat, lon)
st.sidebar.markdown(
    f"<span style='color: green;'>Destino: {ponto_x_coord}</span>",
    unsafe_allow_html=True
)

if "ponto_x_manual" not in st.session_state:
    st.session_state.ponto_x_manual = ponto_x_coord


# ============================================================
# FILTROS
# ============================================================

def filtrar_por_produto(gdf, selecionados):
    if not selecionados:
        return gdf
    mask = gdf[selecionados].any(axis=1)
    return gdf[mask].copy()


def filtrar_por_raio(gdf, centro, raio_km):
    ponto_destino = Point(centro[1], centro[0])

    ponto_gs = gpd.GeoSeries([ponto_destino], crs="EPSG:4326")
    ponto_metros = ponto_gs.to_crs(epsg=3857)
    gdf_metros = gdf.to_crs(epsg=3857)

    distancias = gdf_metros.distance(ponto_metros.iloc[0])

    gdf = gdf.copy()
    gdf["distancia_x_km"] = distancias / 1000
    return gdf[gdf["distancia_x_km"] <= raio_km].copy()


gdf_filtrado_prod = filtrar_por_produto(gdf_agricultores, produtos_selecionados)
agricultores_no_raio = filtrar_por_raio(
    gdf_filtrado_prod,
    st.session_state.ponto_x_manual,
    raio_km
)


# ============================================================
# FUNÇÕES SPRINT 4
# ============================================================

def garantir_colunas_sprint4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "produto_base": "desconhecido",
        "preco_produto_base": np.nan,
        "atratividade_preco": 0.5,
        "macro_regiao_climatica": "desconhecida",
        "precipitacao_7d": np.nan,
        "temperatura_media_7d": np.nan,
        "risco_climatico": "desconhecido",
        "risco_climatico_score": -1,
    }
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value
    return df


def normalizar_serie(series: pd.Series, invert=False, fill_value=0.5) -> pd.Series:
    s = series.copy().astype(float)
    if s.dropna().empty:
        return pd.Series([fill_value] * len(s), index=s.index)

    s_min = s.min()
    s_max = s.max()

    if np.isclose(s_min, s_max):
        normalized = pd.Series([fill_value] * len(s), index=s.index)
    else:
        normalized = (s - s_min) / (s_max - s_min)

    if invert:
        normalized = 1 - normalized

    return normalized.fillna(fill_value)


def calcular_score_final_sprint4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["score_norm"] = normalizar_serie(df["Score"])
    df["logistica_norm"] = normalizar_serie(df["Dist_Real_KM"], invert=True)
    df["preco_norm"] = normalizar_serie(df["atratividade_preco"])
    df["clima_norm"] = normalizar_serie(df["risco_climatico_score"], invert=True, fill_value=0.5)

    df["score_final_sprint4"] = (
        0.40 * df["score_norm"] +
        0.30 * df["logistica_norm"] +
        0.20 * df["preco_norm"] +
        0.10 * df["clima_norm"]
    )

    return df


def mapear_contexto_climatico(valor: str) -> str:
    mapa = {
        "Baixo": "Favorável",
        "Médio": "Moderado",
        "Alto": "Desfavorável",
        "Desconhecido": "Indefinido",
        "baixo": "Favorável",
        "medio": "Moderado",
        "alto": "Desfavorável",
        "desconhecido": "Indefinido",
    }
    return mapa.get(str(valor), str(valor))


def preparar_tabela_exibicao(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "preco_produto_base",
        "atratividade_preco",
        "precipitacao_7d",
        "temperatura_media_7d",
        "distancia_x_km",
        "Dist_Real_KM",
        "Score",
        "score_final_sprint4",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "macro_regiao_climatica" in df.columns:
        df["macro_regiao_climatica"] = df["macro_regiao_climatica"].astype(str).str.title()

    if "risco_climatico" in df.columns:
        df["risco_climatico"] = (
            df["risco_climatico"]
            .astype(str)
            .str.lower()
            .replace({
                "alto": "Alto",
                "medio": "Médio",
                "baixo": "Baixo",
                "desconhecido": "Desconhecido",
            })
            .map(mapear_contexto_climatico)
        )

    if "ferrovia" in df.columns:
        df["ferrovia"] = df["ferrovia"].map({True: "🚆 Sim", False: "—"}).fillna("—")

    if "rodovia" in df.columns:
        df["rodovia"] = df["rodovia"].map({True: "🛣️ Sim", False: "—"}).fillna("—")

    if "hidrovia" in df.columns:
        df["hidrovia"] = df["hidrovia"].map({True: "🚢 Sim", False: "—"}).fillna("—")

    return df


def risco_badge(risco: str) -> str:
    risco = mapear_contexto_climatico(str(risco))
    if risco == "Favorável":
        return '<span class="badge badge-success">Favorável</span>'
    if risco == "Moderado":
        return '<span class="badge badge-warning">Moderado</span>'
    if risco == "Desfavorável":
        return '<span class="badge badge-danger">Desfavorável</span>'
    return '<span class="badge badge-warning">Indefinido</span>'


def render_kpi(label: str, value: str, help_text: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_best_farmer_card(best_row: pd.Series):
    contexto_climatico = mapear_contexto_climatico(str(best_row["risco_climatico"]))

    modal_partes = []
    if bool(best_row.get("ferrovia", False)):
        modal_partes.append("🚆 Ferrovia")
    if bool(best_row.get("rodovia", False)):
        modal_partes.append("🛣️ Rodovia")
    if bool(best_row.get("hidrovia", False)):
        modal_partes.append("🚢 Hidrovia")

    modal_texto = " | ".join(modal_partes) if modal_partes else "—"

    st.markdown(
        f"""
        <div class="best-card">
            <div class="best-title">🏆 Melhor alternativa recomendada no cenário atual — ID {int(best_row["ID"])}</div>
            <div class="best-grid">
                <div class="best-item">
                    <div class="best-item-label">Produto</div>
                    <div class="best-item-value">{best_row["produto_base"]}</div>
                </div>
                <div class="best-item">
                    <div class="best-item-label">Score Final</div>
                    <div class="best-item-value">{best_row["score_final_sprint4"]:.3f}</div>
                </div>
                <div class="best-item">
                    <div class="best-item-label">Distância Real</div>
                    <div class="best-item-value">{best_row["Dist_Real_KM"]:.2f} km</div>
                </div>
                <div class="best-item">
                    <div class="best-item-label">Preço</div>
                    <div class="best-item-value">{best_row["preco_produto_base"]:.2f}</div>
                </div>
                <div class="best-item">
                    <div class="best-item-label">Atratividade</div>
                    <div class="best-item-value">{best_row["atratividade_preco"]:.3f}</div>
                </div>
                <div class="best-item">
                    <div class="best-item-label">Macro Região</div>
                    <div class="best-item-value">{str(best_row["macro_regiao_climatica"]).title()}</div>
                </div>
                <div class="best-item">
                    <div class="best-item-label">Contexto Climático Regional</div>
                    <div class="best-item-value">{contexto_climatico}</div>
                </div>
                <div class="best-item">
                    <div class="best-item-label">Modal da rota</div>
                    <div class="best-item-value">{modal_texto}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================

if not agricultores_no_raio.empty:
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

    top_10_distancia = agricultores_no_raio.sort_values(by="distancia_x_km").head(10).copy()
    top_10_distancia = garantir_colunas_sprint4(top_10_distancia)

    lista_ids = top_10_distancia["ID"].tolist()
    dados_para_ml = (
        df_agr[df_agr["ID"].isin(lista_ids)]
        .set_index("ID")
        .loc[lista_ids]
        .reset_index()
    )

    top_10_distancia["Score"] = modelo.predict(dados_para_ml[colunas_ml])

    top_10_distancia["Dist_Real_KM"] = 0.0
    top_10_distancia["ferrovia"] = False
    top_10_distancia["rodovia"] = False
    top_10_distancia["hidrovia"] = False
    top_10_distancia["n_ferrovia"] = 0
    top_10_distancia["n_rodovia"] = 0
    top_10_distancia["n_hidrovia"] = 0

    # MAPA
    m = folium.Map(location=st.session_state.ponto_x_manual, zoom_start=7)

    folium.Marker(
        st.session_state.ponto_x_manual,
        tooltip="Destino",
        icon=folium.Icon(color="red")
    ).add_to(m)

    folium.Circle(
        location=st.session_state.ponto_x_manual,
        radius=raio_km * 1000,
        color="blue",
        fill=True,
        fill_opacity=0.1
    ).add_to(m)

    ponto_x_geom = Point(
        st.session_state.ponto_x_manual[1],
        st.session_state.ponto_x_manual[0]
    )

    for idx, row in top_10_distancia.iterrows():
        popup_txt = (
            f"ID: {row['ID']}<br>"
            f"Produto: {row.get('produto_base', 'N/D')}<br>"
            f"Dist. reta: {row['distancia_x_km']:.2f} km"
        )

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color="#22c55e",
            fill=True,
            fill_color="#22c55e",
            fill_opacity=0.9,
            popup=popup_txt,
        ).add_to(m)

        resultado = calcular_rota_real_detalhada(G_logistico, row.geometry, ponto_x_geom)

        if resultado:
            top_10_distancia.at[idx, "Dist_Real_KM"] = resultado["dist_km"]
            top_10_distancia.at[idx, "ferrovia"] = resultado["tem_ferrovia"]
            top_10_distancia.at[idx, "rodovia"] = resultado["tem_rodovia"]
            top_10_distancia.at[idx, "hidrovia"] = resultado["tem_hidrovia"]
            top_10_distancia.at[idx, "n_ferrovia"] = resultado["n_ferrovia"]
            top_10_distancia.at[idx, "n_rodovia"] = resultado["n_rodovia"]
            top_10_distancia.at[idx, "n_hidrovia"] = resultado["n_hidrovia"]

            folium.PolyLine(
                resultado["coords"],
                color="#60A5FA",
                weight=4,
                opacity=0.75
            ).add_to(m)

    # SCORE FINAL
    top_10_distancia = calcular_score_final_sprint4(top_10_distancia)
    top_10_distancia = top_10_distancia.sort_values(
        by="score_final_sprint4",
        ascending=False
    ).reset_index(drop=True)

    top_10_display = preparar_tabela_exibicao(top_10_distancia)
    melhor = top_10_distancia.iloc[0]

    contexto_climatico_topo = mapear_contexto_climatico(str(top_10_display["risco_climatico"].mode().iloc[0]))
    precip_media = top_10_distancia["precipitacao_7d"].mean()
    temp_media = top_10_distancia["temperatura_media_7d"].mean()

    # KPIs
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        render_kpi("Melhor Score Final", f"{melhor['score_final_sprint4']:.3f}", f"ID {int(melhor['ID'])}")
    with c2:
        render_kpi("Menor Distância Real", f"{top_10_distancia['Dist_Real_KM'].min():.2f} km", "Eficiência logística")
    with c3:
        render_kpi("Preço Médio", f"{top_10_distancia['preco_produto_base'].mean():.2f}", "Origens filtradas")
    with c4:
        render_kpi(
            "Contexto Climático Regional",
            contexto_climatico_topo,
            f"Precipitação 7d: {precip_media:.1f} mm | Temp. média: {temp_media:.1f} °C"
        )

    render_best_farmer_card(melhor)

    tab1, tab2, tab3 = st.tabs(["📊 Ranking", "🗺️ Mapa & Rotas", "🤖 IA"])

    with tab1:
        st.markdown("### 📊 Ranking consolidado das alternativas")

        colunas_exibicao = [
            "ID",
            "produto_base",
            "preco_produto_base",
            "atratividade_preco",
            "macro_regiao_climatica",
            "precipitacao_7d",
            "temperatura_media_7d",
            "risco_climatico",
            "distancia_x_km",
            "Dist_Real_KM",
            "Score",
            "score_final_sprint4",
            "ferrovia",
            "rodovia",
            "hidrovia",
            "n_ferrovia",
            "n_rodovia",
            "n_hidrovia",
        ]

        st.dataframe(
            top_10_display[colunas_exibicao],
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", format="%d"),
                "produto_base": st.column_config.TextColumn("Produto"),
                "preco_produto_base": st.column_config.NumberColumn("Preço", format="%.2f"),
                "atratividade_preco": st.column_config.NumberColumn("Atratividade", format="%.3f"),
                "macro_regiao_climatica": st.column_config.TextColumn("Macro Região"),
                "precipitacao_7d": st.column_config.NumberColumn("Precipitação 7d", format="%.1f"),
                "temperatura_media_7d": st.column_config.NumberColumn("Temp. Média 7d", format="%.1f"),
                "risco_climatico": st.column_config.TextColumn("Contexto Climático"),
                "distancia_x_km": st.column_config.NumberColumn("Distância Reta (km)", format="%.2f"),
                "Dist_Real_KM": st.column_config.NumberColumn("Distância Real (km)", format="%.2f"),
                "Score": st.column_config.NumberColumn("Score", format="%.2f"),
                "score_final_sprint4": st.column_config.NumberColumn("Score Final", format="%.3f"),
                "ferrovia": st.column_config.TextColumn("Ferrovia"),
                "rodovia": st.column_config.TextColumn("Rodovia"),
                "hidrovia": st.column_config.TextColumn("Hidrovia"),
                "n_ferrovia": st.column_config.NumberColumn("Seg. Ferrovia", format="%d"),
                "n_rodovia": st.column_config.NumberColumn("Seg. Rodovia", format="%d"),
                "n_hidrovia": st.column_config.NumberColumn("Seg. Hidrovia", format="%d"),
            },
        )

        ranking_top5 = top_10_distancia.head(5).copy()
        ranking_top5["rank"] = range(1, len(ranking_top5) + 1)
        ranking_top5["ranking_label"] = ranking_top5.apply(
            lambda row: f'{row["rank"]}º • ID {int(row["ID"])}',
            axis=1
        )

        fig = px.bar(
            ranking_top5.sort_values("score_final_sprint4", ascending=True),
            x="score_final_sprint4",
            y="ranking_label",
            orientation="h",
            text="score_final_sprint4",
            title="Top 5 produtores por Score Final"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            marker=dict(color="#22c55e")
        )

        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=55, b=20),
            showlegend=False,
            xaxis_title="Score Final",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(range=[0, 1], tickformat=".2f", showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False}
        )

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Mapa logístico e rotas calculadas")
        mapa_output = st_folium(m, width=1200, height=580)

        if mapa_output["last_clicked"]:
            nova_lat = mapa_output["last_clicked"]["lat"]
            nova_lon = mapa_output["last_clicked"]["lng"]

            if (nova_lat, nova_lon) != st.session_state.ponto_x_manual:
                st.session_state.ponto_x_manual = (nova_lat, nova_lon)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Consultor estratégico da Sprint 4")

        from mainChat import ask_chatbot

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Resumo do melhor cenário**")
            st.markdown(
                f"""
                - **ID recomendado:** {int(melhor['ID'])}  
                - **Produto:** {melhor['produto_base']}  
                - **Score Final:** {melhor['score_final_sprint4']:.3f}  
                - **Distância Real:** {melhor['Dist_Real_KM']:.2f} km  
                - **Preço:** {melhor['preco_produto_base']:.2f}  
                - **Contexto Climático Regional:** {mapear_contexto_climatico(str(melhor['risco_climatico']))}
                """
            )

        with col_b:
            st.markdown("**Contexto climático predominante**")
            st.markdown(risco_badge(str(melhor["risco_climatico"])), unsafe_allow_html=True)

        if st.button("🪄 Gerar Insight Integrado (IA)"):
            insight_prompt = (
                "Com base nos dados da tabela, faça uma análise comparativa "
                "considerando score, logística, preço e clima, e recomende "
                "o melhor agricultor para compra."
            )
            with st.spinner("O Analista está integrando score, clima, preço e logística..."):
                insight = ask_chatbot(insight_prompt, df_data=top_10_distancia)
                st.info(insight)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: Qual produtor apresenta melhor equilíbrio entre score, rota, clima e preço?"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("IA analisando score, logística, preço e clima..."):
                    response = ask_chatbot(prompt, df_data=top_10_distancia)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("Nenhum agricultor encontrado para os filtros aplicados.")