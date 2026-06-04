import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_CSV = "df_agr.csv"
OUTPUT_CSV = "base_integrada_sprint4.csv"

CLIMATE_WINDOW_DAYS = 7

PRODUCT_COLUMNS = [
    "Soja",
    "acucar",
    "cafe",
    "Milho",
    "arroz",
    "frutos",
    "vegetais",
]

PRICE_COLUMNS_MAP = {
    "Soja": "preco_Soja",
    "acucar": "preco_acucar",
    "cafe": "preco_cafe",
    "Milho": "preco_Milho",
    "arroz": "preco_arroz",
    "frutos": "preco_frutos",
    "vegetais": "preco_vegetais",
}


# ============================================================
# 1. PRODUTO BASE E PREÇO
# ============================================================

def identificar_produto_base(row: pd.Series) -> str | None:
    """
    Identifica o produto principal do registro.
    Regra: pega o primeiro produto marcado como > 0.
    """
    for product in PRODUCT_COLUMNS:
        value = row.get(product, 0)
        if pd.notna(value) and value > 0:
            return product
    return None


def obter_preco_produto_base(row: pd.Series) -> float:
    """
    Recupera o preço associado ao produto base.
    """
    product = row.get("produto_base")
    if not product:
        return np.nan

    price_col = PRICE_COLUMNS_MAP.get(product)
    if not price_col:
        return np.nan

    value = row.get(price_col, np.nan)
    return float(value) if pd.notna(value) and value > 0 else np.nan


def calcular_atratividade_preco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza atratividade de preço por produto.
    Regra simples:
    - menor preço = maior atratividade
    - escala de 0 a 1 dentro de cada produto
    """
    df = df.copy()
    df["atratividade_preco"] = np.nan

    for product in PRODUCT_COLUMNS:
        mask = df["produto_base"] == product
        subset = df.loc[mask, "preco_produto_base"].dropna()

        if subset.empty:
            continue

        p_min = subset.min()
        p_max = subset.max()

        if p_min == p_max:
            df.loc[mask, "atratividade_preco"] = 0.5
        else:
            df.loc[mask, "atratividade_preco"] = (
                (p_max - df.loc[mask, "preco_produto_base"]) / (p_max - p_min)
            )

    return df


# ============================================================
# 2. REGIÃO CLIMÁTICA SIMPLIFICADA
# ============================================================

def classificar_macro_regiao_climatica(lat: float, lon: float) -> str:
    """
    Cria uma regionalização climática simplificada para a PoC.
    Critério simples baseado em latitude:
    - Norte
    - Centro
    - Sul
    """
    if pd.isna(lat) or pd.isna(lon):
        return "desconhecida"

    if lat > -10:
        return "norte"
    elif -25 <= lat <= -10:
        return "centro"
    else:
        return "sul"


def get_macro_region_centroids() -> dict:
    """
    Define coordenadas representativas para cada macro-região climática.
    Esses pontos serão usados para consultar a API.
    """
    return {
        "norte": {"latitude": -5.0, "longitude": -55.0},
        "centro": {"latitude": -15.0, "longitude": -50.0},
        "sul": {"latitude": -27.0, "longitude": -52.0},
    }


# ============================================================
# 3. CLIMA - OPEN-METEO
# ============================================================

def classificar_risco_climatico(precip_7d: float, temp_mean_7d: float) -> tuple[str, int]:
    """
    Classificação heurística simplificada para a PoC.

    Regras:
    - Baixo: condições mais equilibradas de chuva e temperatura
    - Médio: condições intermediárias ou levemente desfavoráveis
    - Alto: ausência total de chuva, excesso forte de chuva ou calor elevado
    """
    if pd.isna(precip_7d) or pd.isna(temp_mean_7d):
        return "desconhecido", -1

    # Alto risco
    if precip_7d == 0 or precip_7d > 140 or temp_mean_7d > 35:
        return "alto", 2

    # Baixo risco
    if 5 <= precip_7d <= 100 and 18 <= temp_mean_7d <= 32:
        return "baixo", 0

    # Caso intermediário
    return "medio", 1


def build_open_meteo_url(lat: float, lon: float, start_date: str, end_date: str) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def fetch_climate_from_open_meteo(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    Busca dados climáticos históricos agregados.
    """
    url = build_open_meteo_url(lat, lon, start_date, end_date)

    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    daily = payload.get("daily", {})
    precip_list = daily.get("precipitation_sum", [])
    temp_list = daily.get("temperature_2m_mean", [])

    precip_7d = float(np.nansum(precip_list)) if precip_list else np.nan
    temp_mean_7d = float(np.nanmean(temp_list)) if temp_list else np.nan

    risco_label, risco_score = classificar_risco_climatico(precip_7d, temp_mean_7d)

    return {
        "precipitacao_7d": precip_7d,
        "temperatura_media_7d": temp_mean_7d,
        "risco_climatico": risco_label,
        "risco_climatico_score": risco_score,
    }


def montar_tabela_climatica_macro_regioes() -> pd.DataFrame:
    """
    Consulta clima apenas para poucos centróides regionais.
    """
    centroids = get_macro_region_centroids()

    end_dt = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=CLIMATE_WINDOW_DAYS - 1)

    start_date = start_dt.isoformat()
    end_date = end_dt.isoformat()

    rows = []

    for regiao, coords in centroids.items():
        try:
            clima = fetch_climate_from_open_meteo(
                lat=coords["latitude"],
                lon=coords["longitude"],
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            print(f"Erro ao consultar clima da região {regiao}: {exc}")
            clima = {
                "precipitacao_7d": np.nan,
                "temperatura_media_7d": np.nan,
                "risco_climatico": "desconhecido",
                "risco_climatico_score": -1,
            }

        clima["macro_regiao_climatica"] = regiao
        rows.append(clima)

    return pd.DataFrame(rows)


# ============================================================
# 4. PIPELINE PRINCIPAL
# ============================================================

def main():
    print("Lendo base original...")
    df = pd.read_csv(INPUT_CSV)

    print("Identificando produto_base...")
    df["produto_base"] = df.apply(identificar_produto_base, axis=1)

    print("Calculando preco_produto_base...")
    df["preco_produto_base"] = df.apply(obter_preco_produto_base, axis=1)

    print("Calculando atratividade_preco...")
    df = calcular_atratividade_preco(df)

    print("Classificando macro_regiao_climatica...")
    df["macro_regiao_climatica"] = df.apply(
        lambda row: classificar_macro_regiao_climatica(row["latitude"], row["longitude"]),
        axis=1
    )

    print("Consultando clima por macro-região...")
    clima_df = montar_tabela_climatica_macro_regioes()

    print("Fazendo merge do clima...")
    df = df.merge(clima_df, on="macro_regiao_climatica", how="left")

    print(f"Salvando base enriquecida em: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print("Concluído com sucesso.")
    print("Colunas novas geradas:")
    print([
        "produto_base",
        "preco_produto_base",
        "atratividade_preco",
        "macro_regiao_climatica",
        "precipitacao_7d",
        "temperatura_media_7d",
        "risco_climatico",
        "risco_climatico_score",
    ])


if __name__ == "__main__":
    main()