import osmnx as ox
import os
import pandas as pd
import geopandas as gpd

def gerar_arquivos_logistica_final():
    if not os.path.exists("ShapeFiles"):
        os.makedirs("ShapeFiles")

    df = pd.read_csv("df_agr.csv")

    norte = df.latitude.max() + 0.5
    sul = df.latitude.min() - 0.5
    leste = df.longitude.max() + 0.5
    oeste = df.longitude.min() - 0.5

    print(f"Buscando malha logistica: N:{norte}, S:{sul}, L:{leste}, O:{oeste}")

    bbox = (oeste, sul, leste, norte)

    filtros = {
        "rodovias": {"highway": ["motorway", "trunk", "primary", "secondary"]},
        "ferrovias": {"railway": "rail"}
    }

    for nome, tag_filtro in filtros.items():
        print(f"--- Baixando {nome} ---")
        try:
            vias = ox.features_from_bbox(bbox=bbox, tags=tag_filtro)

            vias = vias[vias.geometry.type.isin(["LineString", "MultiLineString"])].copy()
            vias["modal"] = "rodoviario" if nome == "rodovias" else "ferroviario"

            vias[["geometry", "modal"]].to_file(f"ShapeFiles/{nome}.shp")
            print(f"✅ Sucesso: {nome}.shp gerado com {len(vias)} segmentos.")

        except Exception as e:
            print(f"❌ Erro ao processar {nome}: {e}")

if __name__ == "__main__":
    gerar_arquivos_logistica_final()