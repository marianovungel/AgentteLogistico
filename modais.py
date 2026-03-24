import osmnx as ox
import os
import pandas as pd
import geopandas as gpd

def gerar_arquivos_logistica_final():
    if not os.path.exists("ShapeFiles"):
        os.makedirs("ShapeFiles")

    # 1. Carregar coordenadas do seu dataset
    df = pd.read_csv('df_agr.csv')
    
    # Definir os limites (Bounding Box)
    norte, sul = df.latitude.max() + 0.5, df.latitude.min() - 0.5
    leste, oeste = df.longitude.max() + 0.5, df.longitude.min() - 0.5

    print(f"Buscando malha logistica: N:{norte}, S:{sul}, L:{leste}, O:{oeste}")

    # 2. Configuração dos filtros de escoamento (Rodovias e Ferrovias) 
    # Filtramos apenas as vias principais para garantir a performance do Grafo [cite: 11]
    filtros = {
        "rodovias": {"highway": ["motorway", "trunk", "primary", "secondary"]},
        "ferrovias": {"railway": "rail"}
    }

    for nome, tag_filtro in filtros.items():
        print(f"--- Baixando {nome} ---")
        try:
            # Correção da chamada da função
            vias = ox.features_from_bbox(
                north=norte, south=sul, east=leste, west=oeste, 
                tags=tag_filtro
            )
            
            # Filtramos para garantir que temos apenas linhas (Arestas) [cite: 14]
            vias = vias[vias.geometry.type.isin(['LineString', 'MultiLineString'])]
            
            # Adicionamos a etiqueta do modal para o cálculo de peso W posterior 
            vias['modal'] = 'rodoviario' if nome == "rodovias" else 'ferroviario'
            
            # Exportação para Shapefile (shp, shx, dbf) 
            vias[['geometry', 'modal']].to_file(f"ShapeFiles/{nome}.shp")
            print(f"✅ Sucesso: {nome}.shp gerado com {len(vias)} segmentos.")
            
        except Exception as e:
            print(f"❌ Erro ao processar {nome}: {e}")

if __name__ == "__main__":
    gerar_arquivos_logistica_final()