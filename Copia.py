import os

# Caminho da pasta
caminho_pasta = r"/home/vungel/Downloads/M2/Maps/bc250_shapefile._2023_11_23"  # Altere para sua pasta

# Lista para armazenar os nomes dos arquivos
lista_arquivos = []

# Percorre os itens da pasta
for item in os.listdir(caminho_pasta):
    caminho_completo = os.path.join(caminho_pasta, item)
    
    # Verifica se é arquivo (ignora pastas)
    if os.path.isfile(caminho_completo):
        lista_arquivos.append(item)

# Exibe a lista
print(lista_arquivos)
