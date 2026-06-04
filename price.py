# ============================================================
# price.py
# ------------------------------------------------------------
# Dicionário simples com faixas de preço estimadas por cultura.
#
# Estrutura:
# - chave: nome do produto
# - valor: lista com [preço_mínimo, preço_máximo]
#
# Esse arquivo funciona como apoio para regras de negócio,
# simulações ou futuras análises econômicas dentro da aplicação.
# ============================================================

sacas = {
    "Soja": [110, 130],
    "milho": [48, 53],
    "cafe": [1400, 1900],
    "acucar": [90, 100],
    "arroz": [55, 60],
    "vegetal": [80, 90],
    "frutos": [70, 75],
}