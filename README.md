# 🌾 AGRO M2 — Dashboard de Saúde e Score Bidirecional

## 📋 Descrição
Este projeto tem como objetivo desenvolver um **dashboard bidimensional de score** para **agricultores e compradores**, utilizando indicadores financeiros, agrícolas e comportamentais. O projeto permite análise exploratória, visualização interativa e avaliação de confiabilidade e performance dos usuários.  

O dashboard é construído em **Streamlit** e os cálculos iniciais podem ser feitos em **Jupyter Notebook**.

---

## 🎯 Objetivos da Análise
1. Avaliar a confiabilidade financeira e comportamental de agricultores e compradores.  
2. Criar métricas de score bidimensional para tomada de decisão.  
3. Visualizar clusters e relações entre variáveis-chave.  
4. Facilitar análises exploratórias e relatórios interativos.

---

## 🏗️ Estrutura do Projeto

```
SCORE/
│
├── data/
│ ├── Agricultor.csv
│ ├── Compradores.csv
| ├── AHP_Criterios_Comprador_Preenchido.xlsx
│ ├── crop_yield.csv
│ ├── Loan_default.csv
│ └── tabela_resumo_10000_linhas.csv
|
├── docs/
│ ├── Descrição Score Bi Direcional.docx.pdf
│ ├── modelos.pdf
│ └── Relatorio_Sprint1_AGROM2.pdf
|
|
├── ShapeFiles
│ ├── agricultores_agro_m2.cpg
│ ├── agricultores_agro_m2.dbf
│ ├── agricultores_agro_m2.prj
│ ├── agricultores_agro_m2.shp
│ └── agricultores_agro_m2.shx
|
|
├── ShapeFiles_RF
│ ├── agricultores_agro_m2.cpg
│ ├── agricultores_agro_m2.dbf
│ ├── agricultores_agro_m2.prj
│ ├── agricultores_agro_m2.shp
│ └── agricultores_agro_m2.shx
|
├── chroma_langchain_db/
│ ├── 26d5bb78-f1ea-4dfc-89f1-0df2de32c400/
│ └── chroma.sqlite3
│
├── venv/ # Ambiente virtual
├── app.py # Dashboard Streamlit
├── Vendedor.ipynb # # Funções de APH + TOPSIS & ML Vendedor
├── Comprador.ipynb # Funções de APH + TOPSIS & ML Comprador
├── grafico_rendimento_por_cluster.png
├── mainChat.py # Notebook principal de análise
├── vectorChat.py # Notebook principal de análise
├── classificador_comprador.pkl # Modelo de Classificação Comprador
├── EDA.ipynb # Notebook de análise e pesquisa do uso da clusterização
├── classificador_vendedor.pkl # Modelo de Classificação Vendedor
└── requirements.txt
```

## 🚀 Como Executar

### 1. Configuração do Ambiente em Ambiente LINUX/MAC

```bash
# Entrar no repositório
cd SCORE

# Criar ambiente virtual
sudo apt update
sudo apt install python3 python3-pip
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install --upgrade pip
pip3 install notebook

# ou
.venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt


# Baixar ShapeFiles_RF do IBGE no Google Drive
Link:https://drive.google.com/drive/folders/1vhEJVRJfVqSTy1nZcv8-MoXT0E8F8Np0?usp=sharing
OBS.: Os arquivos devem estar dentro na Pasta ShapeFile_RF e essa pasta deve estar na raiz do projeto. 

# Baixar ShapeFiles do IBGE no Google Drive
Link:https://drive.google.com/drive/folders/1gBUbSqugtd6DZDHoQKH5ctnf40RMFE-A?usp=sharing
OBS.: Os arquivos devem estar dentro na Pasta ShapeFile_RF e essa pasta deve estar na raiz do projeto. 

# Baixar Datasets .csv do IBGE no Google Drive
Link:https://drive.google.com/drive/folders/136zoQyQ9DODiK03KkIC7se77Gx4fDoDh?usp=sharing
OBS.: Os arquivos devem estar  na raiz do projeto, eles não devem estar dentro de pastas.


## 1️⃣ Executar o programa
1. Execute o comando:
streamlit run criar_dataset.py

```
### 3. Executar a Análise


**Linux / Mac:**
```Abra o Jupyter Notebook Vendedor:
jupyter notebook Vendedor.ipynb
```

**Linux / Mac:**
```Abra o Jupyter Notebook Comprador:
jupyter notebook Comprador.ipynb
```

#### 🚀 Execução Automatizada (Recomendado)

**Linux / Mac:**
```bash
source venv/bin/activate
streamlit run app.py
```

**Linux / Mac:**
```bash
source venv/bin/activate
streamlit run criar_dataset.py
```

**Linux / Mac:**
```bash
source venv/bin/activate
jupyter notebook EDA.ipynb
```

**Windows PowerShell:**
```powershell
venv\Scripts\Activate.ps1
streamlit run app.py
```

**Windows CMD:**
```cmd
venv\Scripts\activate.bat
streamlit run app.py
```

Os scripts automaticamente:
- ✅ Verificam Python
- ✅ Criam ambiente virtual
- ✅ Instalam dependências
- ✅ Iniciam aplicação Streamlit

📖 **Para mais detalhes, veja:** [SCRIPTS_README.md](SCRIPTS_README.md)

#### 📓 Execução Manual


**Jupyter Notebook Vendedor:**
```bash
jupyter notebook Vendedor.ipynb
```

**Jupyter Notebook Comprador:**
```bash
jupyter notebook Comprador.ipynb
```

**Aplicação Streamlit:**
```bash
streamlit run app.py
```

**Jupyter Notebook Análise Exploratória:**
```bash
jupyter notebook EDA.ipynb
```

## 📊 Funcionalidades

### Jupyter Notebook
- ✅ Análise exploratória completa
- ✅ Visualizações estáticas e interativas
- ✅ Testes estatísticos
- ✅ Insights automáticos

### Aplicação Streamlit
- ✅ Interface web interativa
- ✅ Upload dos dataset .csv
- ✅ Dashboards interativos

## 📈 Principais Descobertas

### A clusterização não mostrou-se um caminho bom para esta prova.
- **Método AHP mostrou ser um caminho eficiente para a construção do Score**


## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Matplotlib/Seaborn** - Visualizações estáticas
- **Plotly** - Visualizações interativas
- **Streamlit** - Interface web
- **Jupyter** - Notebooks interativos
- **Scipy** - Análises estatísticas
- **Langchain** - Generative AI
- **RAG** - AI Especialista
- **OLLAMA** - Modelo de IA

## 📝 Metodologia

1. **Carregamento**: Dados obtidos pelos Dataset .csv e .json
2. **Limpeza**: Tratamento de valores ausentes e duplicatas
3. **Exploração**: Análise descritiva e visual
6. **Visualização**: Dashboards interativos

## 👥 Autores

- **Mariano António Vunge- Vungel**
- **Ana Laura Canassa Basseto - Ana Laura**

## 📞 Contato

- GitHub: [@marianovungel](https://github.com/marianovungel)
- Portfólio: [Marinao_Portfólio](https://vungel.vercel.app/)

---
