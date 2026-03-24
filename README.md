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

### 1. Configuração do Ambiente

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

```

### 1️⃣ Configuração do Ambiente
1. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

### 2. Preparação dos Dados

⚠️ **Importante**: Os dados CSV não estão incluídos no repositório por questões de privacidade e tamanho.

Você precisa colocar os arquivos de dados na pasta `data/`:
- `Agricultor.csv`
- `data/Compradores.csv`  
- `data/crop_yield.csv`
- `data/Loan_default.csv`
- `data/tabela_resumo_10000_linhas.csv`

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
