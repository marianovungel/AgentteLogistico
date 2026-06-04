# 🌾 AGRO M2 — Apoio à Decisão Logística com Score, Geolocalização e IA

## 📋 Descrição

Este projeto tem como objetivo desenvolver uma **Prova de Conceito de apoio à decisão logística** no contexto do agronegócio, integrando:

- **score analítico de produtores**;
- **geolocalização**;
- **cálculo de rotas logísticas**;
- **visualização interativa em mapa**;
- **assistente de Inteligência Artificial com RAG local**.

A solução foi implementada em **Streamlit** e utiliza dados tabulares, dados geoespaciais e modelos preditivos para apoiar a escolha de produtores a partir de um ponto de destino definido pelo usuário.

---

## 🎯 Objetivos da Solução

1. Apoiar a seleção de produtores a partir de um ponto de destino.
2. Integrar **score analítico** e **viabilidade logística** em uma mesma análise.
3. Permitir filtragem espacial por raio e por tipo de produto.
4. Calcular rotas com base em **malha rodoviária e ferroviária**.
5. Disponibilizar uma camada de **IA explicativa**, capaz de gerar insights e responder perguntas com base nos dados atuais e em documentos técnicos.

---

## 🏗️ Estrutura do Projeto

```text
AGRO_M2/
│
├── Last_version.py                  # Arquivo principal da aplicação
├── mainChat.py                      # Camada de orquestração do chat com IA
├── vectorChat.py                    # Camada RAG / base vetorial
├── price.py                         # Faixas de preço por cultura
│
├── df_agr.csv                       # Base principal dos agricultores
├── classificador_vendedor.pkl       # Modelo preditivo do vendedor
├── classificador_comprador.pkl      # Modelo do comprador (não usado no fluxo principal validado)
│
├── docs/                            # PDFs usados como contexto técnico da IA
│   ├── arquivo_1.pdf
│   ├── arquivo_2.pdf
│   └── ...
│
├── chroma_langchain_db/             # Base vetorial persistida do Chroma
│   ├── chroma.sqlite3
│   └── ...
│
├── ShapeFiles/                      # Shapefiles dos agricultores
│   ├── agricultores_agro_m2.cpg
│   ├── agricultores_agro_m2.dbf
│   ├── agricultores_agro_m2.prj
│   ├── agricultores_agro_m2.shp
│   └── agricultores_agro_m2.shx
│
├── ShapeFiles_RF/                   # Malha logística rodoviária e ferroviária
│   ├── rod_trecho_rodoviario_l.cpg
│   ├── rod_trecho_rodoviario_l.dbf
│   ├── rod_trecho_rodoviario_l.prj
│   ├── rod_trecho_rodoviario_l.shp
│   ├── rod_trecho_rodoviario_l.shx
│   ├── fer_trecho_ferroviario_l.cpg
│   ├── fer_trecho_ferroviario_l.dbf
│   ├── fer_trecho_ferroviario_l.prj
│   ├── fer_trecho_ferroviario_l.shp
│   └── fer_trecho_ferroviario_l.shx
│
├── EDA.ipynb                        # Notebook exploratório
├── Vendedor.ipynb                   # Notebook do pipeline do vendedor
├── Comprador.ipynb                  # Notebook do pipeline do comprador
├── app.py                           # Arquivo existente no projeto, não utilizado no fluxo principal validado
├── modais.py                        # Arquivo auxiliar/alternativo, não utilizado no fluxo principal validado
└── requirements.txt                 # Dependências do projeto
```

---

## 📥 Arquivos necessários para execução

### 1. Shapefiles da malha logística

Baixe os arquivos da malha rodoviária e ferroviária e coloque o conteúdo extraído dentro da pasta `ShapeFiles_RF/`, na raiz do projeto:

[ShapeFiles_RF no Google Drive](https://drive.google.com/drive/folders/1vhEJVRJfVqSTy1nZcv8-MoXT0E8F8Np0?usp=sharing)

### 2. Shapefiles dos agricultores

Baixe os arquivos dos agricultores e coloque o conteúdo extraído dentro da pasta `ShapeFiles/`, na raiz do projeto:

[ShapeFiles no Google Drive](https://drive.google.com/drive/folders/1gBUbSqugtd6DZDHoQKH5ctnf40RMFE-A?usp=sharing)

### 3. Base de dados CSV

Baixe os arquivos CSV e coloque-os **na raiz do projeto**, conforme a versão funcional validada:

[Datasets CSV no Google Drive](https://drive.google.com/drive/folders/136zoQyQ9DODiK03KkIC7se77Gx4fDoDh?usp=sharing)

> **Observação:** para a pipeline principal validada, o arquivo mais importante é o `df_agr.csv`.

---

## 🚀 Requisitos

Antes de executar o projeto, é necessário ter instalado:

- Python 3.10+
- pip
- ambiente virtual Python
- Ollama
- modelos Ollama:
  - `llama3.2`
  - `mxbai-embed-large`

---

## ⚙️ Configuração do ambiente

### Linux / Mac

```bash
cd AGRO_M2

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

### Windows CMD

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
pip install notebook
```

---

## 🤖 Configuração da IA local com Ollama

Instale o Ollama conforme o sistema operacional e, em seguida, baixe os modelos exigidos pelo projeto.

### Modelos necessários

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Verificação

```bash
ollama list
```

Se os dois modelos aparecerem na lista, a camada de IA local estará pronta para uso.

---

## ▶️ Como executar

### Aplicação principal

A versão funcional validada do projeto utiliza o arquivo `Last_version.py` como ponto de entrada principal:

```bash
streamlit run Last_version.py
```

### Notebooks auxiliares

Caso queira explorar os notebooks do projeto:

```bash
jupyter notebook EDA.ipynb
jupyter notebook Vendedor.ipynb
jupyter notebook Comprador.ipynb
```

---

## 🧭 Fluxo de utilização pelo usuário

1. O usuário define o **modal prioritário** na barra lateral.
2. Em seguida, informa ou ajusta o **ponto de destino** por latitude e longitude.
3. Também pode alterar o ponto clicando diretamente no mapa.
4. Seleciona o **raio de busca** e os **produtos de interesse**.
5. A aplicação filtra os agricultores compatíveis com os critérios.
6. O modelo preditivo estima o **score** dos registros selecionados.
7. O sistema calcula a **distância real na malha logística**.
8. Os resultados são apresentados em **mapa** e **tabela consolidada**.
9. Por fim, o usuário pode acionar o botão de insight ou utilizar o chat para obter uma recomendação explicada pela IA.

---

## 📊 Principais funcionalidades

### Aplicação principal (`Last_version.py`)

- Dashboard Streamlit operacional para análise logística de agricultores.
- Filtro espacial por raio com atualização dinâmica do ponto de destino.
- Visualização cartográfica com marcadores, área de abrangência e rotas.
- Integração de malha multimodal com preferência entre rodovia e ferrovia.
- Score preditivo aplicado aos agricultores mais relevantes do cenário.
- Tabela consolidada com métricas de distância e composição modal da rota.
- Assistente de IA para recomendação e interpretação do resultado final.
- Estrutura de RAG local baseada em PDFs e embeddings persistidos.

### Notebooks

- Análise exploratória de dados.
- Estudos e testes metodológicos.
- Desenvolvimento dos pipelines de vendedor e comprador.
- Apoio à construção do score bidimensional.

---

## 🛠️ Tecnologias utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **GeoPandas**
- **Shapely**
- **NetworkX**
- **momepy**
- **SciPy / KDTree**
- **Folium**
- **Streamlit**
- **streamlit-folium**
- **Joblib**
- **LangChain**
- **Chroma**
- **Ollama**
- **Jupyter Notebook**

---

## 📝 Observações importantes

- O arquivo `app.py` existe no projeto, mas **não foi utilizado no fluxo principal validado**.
- O arquivo `modais.py` foi analisado como alternativa, mas **não foi necessário** para a execução da versão funcional final, pois a solução utilizou shapefiles já baixados.
- A execução da camada de IA depende do Ollama instalado e dos modelos corretos já baixados.
- A pasta `docs/` deve conter os PDFs utilizados como contexto documental pela IA.
- A pasta `chroma_langchain_db/` pode ser criada automaticamente na primeira indexação dos documentos.

---

## 👥 Autores

- **Mariano António Vunge- Vungel**
- **Ana Laura Canassa Basseto - Ana Laura**

---

## ✅ Versão validada da pipeline

A pipeline funcional validada nesta versão utiliza os seguintes arquivos `.py`:

- `Last_version.py`
- `mainChat.py`
- `vectorChat.py`
- `price.py`
