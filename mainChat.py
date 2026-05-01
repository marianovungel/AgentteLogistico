# ============================================================
# mainChat.py - Sprint 4 (otimizado)
# ------------------------------------------------------------
# Versão otimizada para reduzir o tempo de resposta:
# - envia menos colunas da tabela
# - envia menos linhas
# - limita o contexto documental
# - usa fallback se o RAG falhar
# ============================================================

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorChat import retriever


# ============================================================
# 1. MODELO DE LINGUAGEM
# ============================================================
model = OllamaLLM(model="llama3.2")


# ============================================================
# 2. CONFIGURAÇÕES DE CONTEXTO
# ============================================================
MAX_ROWS_CONTEXT = 5
MAX_DOC_CHARS = 2500

CONTEXT_COLUMNS = [
    "ID",
    "produto_base",
    "preco_produto_base",
    "atratividade_preco",
    "macro_regiao_climatica",
    "precipitacao_7d",
    "temperatura_media_7d",
    "risco_climatico",
    "Dist_Real_KM",
    "Score",
    "score_final_sprint4",
]


# ============================================================
# 3. TEMPLATE DE PROMPT
# ============================================================
template = """
Você é o Consultor Estratégico da Agro M2.
Sua missão é ajudar o Trader a tomar a melhor decisão de compra com base em uma análise integrada.

DADOS ATUAIS DA TABELA:
{df_context}

CONHECIMENTO TÉCNICO:
{context}

Pergunta do usuário:
{question}

Instruções:
1. Analise primeiro os dados atuais da tabela.
2. Considere score, logística, preço e clima de forma integrada.
3. Use o contexto técnico apenas para complementar a análise.
4. Quando recomendar um agricultor, informe explicitamente o ID.
5. Explique de forma objetiva os principais fatores favoráveis e desfavoráveis.
6. Quando houver trade-off entre score, preço, clima e logística, explique o trade-off.
7. Seja direto e claro.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# ============================================================
# 4. PREPARAÇÃO DO CONTEXTO TABULAR
# ============================================================
def build_df_context(df_data):
    """
    Reduz a tabela para poucas colunas e poucas linhas,
    deixando o contexto mais leve para o modelo.
    """
    if df_data is None:
        return "Nenhum dado de agricultor carregado no momento."

    colunas_existentes = [col for col in CONTEXT_COLUMNS if col in df_data.columns]

    if not colunas_existentes:
        return "Nenhuma coluna relevante disponível no dataframe atual."

    df_small = df_data[colunas_existentes].head(MAX_ROWS_CONTEXT).copy()

    # Arredondamento para reduzir ruído visual e textual
    for col in ["preco_produto_base", "Dist_Real_KM", "score_final_sprint4", "atratividade_preco",
                "precipitacao_7d", "temperatura_media_7d", "Score"]:
        if col in df_small.columns:
            df_small[col] = df_small[col].astype(float).round(3)

    return df_small.to_string(index=False)


# ============================================================
# 5. PREPARAÇÃO DO CONTEXTO DOCUMENTAL
# ============================================================
def build_pdf_context(question: str) -> str:
    """
    Recupera contexto do RAG, mas limita o tamanho final
    para evitar prompts excessivamente longos.
    """
    try:
        docs = retriever.invoke(question)
        chunks = []

        for doc in docs:
            content = doc.page_content.strip()
            if content:
                chunks.append(content)

        full_text = "\n\n".join(chunks)

        if not full_text:
            return "Nenhum contexto documental relevante foi recuperado."

        return full_text[:MAX_DOC_CHARS]

    except Exception as exc:
        return f"Contexto documental indisponível no momento. Erro ao consultar retriever: {exc}"


# ============================================================
# 6. FUNÇÃO PRINCIPAL DO CHATBOT
# ============================================================
def ask_chatbot(question: str, df_data=None) -> str:
    """
    Gera resposta com base em:
    - pergunta do usuário
    - dataframe resumido do cenário atual
    - contexto documental resumido
    """
    df_context = build_df_context(df_data)
    pdf_context = build_pdf_context(question)

    response = chain.invoke({
        "df_context": df_context,
        "context": pdf_context,
        "question": question
    })

    return response