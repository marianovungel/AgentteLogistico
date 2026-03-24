# mainChat.py atualizado
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorChat import retriever

model = OllamaLLM(model="llama3.2")

# Template atualizado para incluir o contexto dos dados atuais
template = """
Você é o Consultor Estratégico da Agro M2. Sua missão é ajudar o Trader a tomar a melhor decisão de compra.

DADOS ATUAIS DA TABELA (Top Agricultores):
{df_context}

CONHECIMENTO TÉCNICO (PDF):
{context}

Pergunta do usuário: {question}

Instruções de Resposta:
1. Analise os dados da tabela (Dist_Real_KM, Score, Ferrovia, Rodovia).
2. Use o conhecimento do PDF para justificar os pesos das variáveis.
3. Seja direto e recomende um ID específico se solicitado.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def ask_chatbot(question: str, df_data=None) -> str:
    # Busca no PDF
    docs = retriever.invoke(question)
    pdf_context = "\n\n".join([doc.page_content for doc in docs])
    
    # Converte o DataFrame para texto para a IA ler
    df_text = "Nenhum dado de agricultor carregado no momento."
    if df_data is not None:
        df_text = df_data.to_string() # Transforma a tabela em texto legível pela IA

    response = chain.invoke({
        "df_context": df_text,
        "context": pdf_context,
        "question": question
    })

    return response