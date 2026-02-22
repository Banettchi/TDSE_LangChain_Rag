import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    print("=" * 70)
    print("  PROYECTO RAG COMPLETO CON LANGCHAIN, PINECONE Y GEMINI")
    print("=" * 70)
    
    # 1. Cargar Variables de Entorno (Claves API)
    load_dotenv()
    
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
        print("Error: Faltan Claves API. Por favor revisa tu archivo .env.")
        print("Requiere: GOOGLE_API_KEY y PINECONE_API_KEY")
        return
         
    index_name = "lab-rag-index"

    # =====================================================
    # SECCIÓN 1: Explicación teórica de RAG
    # =====================================================
    print("\n" + "=" * 70)
    print("  SECCIÓN 1: ¿Qué es RAG? (Generación Aumentada por Recuperación)")
    print("=" * 70)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    parser = StrOutputParser()

    prompt_explicacion = ChatPromptTemplate.from_messages([
        ("system", "Eres un profesor experto en IA. Responde siempre en español de forma detallada y bien estructurada, usando negritas (**texto**) y listas numeradas."),
        ("user", "{text}")
    ])

    chain_explicacion = prompt_explicacion | llm | parser

    try:
        explicacion_rag = chain_explicacion.invoke({
            "text": "Explica de forma detallada qué es Retrieval-Augmented Generation (RAG). Incluye: qué es la Recuperación (Retrieval), qué es la Generación (Generation), cuáles son los componentes de un modelo RAG, una descripción de alto nivel del proceso RAG paso a paso y sus ventajas."
        })
        print(explicacion_rag)
    except Exception as e:
        print(f"Error al generar explicación: {e}")
        return

    # =====================================================
    # SECCIÓN 2: Carga y procesamiento de documentos
    # =====================================================
    print("\n" + "=" * 70)
    print("  SECCIÓN 2: Carga y Procesamiento de Documentos")
    print("=" * 70)

    print("\n[1] Cargando documento desde la web...")
    try:
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        docs = loader.load()
        print(f"    -> Se cargaron {len(docs)} documentos exitosamente.")
    except Exception as e:
        print(f"Error al cargar el documento: {e}")
        return

    print("\n[2] Dividiendo el documento en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"    -> Documento dividido en {len(splits)} fragmentos.")
    print(f"    -> Tamaño de chunk: 1000 caracteres | Superposición: 200 caracteres")

    # =====================================================
    # SECCIÓN 3: Embeddings y almacenamiento en Pinecone
    # =====================================================
    print("\n" + "=" * 70)
    print("  SECCIÓN 3: Embeddings y Almacenamiento en Pinecone")
    print("=" * 70)

    print("\n[3] Inicializando modelo de embeddings (Google: models/gemini-embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    try:
        print(f"[4] Conectando al índice de Pinecone: '{index_name}'...")
        print(f"    -> Almacenando {len(splits)} fragmentos como vectores...")
        vectorstore = PineconeVectorStore.from_documents(
            splits, 
            embeddings, 
            index_name=index_name
        )
        print(f"    -> ¡Conexión exitosa! Vectores almacenados en Pinecone.")
    except Exception as e:
        print(f"Error conectando a Pinecone: {e}")
        print(f"Asegúrate de haber creado un índice llamado '{index_name}' en la consola de Pinecone.")
        return

    # =====================================================
    # SECCIÓN 4: Cadena de Recuperación RAG
    # =====================================================
    print("\n" + "=" * 70)
    print("  SECCIÓN 4: Consultas al Sistema RAG")
    print("=" * 70)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """Eres un asistente educativo experto. Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta de forma detallada y bien estructurada en español. 
Usa negritas (**texto**) y listas numeradas para organizar tu respuesta.
Si no sabes la respuesta, simplemente di que no lo sabes.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- Pregunta 1 ---
    pregunta1 = "¿Qué es la Descomposición de Tareas (Task Decomposition) y qué técnicas se usan?"
    print(f"\n{'-' * 70}")
    print(f"  PREGUNTA 1: {pregunta1}")
    print(f"{'-' * 70}")
    try:
        respuesta1 = rag_chain.invoke(pregunta1)
        print(respuesta1)
    except Exception as e:
        print(f"Error: {e}")

    # --- Pregunta 2 ---
    pregunta2 = "¿Qué tipos de memoria se mencionan para los agentes autónomos?"
    print(f"\n{'-' * 70}")
    print(f"  PREGUNTA 2: {pregunta2}")
    print(f"{'-' * 70}")
    try:
        respuesta2 = rag_chain.invoke(pregunta2)
        print(respuesta2)
    except Exception as e:
        print(f"Error: {e}")

    # --- Pregunta 3 ---
    pregunta3 = "¿Cuáles son los desafíos y limitaciones de los agentes autónomos basados en LLMs?"
    print(f"\n{'-' * 70}")
    print(f"  PREGUNTA 3: {pregunta3}")
    print(f"{'-' * 70}")
    try:
        respuesta3 = rag_chain.invoke(pregunta3)
        print(respuesta3)
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("  FIN DEL PROYECTO RAG CON LANGCHAIN Y PINECONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
