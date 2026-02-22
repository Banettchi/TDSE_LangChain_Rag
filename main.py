import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    print("--- Flujo Completo RAG con LangChain y Pinecone ---")
    
    # 1. Cargar Variables de Entorno (Claves API)
    # Asegúrate de que OPENAI_API_KEY y PINECONE_API_KEY estén en tu archivo .env
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
         print("Error: Faltan Claves API. Por favor revisa tu archivo .env.")
         print("Requiere: OPENAI_API_KEY y PINECONE_API_KEY")
         return
         
    # Define el Nombre de tu Índice de Pinecone existente aquí
    # (Crea este índice en la consola de Pinecone: dimensiones=1536, métrica=cosine)
    index_name = "lab-rag-index"
    
    print("\n[1] Cargando Documento...")
    try:
        # Cargar un post de blog o documento de muestra
        # (Se requiere BeautifulSoup4 internamente para analizar HTML)
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        docs = loader.load()
        print(f"Se cargaron {len(docs)} documentos.")
    except Exception as e:
        print(f"Error al cargar el documento: {e}")
        return

    print("\n[2] Dividiendo el Documento en Fragmentos (Chunks)...")
    # Dividir el documento en fragmentos más pequeños para el almacén de vectores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Documento dividido en {len(splits)} fragmentos.")

    print("\n[3] Inicializando Embeddings y Almacén de Vectores (Pinecone)...")
    # Configurar los embeddings de OpenAI
    embeddings = OpenAIEmbeddings()
    
    try:
        # Conectar a Pinecone y almacenar los fragmentos
        # Nota: Si el índice ya contiene vectores, esto los añadirá a ellos.
        vectorstore = PineconeVectorStore.from_documents(
            splits, 
            embeddings, 
            index_name=index_name
        )
        print(f"Conexión exitosa al índice de Pinecone: '{index_name}'")
    except Exception as e:
        print(f"Error conectando a Pinecone: {e}")
        print(f"Asegúrate de haber creado un índice llamado '{index_name}' en la consola de Pinecone.")
        return

    print("\n[4] Configurando la Cadena de Recuperación...")
    # Usar el almacén de vectores como un recuperador
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Inicializar el LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Definir una plantilla de prompt para RAG
    template = """Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta. 
Si no sabes la respuesta, simplemente di que no lo sabes. 
Usa tres oraciones como máximo y mantén la respuesta concisa.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Formatear los documentos recuperados en un solo bloque de texto
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Construir la cadena RAG usando LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Consultar la Cadena RAG
    print("\n--- Probando el Sistema RAG ---")
    question = "¿Qué es la Descomposición de Tareas (Task Decomposition)?"
    print(f"Pregunta: {question}")
    
    try:
        print("\nRecuperando contexto y generando respuesta...")
        response = rag_chain.invoke(question)
        print("\n[Respuesta]:")
        print(response)
    except Exception as e:
         print(f"Error consultando la cadena RAG: {e}")

if __name__ == "__main__":
    main()
