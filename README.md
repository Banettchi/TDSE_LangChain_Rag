# Proyecto RAG con LangChain, Pinecone y Google Gemini

## Introducción
Este proyecto lo desarrollé para entender de forma práctica cómo funciona un sistema **RAG (Retrieval-Augmented Generation)** completo. La idea fue ir más allá de un LLM básico e integrar un componente de recuperación de información que le dé contexto real al modelo antes de generar respuestas.

El flujo que implementé:

- Cargar un documento externo (blog post sobre agentes autónomos).
- Dividirlo en fragmentos (chunks) manejables.
- Convertir cada fragmento en un vector (embedding) con Google Gemini.
- Almacenar los vectores en **Pinecone** (base de datos vectorial en la nube).
- Consultar el sistema con preguntas, donde primero se recupera el contexto relevante y luego el LLM genera una respuesta basada en ese contexto.

Usé **Google Gemini (gemini-2.5-flash)** como LLM y modelo de embeddings, y **Pinecone** como vector store, todo orquestado con **LangChain**.

## Arquitectura del Proyecto
El flujo RAG completo es el siguiente:

```
Documento Web
  → WebBaseLoader (carga)
    → RecursiveCharacterTextSplitter (fragmentación)
      → GoogleGenerativeAIEmbeddings (vectorización)
        → PineconeVectorStore (almacenamiento)

Usuario (pregunta)
  → Retriever (busca fragmentos relevantes en Pinecone)
    → ChatPromptTemplate (combina contexto + pregunta)
      → ChatGoogleGenerativeAI (genera respuesta)
        → StrOutputParser (formatea respuesta)
          → Respuesta impresa en consola
```

Más detallado:

1. Se cargan las API keys de Google y Pinecone desde `.env`.
2. Se carga un documento web usando `WebBaseLoader`.
3. Se divide el documento en chunks de 1000 caracteres con 200 de superposición.
4. Se generan embeddings con `GoogleGenerativeAIEmbeddings` (modelo `gemini-embedding-001`, 3072 dimensiones).
5. Se almacenan los vectores en un índice de Pinecone.
6. Se crea un retriever que busca los 3 fragmentos más relevantes por similitud semántica.
7. Se construye la cadena RAG con LCEL: `retriever | prompt | llm | parser`.
8. El modelo responde usando **solo** el contexto recuperado.

## Archivo Principal

**main.py**

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Cargar y fragmentar documento
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embeddings + Pinecone
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = PineconeVectorStore.from_documents(splits, embeddings, index_name="lab-rag-index")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Cadena RAG
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
template = """Usa el contexto para responder la pregunta...
Contexto: {context}
Pregunta: {question}"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

respuesta = rag_chain.invoke("¿Qué es Task Decomposition?")
print(respuesta)
```

## Requisitos
- Python 3.x
- `langchain`
- `langchain-google-genai`
- `langchain-pinecone`
- `pinecone-client`
- `python-dotenv`
- `beautifulsoup4`

Instalación:
```bash
pip install -r requirements.txt
```

## Variables de Entorno
Archivo `.env`:
```env
GOOGLE_API_KEY=tu-clave-de-google
PINECONE_API_KEY=tu-clave-de-pinecone
```

## Configuración de Pinecone
Antes de ejecutar el código, debes crear un índice en tu [consola de Pinecone](https://app.pinecone.io/):
- **Nombre del Índice**: `lab-rag-index`
- **Dimensiones**: `3072` (requerido para `models/gemini-embedding-001` de Google Gemini)
- **Métrica**: `cosine`

## Cómo lo corrí en mi máquina (Windows)
1. Creé un entorno virtual:
   ```bash
   python -m venv .venv
   ```
2. Activé el entorno:
   ```bash
   .venv\Scripts\Activate.ps1
   ```
3. Instalé dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuté:
   ```bash
   py main.py
   ```
La respuesta se imprime directamente en consola.

## Evidencia de Ejecución
Ejemplo de salida generada:

<img width="1912" height="980" alt="image" src="https://github.com/user-attachments/assets/15655252-72c4-402c-8cff-ebb642f061bd" />


## Conceptos Demostrados
- **Retrieval-Augmented Generation (RAG)**: Combinar recuperación de información con generación de texto.
- **Document Loading**: Carga de documentos web con `WebBaseLoader`.
- **Text Splitting**: Fragmentación de documentos con `RecursiveCharacterTextSplitter`.
- **Embeddings**: Vectorización de texto con `GoogleGenerativeAIEmbeddings`.
- **Vector Store**: Almacenamiento y búsqueda semántica con **Pinecone**.
- **Retrieval Chain**: Construcción de cadenas de recuperación con LCEL.
- **Prompt Engineering**: Diseño de prompts que combinan contexto recuperado con la pregunta del usuario.

## Conclusión
- **Arquitectura**: Implementa un flujo RAG completo de extremo a extremo, desde la carga del documento hasta la generación de respuestas contextualizadas.
- **Pinecone como Vector Store**: Permite almacenar embeddings en la nube y realizar búsquedas por similitud semántica de forma eficiente.
- **Respuestas fundamentadas**: A diferencia de un LLM simple, el sistema RAG responde basándose en información real del documento, reduciendo alucinaciones.
- **Sin costos**: Usar Google Gemini y el tier gratuito de Pinecone permite ejecutar todo este proyecto sin pagar.
- **Base para producción**: Con esta estructura se pueden construir chatbots, asistentes de documentación, y sistemas de Q&A empresariales.
