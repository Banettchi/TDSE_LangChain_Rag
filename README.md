# Proyecto RAG con Pinecone y LangChain

Este repositorio contiene el código y la documentación para el proyecto completo del Generador Aumentado por Recuperación (RAG) usando OpenAI y el almacén de vectores Pinecone, parte del laboratorio "Introducción a la Creación de RAGs (Generadores Aumentados por Recuperación) con OpenAI".

## Arquitectura y Componentes

Este proyecto implementa un flujo RAG completo:

1.  **Cargador de Documentos (`WebBaseLoader`)**: Obtiene contenido de una fuente externa (en este caso, un post de blog sobre agentes autónomos).
2.  **Divisor de Texto (`RecursiveCharacterTextSplitter`)**: Fragmenta (chunk) el documento grande en piezas más pequeñas y manejables (1000 caracteres con 200 caracteres de superposición) para prepararlos para la incrustación (embedding).
3.  **Embeddings (`OpenAIEmbeddings`)**: Convierte los fragmentos de texto en representaciones matemáticas densas (vectores) para que puedan ser buscados semánticamente.
4.  **Almacén de Vectores (`PineconeVectorStore`)**: Una base de datos de vectores en la nube que almacena los fragmentos incrustados y proporciona una búsqueda rápida de vecinos más cercanos para recuperar contenido relevante.
5.  **Recuperador (`as_retriever`)**: Conecta el almacén de vectores a la cadena, consultando a Pinecone con la pregunta del usuario para recuperar los 3 fragmentos de documento más relevantes.
6.  **Modelo de Lenguaje (`ChatOpenAI`)**: El LLM (gpt-3.5-turbo) que sintetiza una respuesta final usando *solo* el contexto proporcionado por el recuperador, guiado por una plantilla específica (`ChatPromptTemplate`).

## Requisitos Previos
* Python 3.8+
* Una clave API activa de [OpenAI](https://platform.openai.com/api-keys).
* Una clave API activa de [Pinecone](https://app.pinecone.io/) y configuración del entorno.

## Instrucciones de Instalación

1.  **Abre el directorio**: Asegúrate de estar en el directorio raíz de este repositorio (donde se encuentran este `README.md` y `requirements.txt`).
2.  **Crear un entorno virtual (opcional pero recomendado)**:
    ```bash
    python -m venv venv
    # En Windows
    .\venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```
3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configurar claves de API**:
    Crea un archivo llamado `.env` en la raíz del directorio y añade tus claves:
    ```env
    OPENAI_API_KEY="tu-clave-openai"
    PINECONE_API_KEY="tu-clave-pinecone"
    ```
5.  **Configurar Pinecone**:
    Debes crear un índice en tu consola de Pinecone antes de ejecutar este código.
    *   **Nombre del Índice**: `lab-rag-index` (o modifica `index_name` en `main.py`).
    *   **Dimensiones**: `1536` (requerido para `text-embedding-ada-002` de OpenAI).
    *   **Métrica**: `cosine`

## Cómo Ejecutar

Ejecuta el script principal:
```bash
python main.py
```

## Ejemplo de Acción / Salida
```
--- Flujo Completo RAG con LangChain y Pinecone ---

[1] Cargando Documento...
Se cargaron 1 documentos.

[2] Dividiendo el Documento en Fragmentos (Chunks)...
Documento dividido en 66 fragmentos.

[3] Inicializando Embeddings y Almacén de Vectores (Pinecone)...
Conexión exitosa al índice de Pinecone: 'lab-rag-index'

[4] Configurando la Cadena de Recuperación...

--- Probando el Sistema RAG ---
Pregunta: ¿Qué es la Descomposición de Tareas (Task Decomposition)?

Recuperando contexto y generando respuesta...

[Respuesta]:
La Descomposición de Tareas es una técnica que divide tareas complejas en pasos más pequeños y simples, haciéndolas más manejables para un agente o modelo. Este proceso implica pensar paso a paso y puede facilitarse mediante prompting simple, instrucciones específicas para la tarea, o aportes humanos. Al descomponer las tareas, arroja luz sobre el proceso de razonamiento y ayuda a los agentes a planificar de manera efectiva.
```
