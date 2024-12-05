# Configuración del entorno
import logging
from pinecone import Pinecone, ServerlessSpec, QueryResponse
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import os
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import Tool


load_dotenv()

logger = logging.getLogger(__name__)

class RAG:

    def __init__(self, embedding_model_name:str, embedding_model_dim:int, index_name:str, groq_model:str):

        self.groq_model = groq_model
        groq_api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(
            api_key=groq_api_key,
        )

        self.embedding_model_name = embedding_model_name
        self.embedding_model_dim = embedding_model_dim

        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.index_name = index_name

        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Valida si el index existe o no
        existing_indexes = self.pc.list_indexes()
        index_names = [index.name for index in existing_indexes]

        if index_name not in index_names:
            logger.info("Index doesn't exist. Creating!")
            self.pc.create_index(
                name=index_name,
                dimension=embedding_model_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )
        else:
            logger.info("Index with name " + index_name + " already created. Skipping!")

        # Cargando el index de Pinecone
        self.index = self.pc.Index(self.index_name)


    def _extract_text_from_pdf(self, file_path:str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text


    def _generate_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

        
    def load_documents(self, pdf_dir:Path):
        # Carga y procesamiento de PDFs
        # pdf_dir = "docs"  # Directorio donde se almacenan los PDFs
        # pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

        pdf_files = [f.name for f in pdf_dir.iterdir() if f.is_file() and f.suffix == ".pdf"]

        # Procesar todos los PDFs
        documents = []
        for pdf in pdf_files:
            text = self._extract_text_from_pdf(os.path.join(pdf_dir, pdf))
            documents.append({"filename": pdf, "text": text})

        # Documentos cargados en el directorio
        documents_names = [document['filename'] for document in documents]

        # Documentos cargados en Pinecone
        existing_indexes = self.pc.list_indexes()
        index_names = [index.name for index in existing_indexes]

        for document_name in documents_names:
            logger.info("Processing document: " + document_name)

            dummy_vector = [0.0] * 384

            # Realiza la búsqueda con el filtro en el campo 'filename' de la metadata
            query_results = self.index.query(
                vector=dummy_vector,
                top_k=10,  # Número máximo de resultados a devolver
                filter={'filename': {'$eq': document_name}},  # Filtro de metadata
                include_metadata=True  # Incluir metadata en los resultados
            )

            # # Imprime los resultados
            # for result in query_results['matches']:
            #     print(f"ID: {result['id']}")
            #     print(f"Filename: {result['metadata']['filename']}")
            #     print(f"Score: {result['score']}")

            if query_results['matches']:
                logger.info(f"File '{document_name}' found on index. Upserting aborted...")
            else:
                logger.info(f"File '{document_name}' not found. Initializing upserting!!..")

                # Chunking recursivo
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = []

                doc = next((doc for doc in documents if doc['filename'] == document_name), None)

                doc_chunks = text_splitter.split_text(doc["text"])
                chunks.extend([{"filename": doc["filename"], "chunk": chunk} for chunk in doc_chunks])
                
                for chunk in chunks:
                    chunk["embedding"] = self._generate_embedding(chunk["chunk"])

                # Subir los vectores a Pinecone
                for i, chunk in enumerate(chunks):
                    self.index.upsert([(str(i), chunk["embedding"], {"filename": chunk["filename"], "chunk": chunk["chunk"]})])
                
                logger.info("Upserting ended.")


    # Probar una consulta
    def _find_similar(self, query:str) -> QueryResponse:
        query_embedding = self._generate_embedding(query)
        response = self.index.query(vector=query_embedding, top_k=1, include_metadata=True)
        # sample: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace='my_namespace')
        logger.debug(response)
        return response


    def ask(self, query:str) -> str:
    
        """
        Realiza una consulta sobre la base de datos del RAG.

        Args:
            query (str): La cadena de texto que representa la consulta a realizar. Debe ser una pregunta o comando que el sistema puede interpretar.

        Returns:
            str: Texto con la respuesta.
        """
        response = self._find_similar(query)
        context = response['matches'][0]['metadata']['chunk']

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Answer this question: '"+query+"' using this information: " + context,
                }
            ],
            model=self.groq_model,
        )
        return (chat_completion.choices[0].message.content)
    

    def __call__(self, query):
        """
        Permite invocar la clase como una herramienta para responder consultas.
        :param query: Pregunta realizada.
        :return: Respuesta generada por el LLM.
        """
        return self.ask( query)



if __name__ == "__main__":

    # Inicializar la clase RAG
    groq_model="llama3-8b-8192"
    embedding_model_name="all-MiniLM-L6-v2"
    embedding_model_dim=384
    index_name = "cvs-embeddings"
    rag = RAG(embedding_model_name, embedding_model_dim, index_name, groq_model)
    path = Path('docs/')
    rag.load_documents(path)

    # Registrar como herramienta
    rag_tool_langchain = Tool(
        name="RAG_Tool",
        func=rag,
        description="Una herramienta para responder preguntas utilizando recuperación de contexto y un modelo LLM en Groq."
    )

    # Inicializar un LLM base
    ChatGroq.api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model_name="llama3-8b-8192")

    # Inicializar el agente con la herramienta RAG
    tools = [rag_tool_langchain]
    agent = initialize_agent(
        tools=tools,
        llm=llm,  # Se pasa el modelo base como argumento
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    query = "Does Javier know Java programming?"
    # Ejemplo de consulta al agente
    response = agent.run(query)
    print(response)
