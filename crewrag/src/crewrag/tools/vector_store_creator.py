"""
Vector Store Creator Tool - Only creates vector stores from document collections
"""
from crewai.tools import BaseTool
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field, PrivateAttr
import os
import logging
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorStoreCreator")

class VectorStoreCreatorInput(BaseModel):
    documents: Dict[str, Any] = Field(..., description="Dictionary of document objects to create vector store from.")
    chunk_size: int = Field(default=1000, description="Size of text chunks for splitting documents.")
    chunk_overlap: int = Field(default=200, description="Overlap between text chunks.")

class VectorStoreCreatorTool(BaseTool):
    name: str = "Vector Store Creator Tool"
    description: str = "Creates a vector store from documents. Only creates the vector database, doesn't answer questions."
    args_schema: Type[BaseModel] = VectorStoreCreatorInput
    
    # Store the vector database
    _vector_store: Optional[Any] = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vector_store = None
    
    def _run(self, documents: Dict[str, Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        """Create a vector store from the provided documents."""
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        if not documents:
            return "No documents provided to create vector store from."
        
        # Convert dict values to list if they are documents
        doc_list = []
        for doc_key, doc_value in documents.items():
            if isinstance(doc_value, Document):
                doc_list.append(doc_value)
            else:
                logger.warning(f"Skipping non-Document item: {doc_key}")
        
        if not doc_list:
            return "No valid Document objects found in the provided dictionary."
            
        logger.info(f"Processing {len(doc_list)} Document objects")
        
        # Split documents into chunks
        logger.info(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        try:
            chunks = text_splitter.split_documents(doc_list)
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Create embeddings and vector store
            logger.info("Creating embeddings and vector store")
            if not openai_api_key:
                return "OpenAI API key not found in environment variables."
                
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            self._vector_store = FAISS.from_documents(chunks, embeddings)
            
            return f"Successfully created vector store with {len(chunks)} chunks from {len(doc_list)} documents."
        except Exception as e:
            error_msg = f"Error creating vector store: {e}"
            logger.error(error_msg)
            return error_msg
    
    def get_vector_store(self):
        """Return the created vector store."""
        return self._vector_store 