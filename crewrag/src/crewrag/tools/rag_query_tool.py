"""
RAG Query Tool - Only answers questions using a vector store retriever
"""
from crewai.tools import BaseTool
from typing import Type, Optional, Any
from pydantic import BaseModel, Field, PrivateAttr
import os
import logging
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGQuery")

class RAGQueryInput(BaseModel):
    question: str = Field(..., description="Question to ask the RAG system.")
    vector_store: Any = Field(..., description="Vector store to query.")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model to use for RAG.")
    temperature: float = Field(default=0.7, description="Temperature for the LLM response.")
    search_k: int = Field(default=5, description="Number of documents to retrieve from vector store.")

class RAGQueryTool(BaseTool):
    name: str = "RAG Query Tool"
    description: str = "Answers questions using a vector store retriever. Only answers questions, doesn't process files."
    args_schema: Type[BaseModel] = RAGQueryInput
    
    # Store conversation chain
    _chat_chain: Optional[ConversationalRetrievalChain] = PrivateAttr(default=None)
    _memory: Optional[ConversationBufferMemory] = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chat_chain = None
        self._memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def _run(self, question: str, vector_store: Any, model_name: str = "gpt-4o-mini", 
             temperature: float = 0.7, search_k: int = 5) -> str:
        """Answer a question using the RAG system."""
        logger.info(f"Answering question using RAG: '{question}'")
        
        if not vector_store:
            return "No vector store provided to query."
            
        if not openai_api_key:
            return "OpenAI API key not found in environment variables."
            
        try:
            # Check if we need to create a new chain or can reuse existing one
            if not self._chat_chain or hasattr(self, '_last_model') and self._last_model != model_name:
                logger.info(f"Creating new RAG chain with model {model_name}")
                llm = ChatOpenAI(
                    model=model_name, 
                    temperature=temperature, 
                    openai_api_key=openai_api_key
                )
                
                # Set up the retrieval chain
                self._chat_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_store.as_retriever(search_kwargs={"k": search_k}),
                    memory=self._memory
                )
                setattr(self, '_last_model', model_name)
                
            # Process the question
            logger.info(f"Retrieving information and generating answer")
            response = self._chat_chain({"question": question})
            
            logger.info("Generated response successfully")
            return response["answer"]
            
        except Exception as e:
            error_msg = f"Error answering question: {e}"
            logger.error(error_msg)
            return error_msg
            
    def clear_conversation_history(self):
        """Clear the conversation history."""
        if self._memory:
            self._memory.clear()
            return "Conversation history cleared."
        return "No conversation history to clear." 