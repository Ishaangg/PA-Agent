"""
Tool Orchestrator - Coordinates the individual tools but doesn't do their work
"""
from crewai.tools import BaseTool
from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field, PrivateAttr
import os
import logging

# Import all individual tools
from tools.pdf_processor import PDFProcessorTool
from tools.markdown_processor import MarkdownProcessorTool
from tools.file_lister import FileListerTool
from tools.vector_store_creator import VectorStoreCreatorTool
from tools.rag_query_tool import RAGQueryTool
from tools.data_extraction_tool import DataExtractionTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ToolOrchestrator")

class OrchestratorInput(BaseModel):
    folder_path: str = Field(..., description="Path to the folder to process.")
    action: str = Field(..., description="Action to perform: 'list_files', 'process_files', 'create_db', 'answer', 'extract'")
    question: str = Field(default="", description="Question to answer (for 'answer' action).")
    extraction_type: str = Field(default="all", description="Type of data to extract (for 'extract' action).")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model to use.")

class ToolOrchestratorTool(BaseTool):
    name: str = "Tool Orchestrator"
    description: str = "Coordinates the individual tools but doesn't do their work."
    args_schema: Type[BaseModel] = OrchestratorInput
    
    # Store tool instances and state
    _pdf_processor: Optional[PDFProcessorTool] = PrivateAttr(default=None)
    _md_processor: Optional[MarkdownProcessorTool] = PrivateAttr(default=None)
    _file_lister: Optional[FileListerTool] = PrivateAttr(default=None)
    _vector_store_creator: Optional[VectorStoreCreatorTool] = PrivateAttr(default=None)
    _rag_query: Optional[RAGQueryTool] = PrivateAttr(default=None)
    _data_extractor: Optional[DataExtractionTool] = PrivateAttr(default=None)
    
    # Store processed documents and vector store
    _processed_documents: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _is_db_created: bool = PrivateAttr(default=False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize all tools
        self._pdf_processor = PDFProcessorTool()
        self._md_processor = MarkdownProcessorTool()
        self._file_lister = FileListerTool()
        self._vector_store_creator = VectorStoreCreatorTool()
        self._rag_query = RAGQueryTool()
        self._data_extractor = DataExtractionTool()
        
        # Initialize state
        self._processed_documents = {}
        self._is_db_created = False
    
    def _run(self, folder_path: str, action: str, question: str = "", 
            extraction_type: str = "all", model_name: str = "gpt-4o-mini") -> str:
        """Coordinate tools based on the requested action."""
        logger.info(f"Orchestrating action '{action}' on folder '{folder_path}'")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            error_msg = f"Error: The folder path '{folder_path}' does not exist."
            logger.error(error_msg)
            return error_msg
        
        # Dispatch to appropriate action
        if action == "list_files":
            return self._list_files(folder_path)
        elif action == "process_files":
            return self._process_files(folder_path)
        elif action == "create_db":
            return self._create_vector_db(folder_path)
        elif action == "answer":
            return self._answer_question(folder_path, question, model_name)
        elif action == "extract":
            return self._extract_data(folder_path, extraction_type, model_name)
        else:
            return f"Unknown action: {action}. Valid actions are: list_files, process_files, create_db, answer, extract"
    
    def _list_files(self, folder_path: str) -> str:
        """List files in the folder."""
        logger.info(f"Listing files in {folder_path}")
        return self._file_lister._run(folder_path=folder_path, file_types=["pdf", "md"])
        
    def _process_files(self, folder_path: str) -> str:
        """Process all files in the folder."""
        logger.info(f"Processing all files in {folder_path}")
        
        # Process PDF files
        pdf_result = self._pdf_processor._run(folder_path=folder_path)
        pdf_docs = self._pdf_processor.get_processed_documents()
        
        # Process Markdown files
        md_result = self._md_processor._run(folder_path=folder_path)
        md_docs = self._md_processor.get_processed_documents()
        
        # Combine results
        self._processed_documents = {**pdf_docs, **md_docs}
        
        return f"File processing complete.\n\nPDF Processing: {pdf_result}\n\nMarkdown Processing: {md_result}\n\nTotal documents processed: {len(self._processed_documents)}"
        
    def _create_vector_db(self, folder_path: str) -> str:
        """Create a vector database from processed files."""
        logger.info(f"Creating vector database from files in {folder_path}")
        
        # Process files if not already processed
        if not self._processed_documents:
            processing_result = self._process_files(folder_path)
            logger.info(f"Processing files first: {processing_result}")
            
        if not self._processed_documents:
            return "No documents were processed successfully. Cannot create vector database."
            
        # Create vector store
        result = self._vector_store_creator._run(documents=self._processed_documents)
        if "Successfully created vector store" in result:
            self._is_db_created = True
            
        return result
        
    def _answer_question(self, folder_path: str, question: str, model_name: str) -> str:
        """Answer a question using the RAG system."""
        logger.info(f"Answering question: {question}")
        
        if not question.strip():
            return "No question provided to answer."
            
        # Create vector DB if not already created
        if not self._is_db_created:
            db_result = self._create_vector_db(folder_path)
            logger.info(f"Creating vector database first: {db_result}")
            
        vector_store = self._vector_store_creator.get_vector_store()
        if not vector_store:
            return "Vector store creation failed. Cannot answer question."
            
        # Answer the question
        return self._rag_query._run(
            question=question,
            vector_store=vector_store,
            model_name=model_name
        )
        
    def _extract_data(self, folder_path: str, extraction_type: str, model_name: str) -> str:
        """Extract structured data from processed files."""
        logger.info(f"Extracting {extraction_type} data from files in {folder_path}")
        
        # Process files if not already processed
        if not self._processed_documents:
            processing_result = self._process_files(folder_path)
            logger.info(f"Processing files first: {processing_result}")
            
        if not self._processed_documents:
            return "No documents were processed successfully. Cannot extract data."
            
        # Combine all document contents
        all_content = "\n\n".join([
            f"--- Document: {doc_id} ---\n{doc.page_content}"
            for doc_id, doc in self._processed_documents.items()
        ])
        
        # Extract data
        return self._data_extractor._run(
            content=all_content,
            extraction_type=extraction_type,
            model_name=model_name
        )
        
    def clear_conversation_history(self):
        """Clear the conversation history in the RAG query tool."""
        if self._rag_query:
            return self._rag_query.clear_conversation_history()
        return "RAG query tool not initialized." 