"""
PDF Processor Tool - Only processes PDF files
"""
from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
import os
import logging
import time
import glob

# LlamaParse import
from llama_parse import LlamaParse
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDFProcessor")

class PDFProcessorInput(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing PDF files.")

class PDFProcessorTool(BaseTool):
    name: str = "PDF Processor Tool"
    description: str = "Processes PDF files from a folder and returns document objects. Only processes PDF files."
    args_schema: Type[BaseModel] = PDFProcessorInput
    
    # Store processed documents
    _processed_pdfs: Dict[str, Document] = PrivateAttr(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._processed_pdfs = {}
        
    def _run(self, folder_path: str) -> str:
        """Process all PDF files in the specified folder."""
        logger.info(f"Processing PDF files in folder: {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            error_msg = f"Error: The folder path '{folder_path}' does not exist."
            logger.error(error_msg)
            return error_msg
            
        # Get all PDF files using glob (recursive)
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        
        if not pdf_files:
            msg = f"No PDF files found in {folder_path}"
            logger.info(msg)
            return msg
            
        # Process each PDF file
        processed_count = 0
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing PDF file: {pdf_path}")
                
                # Use LlamaParse to extract text
                result = LlamaParse(result_type="markdown").load_data(pdf_path)
                
                if isinstance(result, list):
                    extracted_text = "\n".join([doc.get_content() for doc in result])
                elif isinstance(result, str):
                    extracted_text = result
                else:
                    extracted_text = ""
                    
                if not extracted_text.strip():
                    logger.warning(f"No text extracted from {pdf_path}")
                    continue
                
                # Create document with metadata
                file_name = os.path.basename(pdf_path)
                metadata = {
                    "file_name": file_name,
                    "file_path": pdf_path,
                    "file_type": "pdf",
                    "creation_time": time.ctime(os.path.getctime(pdf_path)),
                    "modified_time": time.ctime(os.path.getmtime(pdf_path)),
                }
                
                # Store document
                self._processed_pdfs[file_name] = Document(
                    page_content=extracted_text,
                    metadata=metadata
                )
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}")
                
        return f"Successfully processed {processed_count} PDF files out of {len(pdf_files)} found."
        
    def get_processed_documents(self) -> Dict[str, Document]:
        """Return all processed PDF documents."""
        return self._processed_pdfs 