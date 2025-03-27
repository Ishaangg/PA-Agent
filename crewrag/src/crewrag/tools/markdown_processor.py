"""
Markdown Processor Tool - Only processes Markdown files
"""
from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
import os
import logging
import time
import glob

from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MarkdownProcessor")

class MarkdownProcessorInput(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing Markdown files.")

class MarkdownProcessorTool(BaseTool):
    name: str = "Markdown Processor Tool"
    description: str = "Processes Markdown files from a folder and returns document objects. Only processes Markdown files."
    args_schema: Type[BaseModel] = MarkdownProcessorInput
    
    # Store processed documents
    _processed_markdowns: Dict[str, Document] = PrivateAttr(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._processed_markdowns = {}
        
    def _run(self, folder_path: str) -> str:
        """Process all Markdown files in the specified folder."""
        logger.info(f"Processing Markdown files in folder: {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            error_msg = f"Error: The folder path '{folder_path}' does not exist."
            logger.error(error_msg)
            return error_msg
            
        # Get all Markdown files using glob (recursive)
        md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)
        
        if not md_files:
            msg = f"No Markdown files found in {folder_path}"
            logger.info(msg)
            return msg
            
        # Process each Markdown file
        processed_count = 0
        for md_path in md_files:
            try:
                logger.info(f"Processing Markdown file: {md_path}")
                
                # Read Markdown file directly
                with open(md_path, 'r', encoding='utf-8') as file:
                    extracted_text = file.read()
                    
                if not extracted_text.strip():
                    logger.warning(f"Empty Markdown file: {md_path}")
                    continue
                
                # Create document with metadata
                file_name = os.path.basename(md_path)
                metadata = {
                    "file_name": file_name,
                    "file_path": md_path,
                    "file_type": "markdown",
                    "creation_time": time.ctime(os.path.getctime(md_path)),
                    "modified_time": time.ctime(os.path.getmtime(md_path)),
                }
                
                # Store document
                self._processed_markdowns[file_name] = Document(
                    page_content=extracted_text,
                    metadata=metadata
                )
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing Markdown {md_path}: {e}")
                
        return f"Successfully processed {processed_count} Markdown files out of {len(md_files)} found."
        
    def get_processed_documents(self) -> Dict[str, Document]:
        """Return all processed Markdown documents."""
        return self._processed_markdowns 