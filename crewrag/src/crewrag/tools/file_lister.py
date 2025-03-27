"""
File Lister Tool - Only lists files in a directory
"""
from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
import os
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FileLister")

class FileListerInput(BaseModel):
    folder_path: str = Field(..., description="Path to the folder to list files from.")
    file_types: List[str] = Field(default=["pdf", "md"], description="File extensions to look for.")

class FileListerTool(BaseTool):
    name: str = "File Lister Tool"
    description: str = "Lists files in a directory. Only lists files, doesn't process content."
    args_schema: Type[BaseModel] = FileListerInput
    
    def _run(self, folder_path: str, file_types: List[str] = ["pdf", "md"]) -> str:
        """List all files of specified types in the specified folder."""
        logger.info(f"Listing files in folder: {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            error_msg = f"Error: The folder path '{folder_path}' does not exist."
            logger.error(error_msg)
            return error_msg
        
        # Format file types for glob patterns
        glob_patterns = [f"**/*.{ft}" if not ft.startswith('.') else f"**/*{ft}" for ft in file_types]
        
        # Store results by file type
        files_by_type: Dict[str, List[str]] = {}
        
        # Get all files of each type using glob (recursive)
        for pattern in glob_patterns:
            file_type = pattern.split('.')[-1]
            matching_files = glob.glob(os.path.join(folder_path, pattern), recursive=True)
            files_by_type[file_type] = matching_files
        
        # Format the response
        response = f"Files found in {folder_path}:\n\n"
        total_files = 0
        
        for file_type, files in files_by_type.items():
            if files:
                response += f"{file_type.upper()} Files ({len(files)}):\n"
                for i, file_path in enumerate(files, 1):
                    file_name = os.path.basename(file_path)
                    response += f"{i}. {file_name}\n"
                response += f"\n"
                total_files += len(files)
        
        if total_files == 0:
            response = f"No files of types {', '.join(file_types)} found in {folder_path}"
        else:
            response += f"Total: {total_files} file(s)"
            
        return response 