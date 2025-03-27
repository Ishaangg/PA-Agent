#!/usr/bin/env python
import sys
import warnings
from crew import ContentRAGCrew
import os
import traceback
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("content_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ContentBot")

# Suppress SyntaxWarning from pysbd module
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    # Default folder path
    default_folder_path = r'C:\Users\OMEN\Desktop\PA policy'
    
    # Check if default folder exists, if not use knowledge folder
    if not os.path.exists(default_folder_path):
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../knowledge')
        logger.warning(f"Default folder not found. Using {folder_path} instead.")
    else:
        folder_path = default_folder_path
    
    # Ensure folder path exists
    if not os.path.exists(folder_path):
        logger.error(f"Error: Folder path {folder_path} does not exist. Please provide a valid folder path.")
        return
        
    logger.info(f"Using folder path: {folder_path}")
    
    try:
        # Create just one crew instance for the entire session
        crew_instance = ContentRAGCrew().crew()
        
        print("\n=== Enhanced Content RAG Chatbot with Discrete Tools ===")
        print("This bot processes both PDF and Markdown files using specialized tools.")
        print("Available commands:")
        print("  'list' - List all files in the directory")
        print("  'process' - Process all files in the directory")
        print("  'extract [type]' - Extract structured data (payor, pa_status, cpt_codes, or all)")
        print("  'exit' - Exit the program")
        print("  Any other input will be treated as a question to answer\n")
        
        # Flag to track if files have been processed
        files_processed = False
        
        while True:
            try:
                user_input = input("\nEnter command or question: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                    
                elif user_input.lower() == 'list':
                    print("\nListing files...")
                    # Run the list files task
                    inputs = {
                        "folder_path": folder_path,
                        "question": "List all files in the directory"
                    }
                    result = crew_instance.kickoff(inputs=inputs)
                    print("\nFiles in directory:")
                    print(result)
                    continue
                    
                elif user_input.lower() == 'process':
                    print("\nProcessing files...")
                    # Run the process files task
                    inputs = {
                        "folder_path": folder_path,
                        "question": "Process all files in the directory"
                    }
                    result = crew_instance.kickoff(inputs=inputs)
                    print("\nFile processing results:")
                    print(result)
                    files_processed = True
                    continue
                    
                elif user_input.lower().startswith('extract'):
                    # Parse extraction type
                    parts = user_input.split()
                    extraction_type = "all"
                    if len(parts) > 1:
                        extraction_type = parts[1].lower()
                        
                    print(f"\nExtracting {extraction_type} data...")
                    # Run the extract data task
                    inputs = {
                        "folder_path": folder_path,
                        "question": f"Extract {extraction_type} data from all documents",
                        "extraction_type": extraction_type
                    }
                    result = crew_instance.kickoff(inputs=inputs)
                    print("\nExtracted data:")
                    print(result)
                    continue
                
                # Treat any other input as a question
                if user_input.strip():
                    print("\nAnswering your question...")
                    # If files haven't been processed yet, recommend doing so
                    if not files_processed:
                        print("Note: Files have not been processed yet. Processing now...")
                    
                    # Answer the question
                    inputs = {
                        "folder_path": folder_path,
                        "question": user_input
                    }
                    result = crew_instance.kickoff(inputs=inputs)
                    print("\nAnswer:")
                    print(result)
                    
            except KeyboardInterrupt:
                print("\nOperation interrupted. Type 'exit' to quit or ask another question.")
                continue
            except Exception as e:
                logger.error(f"Error processing your request: {e}")
                traceback.print_exc()
                print(f"\nError: {str(e)}")
                print("Please try again with a different command or question.")
                
    except Exception as e:
        logger.error(f"Error initializing the chatbot: {e}")
        traceback.print_exc()
        print(f"\nFatal error: {str(e)}")

if __name__ == "__main__":
    run()
