#!/usr/bin/env python
import sys
import warnings
from crew import ContentRAGCrew
import os
import traceback
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
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
        
        print("\n=== Enhanced Content RAG Chatbot ===")
        print("This bot can process both PDF and Markdown files.")
        print("Type 'exit' to quit, 'files' to list available files.")
        print("For best results with data extraction, ask questions like:")
        print(" - 'What are all the payor names and PA statuses across all files?'")
        print(" - 'Extract all CPT codes and their PA status from all documents'\n")
        
        # Conversation context
        conversation_history = []
        
        while True:
            try:
                question = input("\nYour question: ")
                
                # Handle special commands
                if question.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif question.lower() in ['files', 'list files', 'show files']:
                    # Special command to list files
                    files_question = "What files are available?"
                    inputs = {
                        "folder_path": folder_path,
                        "question": files_question
                    }
                    result = crew_instance.kickoff(inputs=inputs)
                    print(f"\nFiles in {folder_path}:\n{result}")
                    continue
                    
                # Regular question handling
                if question.strip():
                    # Determine if this is a data extraction query
                    is_extraction_query = any(keyword in question.lower() for keyword in 
                                           ['extract', 'list all', 'table', 'from all files', 
                                           'across all', 'every file', 'all documents', 'payor name'])
                    
                    # Add context from previous conversation
                    context = ""
                    if conversation_history and len(conversation_history) > 0:
                        context = "Based on our previous conversation, "
                    
                    # Ensure the question is specific
                    if len(question.strip()) < 15 and not is_extraction_query:
                        print("\nPlease provide a more specific question for better results.")
                        continue
                        
                    inputs = {
                        "folder_path": folder_path,
                        "question": question
                    }
                    
                    print("\nProcessing your question...")
                    results = crew_instance.kickoff(inputs=inputs)
                    
                    # Store conversation for context
                    conversation_history.append({"question": question, "answer": results})
                    
                    # Format and display the result
                    if isinstance(results, dict):
                        print("\nFindings from your documents:")
                        for task_id, result in results.items():
                            if 'extraction' in task_id.lower():
                                print("\n=== Extracted Data ===")
                            else:
                                print("\n=== General Analysis ===")
                            print(result)
                    else:
                        print("\nAnswer:")
                        print(results)
            except KeyboardInterrupt:
                print("\nOperation interrupted. Type 'exit' to quit or ask another question.")
                continue
            except Exception as e:
                logger.error(f"Error processing your question: {e}")
                traceback.print_exc()
                print("\nPlease try again with a different question.")
                
    except Exception as e:
        logger.error(f"Error initializing the chatbot: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run()
