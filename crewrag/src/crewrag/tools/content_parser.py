from crewai.tools import BaseTool
from typing import Type, Optional, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
import os
from dotenv import load_dotenv
import glob
import logging
import time
import re

# LlamaParse and LangChain imports
from llama_parse import LlamaParse
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentRAG")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")

# Global cache for ContentRAGTool instances
_content_rag_tool_cache: Dict[str, 'ContentRAGTool'] = {}


class ContentRAGToolInput(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing PDF and Markdown files.")
    question: str = Field(..., description="Question to ask based on the file content.")


class ContentRAGTool(BaseTool):
    name: str = "Content RAG Tool"
    description: str = (
        "Processes PDF and Markdown files from a folder, builds a retrieval chain, and answers questions based on the content."
    )
    args_schema: Type[BaseModel] = ContentRAGToolInput

    # Define internal state as private attributes
    _initialized: bool = PrivateAttr(default=False)
    _chat_chain: Optional[ConversationalRetrievalChain] = PrivateAttr(default=None)
    _folder_path: Optional[str] = PrivateAttr(default=None)
    _files_processed: List[str] = PrivateAttr(default_factory=list)
    _processed_file_contents: Dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False
        self._chat_chain = None
        self._folder_path = None
        self._files_processed = []
        self._processed_file_contents = {}

    @classmethod
    def get_instance(cls, folder_path: str) -> 'ContentRAGTool':
        """Get or create a cached instance for the given folder path."""
        if folder_path not in _content_rag_tool_cache:
            instance = cls()
            instance._folder_path = folder_path
            _content_rag_tool_cache[folder_path] = instance
        
        return _content_rag_tool_cache[folder_path]

    def extract_payors_from_filename(self, filename: str) -> str:
        """Extract payor names from filename if present."""
        patterns = [
            r'PA_DECISION\s+\d+\s+([A-Z\s]+)\s+HORIZON',  # Matches PA_DECISION 14003999 AMBETTER HORIZON
            r'([A-Z][A-Za-z\s]+)(?:\s*\(\d+\))?\.md$',      # Matches COTIVITI HORIZON HF (1).md
            r'([A-Z][A-Za-z\s]+)\.md$'                       # Matches INDEPENDENT HEALTH HORIZON HF.md
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1).strip()
        
        return None

    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        """Process a PDF file using LlamaParse and return Document objects."""
        try:
            logger.info(f"Processing PDF file: {pdf_path}")
            # First try LlamaParse
            try:
                result = LlamaParse(result_type="markdown").load_data(pdf_path)
                
                if isinstance(result, list):
                    extracted_text = "\n".join([doc.get_content() for doc in result])
                elif isinstance(result, str):
                    extracted_text = result
                else:
                    extracted_text = ""
            except Exception as e:
                logger.warning(f"LlamaParse failed: {e}, trying fallback method for {pdf_path}")
                extracted_text = ""

            # If LlamaParse failed to extract text, try a different method
            if not extracted_text.strip():
                logger.warning(f"No text extracted from {pdf_path}, trying fallback")
                try:
                    # Alternative PDF extraction could go here if needed
                    pass
                except Exception as fallback_error:
                    logger.error(f"Fallback extraction also failed: {fallback_error}")
                    return []

            if not extracted_text.strip():
                logger.warning(f"Could not extract text from {pdf_path}")
                return []

            # Create document with file info
            file_name = os.path.basename(pdf_path)
            # Extract payor info from filename if possible
            payor_name = self.extract_payors_from_filename(file_name)
            
            metadata = {
                "file_name": file_name, 
                "file_path": pdf_path, 
                "file_type": "pdf",
                "creation_time": time.ctime(os.path.getctime(pdf_path)),
                "modified_time": time.ctime(os.path.getmtime(pdf_path)),
                "payor_name": payor_name if payor_name else "Not extracted"
            }
            
            # Store contents for potential direct access
            self._processed_file_contents[file_name] = extracted_text
            self._files_processed.append(file_name)
            
            return [Document(page_content=extracted_text, metadata=metadata)]
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []

    def process_markdown_file(self, md_path: str) -> List[Document]:
        """Process a Markdown file and return Document objects."""
        try:
            logger.info(f"Processing Markdown file: {md_path}")
            with open(md_path, 'r', encoding='utf-8') as file:
                extracted_text = file.read()

            if not extracted_text.strip():
                logger.warning(f"Empty Markdown file: {md_path}")
                return []

            file_name = os.path.basename(md_path)
            # Extract payor info from filename if possible
            payor_name = self.extract_payors_from_filename(file_name)
            
            metadata = {
                "file_name": file_name, 
                "file_path": md_path, 
                "file_type": "markdown",
                "creation_time": time.ctime(os.path.getctime(md_path)),
                "modified_time": time.ctime(os.path.getmtime(md_path)),
                "payor_name": payor_name if payor_name else "Not extracted"
            }
            
            # Store contents for potential direct access
            self._processed_file_contents[file_name] = extracted_text
            self._files_processed.append(file_name)
            
            return [Document(page_content=extracted_text, metadata=metadata)]
        except Exception as e:
            logger.error(f"Error processing Markdown {md_path}: {e}")
            return []

    def initialize_chain(self, folder_path: str):
        """Initialize the RAG chain with content from the folder."""
        if self._initialized and self._folder_path == folder_path:
            logger.info(f"Using cached chain for folder: {folder_path}")
            return
            
        logger.info(f"Initializing chain for folder: {folder_path}")
        self._folder_path = folder_path
        self._files_processed = []
        self._processed_file_contents = {}
        
        all_docs = []
        pdf_count = 0
        md_count = 0
        
        # Use glob to ensure we get all files
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)
        
        logger.info(f"Found {len(pdf_files)} PDF files and {len(md_files)} Markdown files")
        
        # Process PDF files
        for file_path in pdf_files:
            docs = self.process_pdf_file(file_path)
            all_docs.extend(docs)
            pdf_count += 1 if docs else 0
            
        # Process Markdown files
        for file_path in md_files:
            docs = self.process_markdown_file(file_path)
            all_docs.extend(docs)
            md_count += 1 if docs else 0

        logger.info(f"Successfully processed {pdf_count}/{len(pdf_files)} PDF files and {md_count}/{len(md_files)} Markdown files")
        logger.info(f"Files processed: {', '.join(self._files_processed)}")
        
        if not all_docs:
            raise ValueError("No content could be extracted from the files")

        # Split documents into chunks with more overlap for better context
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(all_docs)
        logger.info(f"Created {len(docs)} document chunks")

        # Create the FAISS vector store using OpenAI embeddings
        logger.info("Creating vector embeddings...")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        logger.info("Vector store created successfully")

        # Set up the Chat model and the retrieval chain with conversation memory
        logger.info("Initializing chat model and retrieval chain...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, openai_api_key=openai_api_key)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),  # Retrieve more documents for better coverage
            memory=memory
        )
        self._initialized = True
        logger.info("Chain initialization complete")

    def get_file_content(self, file_name: str) -> str:
        """Get the content of a specific file if it was processed."""
        if file_name in self._processed_file_contents:
            return self._processed_file_contents[file_name]
        return None

    def _run(self, folder_path: str, question: str) -> str:
        """Run the tool with caching support to avoid multiple initializations."""
        logger.info(f"ContentRAGTool: Processing folder at '{folder_path}'")
        
        # Validate and improve the question
        if not question or len(question.strip()) < 5:
            return "Please provide a more specific question to get accurate results from the documents."
        
        # Check if the question specifically asks for a file by name that we can directly access
        file_match = re.search(r'(PA_DECISION \d+ [A-Z\s]+|[A-Z][A-Za-z\s]+(?:\s*\(\d+\))?)\.md', question)
        if file_match:
            file_partial_name = file_match.group(0)
            # Find any matching file in our processed files
            matches = [f for f in self._processed_file_contents.keys() if file_partial_name in f]
            if matches:
                logger.info(f"Found direct file request for {matches[0]}")
                return f"Content of {matches[0]}:\n\n{self._processed_file_contents[matches[0]]}"
            
        # Get the cached instance for this folder path
        instance = self.get_instance(folder_path)
        
        # First, check if the folder exists
        if not os.path.exists(folder_path):
            error_msg = f"Error: The folder path '{folder_path}' does not exist."
            logger.error(error_msg)
            return error_msg
        
        # Check if the question is about listing files
        if any(keyword in question.lower() for keyword in ["file names", "files", "what files", "what are the files", "list files", "name of files", "names of files"]):
            return instance.list_files(folder_path)
            
        # Check if the folder has PDF or Markdown files using glob
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)
        
        if not pdf_files and not md_files:
            error_msg = f"Error: No PDF or Markdown files found in folder '{folder_path}'."
            logger.error(error_msg)
            return error_msg
            
        logger.info(f"ContentRAGTool: Found {len(pdf_files)} PDF files and {len(md_files)} Markdown files")
        
        # If this is a new instance or one with a different folder path,
        # we'll need to re-initialize it
        if not instance._initialized or instance._folder_path != folder_path:
            try:
                instance.initialize_chain(folder_path)
                logger.info("ContentRAGTool: Chain initialized successfully")
            except Exception as e:
                error_msg = f"Error initializing chain: {e}"
                logger.error(error_msg)
                return error_msg
        else:
            logger.info(f"ContentRAGTool: Using cached instance for {folder_path}")

        try:
            logger.info(f"ContentRAGTool: Processing question: '{question}'")
            
            # Add metadata about processed files to help with context
            enhanced_question = f"{question}\n\nContext: This query applies to the following files: {', '.join(instance._files_processed)}"
            
            # Add explicit instruction not to claim lack of access to files
            enhanced_question += "\n\nImportant: Do not claim you don't have access to any files. You have access to all files mentioned above."
            
            response = instance._chat_chain({"question": enhanced_question})
            logger.info("ContentRAGTool: Response generated successfully")
            
            # Check if response claims no access to files
            answer = response["answer"]
            if "do not have access" in answer.lower() or "don't have access" in answer.lower():
                # Try a more direct approach by including file contents directly in the prompt
                logger.info("Response incorrectly claims no file access, trying enhanced approach")
                
                # Get specific file names mentioned in the response
                mentioned_files = re.findall(r'(PA_DECISION \d+ [A-Z\s]+|[A-Z][A-Za-z\s]+(?:\s*\(\d+\))?)\.md', answer)
                
                file_contents = ""
                for partial_name in mentioned_files:
                    # Find any matching file in our processed files
                    matches = [f for f in instance._processed_file_contents.keys() if partial_name in f]
                    if matches:
                        file_name = matches[0]
                        content = instance._processed_file_contents[file_name]
                        file_contents += f"\n\n--- BEGIN {file_name} ---\n{content}\n--- END {file_name} ---\n"
                
                if file_contents:
                    direct_prompt = f"""
                    {question}
                    
                    Here are the contents of the files you claimed you didn't have access to:
                    {file_contents}
                    
                    Based on these file contents, please answer the original question.
                    """
                    follow_up = instance._chat_chain({"question": direct_prompt})
                    answer = follow_up["answer"]
            
            # If response seems insufficient, try to enhance it
            if "don't know" in answer.lower() or "no information" in answer.lower() or "not specified" in answer.lower():
                # Try a more direct follow-up
                logger.info("Initial answer was insufficient, trying a more direct approach")
                more_specific = f"List all payor names and PA statuses explicitly mentioned in any of these files: {', '.join(instance._files_processed)}. Even partial or implied information should be included."
                follow_up = instance._chat_chain({"question": more_specific})
                
                # Only use the follow-up if it's more informative
                if len(follow_up["answer"]) > len(answer) and "don't know" not in follow_up["answer"].lower():
                    answer = follow_up["answer"]
                    logger.info("Using enhanced follow-up answer")
            
            return answer
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            logger.error(error_msg)
            return error_msg

    def list_files(self, folder_path: str) -> str:
        """Special method to handle questions about file names in the folder."""
        try:
            # Use glob for recursive file listing
            pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
            md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)
            
            # Prepare a formatted response
            if not pdf_files and not md_files:
                return f"I don't have access to any PDF or Markdown files in the folder '{folder_path}'."
            
            response = f"I have access to the following files in '{folder_path}':\n\n"
            
            if pdf_files:
                response += "PDF Files:\n"
                for i, file_path in enumerate(pdf_files, 1):
                    file_name = os.path.basename(file_path)
                    response += f"{i}. {file_name} ({file_path})\n"
                response += f"\nTotal: {len(pdf_files)} PDF files.\n\n"
                
            if md_files:
                response += "Markdown Files:\n"
                for i, file_path in enumerate(md_files, 1):
                    file_name = os.path.basename(file_path)
                    response += f"{i}. {file_name} ({file_path})\n"
                response += f"\nTotal: {len(md_files)} Markdown files."
            
            return response
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return f"Error listing files: {e}" 