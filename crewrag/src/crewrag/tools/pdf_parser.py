from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
import os
from dotenv import load_dotenv

# LlamaParse and LangChain imports
from llama_parse import LlamaParse
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")


class PDFRAGToolInput(BaseModel):
    folder_path: str = Field(..., description="Path to the folder containing PDF files.")
    question: str = Field(..., description="Question to ask based on the PDFs.")


class PDFRAGTool(BaseTool):
    name: str = "PDF RAG Tool"
    description: str = (
        "Processes PDFs from a folder, builds a retrieval chain, and answers questions based on the content."
    )
    args_schema: Type[BaseModel] = PDFRAGToolInput

    # Define internal state as private attributes
    _initialized: bool = PrivateAttr(default=False)
    _chat_chain: Optional[ConversationalRetrievalChain] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False
        self._chat_chain = None

    def initialize_chain(self, folder_path: str):
        all_docs = []
        # Iterate over all PDF files in the provided folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(folder_path, filename)
                try:
                    result = LlamaParse(result_type="markdown").load_data(pdf_path)
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
                    continue

                if isinstance(result, list):
                    extracted_text = "\n".join([doc.get_content() for doc in result])
                elif isinstance(result, str):
                    extracted_text = result
                else:
                    continue

                if not extracted_text.strip():
                    continue

                file_name = os.path.basename(pdf_path)
                metadata = {"file_name": file_name, "file_path": pdf_path}
                doc = Document(page_content=extracted_text, metadata=metadata)
                all_docs.append(doc)


        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(all_docs)

        # Create the FAISS vector store using OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        print("Vector store created successfully")

        # Set up the Chat model and the retrieval chain with conversation memory
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, openai_api_key=openai_api_key)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory
        )
        self._initialized = True

    def _run(self, folder_path: str, question: str) -> str:
        print(f"PDFRAGTool: Processing folder at '{folder_path}'")
        
        # First, check if the folder exists
        if not os.path.exists(folder_path):
            error_msg = f"Error: The folder path '{folder_path}' does not exist."
            print(error_msg)
            return error_msg
        
        # Check if the question is about listing files
        if any(keyword in question.lower() for keyword in ["file names", "files", "what files", "what are the files", "list files", "name of files", "names of files"]):
            return self.list_files(folder_path)
            
        # Check if the folder is empty
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        if not files:
            error_msg = f"Error: No PDF files found in folder '{folder_path}'."
            print(error_msg)
            return error_msg
            
        print(f"PDFRAGTool: Found {len(files)} PDF files: {', '.join(files)}")
        
        if not self._initialized:
            try:
                self.initialize_chain(folder_path)
                print("PDFRAGTool: Chain initialized successfully")
            except Exception as e:
                error_msg = f"Error initializing chain: {e}"
                print(error_msg)
                return error_msg

        try:
            print(f"PDFRAGTool: Processing question: '{question}'")
            response = self._chat_chain({"question": question})
            print("PDFRAGTool: Response generated successfully")
            return response["answer"]
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            print(error_msg)
            return error_msg

    def list_files(self, folder_path: str) -> str:
        """Special method to handle questions about file names in the folder."""
        try:
            all_files = os.listdir(folder_path)
            
            # Filter for PDF files
            pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
            
            # Prepare a formatted response
            if not pdf_files:
                return f"I don't have access to any PDF files in the folder '{folder_path}'. The folder exists but contains no PDF files."
            
            response = f"I have access to the following files in '{folder_path}':\n\n"
            for i, file in enumerate(pdf_files, 1):
                response += f"{i}. {file}\n"
                
            response += f"\nTotal: {len(pdf_files)} PDF files."
            
            # Add information about non-PDF files if they exist
            non_pdf_files = [f for f in all_files if not f.lower().endswith('.pdf')]
            if non_pdf_files:
                response += f"\n\nAdditionally, there are {len(non_pdf_files)} non-PDF files in the folder, but I'm not configured to process them."
            
            return response
        except Exception as e:
            return f"Error listing files: {e}"


