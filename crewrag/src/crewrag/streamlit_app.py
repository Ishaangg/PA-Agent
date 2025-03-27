import streamlit as st
import os
import sys
import time
import logging
import glob
from streamlit.logger import get_logger
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_app.log"),
        logging.StreamHandler()
    ]
)
logger = get_logger(__name__)

# Add the project root to the Python path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Load environment variables
load_dotenv()

# Try to import the ContentRAGCrew
try:
    from crew import ContentRAGCrew
    from tools.content_parser import ContentRAGTool
    logger.info("Successfully imported ContentRAGCrew")
except ImportError:
    try:
        sys.path.append(os.path.join(current_dir, "src"))
        from crewrag.crew import ContentRAGCrew
        from crewrag.tools.content_parser import ContentRAGTool
        logger.info("Successfully imported using fallback method")
    except ImportError as e:
        logger.error(f"Failed to import ContentRAGCrew: {e}")
        st.error(f"Failed to import ContentRAGCrew: {e}")

# Set page config
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
.stButton>button {
    width: 100%;
}
.title-container {
    background-color: #4e8cff;
    padding: 1.5rem;
    border-radius: 0.5rem;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}
.file-display {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
}
.chat-container {
    border: 1px solid #e0e0e0;
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: white;
    height: 400px;
    overflow-y: auto;
}
.user-message {
    background-color: #e1f5fe;
    padding: 0.75rem;
    border-radius: 0.5rem;
    margin-bottom: 0.75rem;
    max-width: 80%;
    align-self: flex-end;
    margin-left: auto;
}
.assistant-message {
    background-color: #f0f0f0;
    padding: 0.75rem;
    border-radius: 0.5rem;
    margin-bottom: 0.75rem;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'crew_instance' not in st.session_state:
        st.session_state.crew_instance = None
    if 'folder_path' not in st.session_state:
        # Default folder path
        default_path = r'C:\Users\OMEN\Desktop\PA policy'
        if os.path.exists(default_path):
            st.session_state.folder_path = default_path
        else:
            st.session_state.folder_path = os.path.join(current_dir, "knowledge")
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = []

def get_folder_files(folder_path):
    """Get all PDF and Markdown files in the folder"""
    pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
    md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)
    return pdf_files, md_files

def format_file_path(file_path):
    """Format file path for display"""
    return os.path.basename(file_path)

def process_question(question, folder_path):
    """Process a question using the RAG system"""
    st.session_state.processing = True
    
    try:
        # Initialize crew if not already done
        if st.session_state.crew_instance is None:
            with st.spinner("Initializing document analysis system..."):
                st.session_state.crew_instance = ContentRAGCrew().crew()
                logger.info("Created ContentRAGCrew instance")
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Process the question
        with st.spinner("Processing your question..."):
            inputs = {
                "folder_path": folder_path,
                "question": question
            }
            
            results = st.session_state.crew_instance.kickoff(inputs=inputs)
            
            # Format and add response to chat
            if isinstance(results, dict):
                formatted_result = ""
                for task_id, result in results.items():
                    if 'extraction' in task_id.lower():
                        formatted_result += "📊 EXTRACTED DATA:\n\n"
                    else:
                        formatted_result += "🔍 ANALYSIS:\n\n"
                    formatted_result += f"{result}\n\n"
                st.session_state.messages.append({"role": "assistant", "content": formatted_result})
            else:
                st.session_state.messages.append({"role": "assistant", "content": results})
            
            # Get list of processed files from the tool instance
            try:
                content_tool = ContentRAGTool.get_instance(folder_path)
                st.session_state.files_processed = content_tool._files_processed
            except Exception as e:
                logger.error(f"Error getting processed files: {e}")
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
    
    finally:
        st.session_state.processing = False

def main():
    initialize_session_state()
    
    # App header
    st.markdown('<div class="title-container"><h1>📑 Document RAG Assistant</h1><p>PDF & Markdown Document Analysis with AI</p></div>', unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Document Settings")
        folder_path = st.text_input("Document Folder Path", st.session_state.folder_path)
        
        if folder_path != st.session_state.folder_path:
            st.session_state.folder_path = folder_path
            st.session_state.crew_instance = None  # Reset crew when folder changes
            st.session_state.files_processed = []
        
        if st.button("📂 Update Folder"):
            if os.path.exists(folder_path):
                st.session_state.folder_path = folder_path
                st.session_state.crew_instance = None  # Reset crew
                st.session_state.files_processed = []
                st.success(f"Folder updated to: {folder_path}")
            else:
                st.error(f"Folder does not exist: {folder_path}")
        
        # Display files in the folder
        pdf_files, md_files = get_folder_files(folder_path)
        
        st.subheader(f"📄 Available Files ({len(pdf_files) + len(md_files)})")
        
        with st.expander("PDF Files", expanded=len(pdf_files) < 10):
            for file in pdf_files:
                file_name = format_file_path(file)
                if file_name in st.session_state.files_processed:
                    st.markdown(f"✅ **{file_name}**")
                else:
                    st.markdown(f"- {file_name}")
        
        with st.expander("Markdown Files", expanded=len(md_files) < 10):
            for file in md_files:
                file_name = format_file_path(file)
                if file_name in st.session_state.files_processed:
                    st.markdown(f"✅ **{file_name}**")
                else:
                    st.markdown(f"- {file_name}")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What files are available?",
            "What are all the payor names across all files?",
            "Extract all payor names and PA statuses from all documents",
            "Summarize the content of all files",
            "What CPT codes are mentioned in the documents?",
        ]
        
        for question in sample_questions:
            if st.button(question):
                process_question(question, folder_path)
    
    with col2:
        st.subheader("💬 Chat with your Documents")
        
        # Chat interface
        chat_placeholder = st.container()
        
        # Display chat messages
        with chat_placeholder:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 <b>You:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message">🤖 <b>Assistant:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
        
        # Input for new question
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_area("Your question:", placeholder="Ask something about your documents...", height=100)
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_question:
                process_question(user_question, folder_path)
                st.experimental_rerun()
        
        # Processing indicator
        if st.session_state.processing:
            st.info("Processing your question... Please wait.")
        
        # Reset chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

if __name__ == "__main__":
    main() 