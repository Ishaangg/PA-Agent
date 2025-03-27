from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import logging

# Import the new tools
from tools.tool_orchestrator import ToolOrchestratorTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentCrew")

@CrewBase
class ContentRAGCrew():
    """Content RAG Crew: Process PDF and Markdown files and run a Retrieval-Augmented Chatbot"""

    # Store a persistent instance of the orchestrator tool
    _orchestrator_tool = None
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing ContentRAGCrew")
        # Create a single instance of the orchestrator tool to be shared
        if ContentRAGCrew._orchestrator_tool is None:
            ContentRAGCrew._orchestrator_tool = ToolOrchestratorTool()
            logger.info("Created ToolOrchestratorTool instance")

    @agent
    def file_explorer_agent(self) -> Agent:
        """Agent for exploring and listing files"""
        logger.info("Creating file explorer agent")
        return Agent(
            role="File Explorer",
            goal="List and explore files in the specified directory",
            backstory="""You analyze directories to find relevant PDF and Markdown files.
            You provide clear summaries of available files to help users understand what content is available.""",
            verbose=True,
            tools=[ContentRAGCrew._orchestrator_tool]
        )
        
    @agent
    def content_processor_agent(self) -> Agent:
        """Agent for processing document content"""
        logger.info("Creating content processor agent")
        return Agent(
            role="Document Processor",
            goal="Process documents to extract their text content",
            backstory="""You are specialized in efficiently processing PDF and Markdown documents.
            You transform raw files into structured content that can be used for analysis.""",
            verbose=True,
            tools=[ContentRAGCrew._orchestrator_tool]
        )
        
    @agent
    def knowledge_base_agent(self) -> Agent:
        """Agent for building and querying knowledge bases"""
        logger.info("Creating knowledge base agent")
        return Agent(
            role="Knowledge Base Expert",
            goal="Create and query vector databases from document content",
            backstory="""You build powerful knowledge bases from document collections.
            You can create optimized vector stores and retrieve relevant information to answer specific questions.""",
            verbose=True,
            tools=[ContentRAGCrew._orchestrator_tool]
        )
        
    @agent
    def data_extraction_agent(self) -> Agent:
        """Agent for extracting structured data"""
        logger.info("Creating data extraction agent")
        return Agent(
            role="Data Extraction Specialist",
            goal="Extract structured data from document content",
            backstory="""You are an expert at finding and extracting specific data points from documents.
            You can identify payor information, PA statuses, CPT codes, and other structured data.""",
            verbose=True,
            tools=[ContentRAGCrew._orchestrator_tool]
        )

    @task
    def list_files_task(self) -> Task:
        """Task for listing files in a directory"""
        logger.info("Creating list files task")
        return Task(
            description="""
            List all PDF and Markdown files in the specified {folder_path}.
            Provide a clear summary of the available files.
            """,
            expected_output="A formatted list of all PDF and Markdown files found in the directory.",
            agent=self.file_explorer_agent(),
        )
        
    @task
    def process_files_task(self) -> Task:
        """Task for processing files"""
        logger.info("Creating process files task")
        return Task(
            description="""
            Process all PDF and Markdown files in the {folder_path} to extract their text content.
            This is a necessary prerequisite for creating a knowledge base or answering questions.
            """,
            expected_output="A summary of the files processed and any relevant information about them.",
            agent=self.content_processor_agent(),
        )
        
    @task
    def answer_question_task(self) -> Task:
        """Task for answering questions"""
        logger.info("Creating answer question task")
        return Task(
            description="""
            Answer the {question} based on the content of documents in {folder_path}.
            This requires creating a vector database from the documents and using RAG to provide an answer.
            """,
            expected_output="A direct, accurate answer to the question based solely on the content of the documents.",
            agent=self.knowledge_base_agent(),
        )
        
    @task
    def extract_data_task(self) -> Task:
        """Task for extracting structured data"""
        logger.info("Creating extract data task")
        return Task(
            description="""
            Extract structured data from all documents in {folder_path} related to the {question}.
            Focus on extracting payor names, PA statuses, and CPT codes in a structured format.
            """,
            expected_output="A structured list or table of extracted data points from the documents.",
            agent=self.data_extraction_agent(),
        )

    @crew
    def crew(self) -> Crew:
        """Create a crew with all agents and tasks"""
        logger.info("Creating ContentRAGCrew with discrete tool agents")
        return Crew(
            agents=[
                self.file_explorer_agent(),
                self.content_processor_agent(),
                self.knowledge_base_agent(),
                self.data_extraction_agent()
            ],
            tasks=[
                self.list_files_task(),
                self.process_files_task(),
                self.answer_question_task(),
                self.extract_data_task()
            ],
            process=Process.sequential,
            verbose=True,
        )
