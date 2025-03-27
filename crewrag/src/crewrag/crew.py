from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tools.content_parser import ContentRAGTool  # Fix the import path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentCrew")

@CrewBase
class ContentRAGCrew():
    """Content RAG Crew: Process PDF and Markdown files and run a Retrieval-Augmented Chatbot"""

    # YAML configuration files (assumed to be in config/ folder)
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Store a persistent instance of the ContentRAGTool
    _content_tool = None
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing ContentRAGCrew")
        # Create a single instance of the tool to be shared
        if ContentRAGCrew._content_tool is None:
            ContentRAGCrew._content_tool = ContentRAGTool()
            logger.info("Created ContentRAGTool instance")

    @agent
    def rag_chat_agent(self) -> Agent:
        logger.info("Creating RAG chat agent")
        return Agent(
            config=self.agents_config['rag_chat_agent'],
            verbose=True,
            tools=[ContentRAGCrew._content_tool]  # Use shared instance
        )
        
    @agent
    def extraction_agent(self) -> Agent:
        """Agent specifically for data extraction from documents"""
        logger.info("Creating extraction agent")
        return Agent(
            role="Document Data Extractor",
            goal="Extract structured data from documents with high accuracy, ensuring all files are processed",
            backstory="""You are an expert at finding and extracting specific data points from documents.
            You understand that all files in the specified folder are accessible to you.
            You never claim you lack access to files - you have full access to all files listed.
            You thoroughly process every file to ensure no information is missed.""",
            verbose=True,
            tools=[ContentRAGCrew._content_tool]  # Use shared instance
        )

    @task
    def content_rag_chat_task(self) -> Task:
        logger.info("Creating content RAG chat task")
        return Task(
            description="""
            Process ALL PDF and Markdown documents from the specified {folder_path} once and answer the {question} directly based on their content.
            Use the Content RAG Tool only once per question to get a complete answer.
            NEVER claim you don't have access to files that exist in the folder - you have complete access to all files.
            """,
            expected_output="A focused answer to the provided question based solely on the PDF and Markdown documents. Your answer should be concise and directly address the question.",
            agent=self.rag_chat_agent(),
        )
        
    @task
    def data_extraction_task(self) -> Task:
        """Task specifically for extracting structured data from documents"""
        logger.info("Creating data extraction task")
        return Task(
            description="""
            Extract all instances of the following data points from ALL documents in {folder_path}:
            1. Payor names - any organization or entity mentioned as a payor, insurer, or insurance provider
            2. PA status - whether prior authorization is approved, denied, or pending
            3. Any associated codes (CPT, ICD) and their descriptions
            
            IMPORTANT: You have access to ALL files in the folder. Never claim you don't have access to any file.
            If file names are mentioned in your answer, you DO have access to those files.
            
            Present this information in a structured format (preferably a table) for the {question}.
            """,
            expected_output="A comprehensive table or structured list of all payor names and PA statuses found in any document, even if the information is partial or implied.",
            agent=self.extraction_agent(),
        )

    @crew
    def crew(self) -> Crew:
        logger.info("Creating ContentRAGCrew with multiple agents and tasks")
        return Crew(
            agents=[self.rag_chat_agent(), self.extraction_agent()],
            tasks=[self.content_rag_chat_task(), self.data_extraction_task()],
            process=Process.sequential,
            verbose=True,
        )
