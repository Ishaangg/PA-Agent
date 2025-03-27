"""
Data Extraction Tool - Only extracts structured data from document content
"""
from crewai.tools import BaseTool
from typing import Type, Dict, List, Any, Optional
from pydantic import BaseModel, Field
import os
import logging
import json
import re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataExtractor")

class DataExtractionInput(BaseModel):
    content: str = Field(..., description="The text content to extract data from.")
    extraction_type: str = Field(..., description="Type of data to extract: 'payor', 'pa_status', 'cpt_codes', or 'all'")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model to use for extraction.")

class DataExtractionTool(BaseTool):
    name: str = "Data Extraction Tool"
    description: str = "Extracts structured data from document content. Only extracts data, doesn't process files or answer questions."
    args_schema: Type[BaseModel] = DataExtractionInput
    
    def _run(self, content: str, extraction_type: str = "all", model_name: str = "gpt-4o-mini") -> str:
        """Extract structured data from the provided content."""
        logger.info(f"Extracting {extraction_type} data from content")
        
        if not content.strip():
            return "No content provided for data extraction."
            
        if not openai_api_key:
            return "OpenAI API key not found in environment variables."
            
        # Create extraction prompts based on extraction type
        prompts = {
            "payor": "Extract all payor names (insurance companies, health plans) mentioned in this text. Return as a JSON list.",
            "pa_status": "Extract all prior authorization (PA) statuses mentioned in this text. Include the status (approved, denied, pending) and any associated context. Return as a JSON object with status as keys.",
            "cpt_codes": "Extract all CPT/procedure codes mentioned in this text. Include the code, any description, and associated status. Return as a JSON object with codes as keys.",
            "all": "Extract the following structured data from this text:\n1. Payor names (insurance companies)\n2. PA statuses (approved, denied, pending)\n3. CPT/procedure codes with descriptions\n\nReturn as a structured JSON object with these three categories as keys."
        }
        
        if extraction_type not in prompts:
            return f"Invalid extraction type: {extraction_type}. Valid types are: {', '.join(prompts.keys())}"
            
        try:
            # Use OpenAI to extract the data
            llm = ChatOpenAI(
                model=model_name, 
                temperature=0, 
                openai_api_key=openai_api_key
            )
            
            # Build the extraction prompt
            extraction_prompt = f"""
            {prompts[extraction_type]}
            
            Text to extract from:
            ---
            {content}
            ---
            
            Format your response ONLY as valid JSON.
            """
            
            # Get the extraction result
            logger.info(f"Sending extraction prompt to model {model_name}")
            result = llm.invoke(extraction_prompt)
            
            # Extract JSON from the response
            response_text = result.content
            
            # Try to parse as JSON (clean up if needed)
            try:
                # Find JSON in the response (in case the model added extra text)
                json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                else:
                    data = json.loads(response_text)
                    
                # Format the result
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                return response_text
                
        except Exception as e:
            error_msg = f"Error extracting data: {e}"
            logger.error(error_msg)
            return error_msg 