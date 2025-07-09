import os
from openai import OpenAI
from dotenv import load_dotenv
from .structures import Paper, Author
import json
import re 


load_dotenv()

class Extractor:
    """
    Uses an LLM to extract structured information from a paper's text.
    Can be configured to use OpenAI's API or a local Ollama instance.
    """
    def __init__(self, api_type: str = "local", model: str = "phi3:instruct"):
        self.model = model
        if api_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables for 'openai' api_type.")
            self.client = OpenAI(api_key=api_key)
            print(f"Extractor initialized for OpenAI with model: {self.model}")
        elif api_type == "local":
            self.client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama',
            )
            print(f"Extractor initialized for LOCAL Ollama with model: {self.model}")
        else:
            raise ValueError(f"Unsupported api_type: {api_type}. Choose 'openai' or 'local'.")

    def _clean_llm_response(self, response_text: str) -> str:
        """
        Cleans the raw text response from an LLM to extract a valid JSON string.
        """
        # Find the first '{' and the last '}' to extract the JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            print("Warning: Could not find a JSON object in the response.")
            return ""
        json_str = json_match.group(0)
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        return json_str

    def extract_metadata(self, paper: Paper) -> Paper:
        """
        Extracts title, authors, and abstract from the paper's raw text.
        """
        text_for_extraction = paper.full_text[:15000]
        prompt = f"""
        From the text of a scientific paper below, extract the title, authors, and abstract.
        Provide the output ONLY as a single, clean JSON object. Do not add any other text before or after the JSON.
        The JSON object must have these exact keys: "title", "authors", "abstract".
        "authors" should be a list of strings of author names. The author names might have symbols like '*' or '†' next to them; please remove these symbols.

        Here is the text:
        ---
        {text_for_extraction}
        ---
        JSON_OUTPUT:
        """
        print(f"Sending request to LOCAL LLM ({self.model}) for metadata extraction...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a JSON-emitting assistant specializing in scientific papers. You only output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            raw_content = response.choices[0].message.content
            # Ensure raw_content is a string
            if not isinstance(raw_content, str):
                raw_content = ""
            # Use our new cleaning function
            cleaned_json_str = self._clean_llm_response(raw_content)
            if not cleaned_json_str:
                raise ValueError("Cleaned response is empty.")
            extracted_data = json.loads(cleaned_json_str)
            paper.title = extracted_data.get("title")
            paper.abstract = extracted_data.get("abstract")
            author_names = extracted_data.get("authors", [])
            # Also clean the individual author names to remove symbols
            if author_names:
                cleaned_authors = [re.sub(r'[\*†]', '', name).strip() for name in author_names]
                paper.authors = [Author(name=name) for name in cleaned_authors]
            print("Metadata extraction successful.")
            return paper
        except Exception as e:
            # Catch a broader range of exceptions during development
            print(f"Error processing or parsing LLM response: {e}")
            if 'response' in locals():
                print(f"Raw response: {response.choices[0].message.content}")
            return paper