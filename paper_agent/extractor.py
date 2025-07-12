import os
from dotenv import load_dotenv
from .structures import Paper, Author
import json
import re 
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Extractor:
    """
    Uses an LLM to extract structured information from a paper's text.
    Supports only Google Gemini (api_type='google') and local Ollama (api_type='local').
    """
    def __init__(self, api_type: str = "local", model: str = "llama3:8b-instruct-q4_K_M"):
        self.model = model
        if api_type == "google":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables for 'google' api_type.")
            self.client = ChatGoogleGenerativeAI(model=self.model, temperature=0.1)
            self.backend = "google"
            print(f"Extractor initialized for Google Gemini with model: {self.model}")
        elif api_type == "local":
            from openai import OpenAI
            self.client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama',
            )
            self.backend = "local"
            print(f"Extractor initialized for LOCAL Ollama with model: {self.model}")
        else:
            raise ValueError(f"Unsupported api_type: {api_type}. Choose 'google' or 'local'.")

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
        print(f"Sending request to LLM ({self.model}) for metadata extraction...")
        try:
            if isinstance(self.client, ChatGoogleGenerativeAI):
                response = self.client.invoke(
                    f"From the text of a scientific paper below, extract the title, authors, and abstract. Provide the output ONLY as a single, clean JSON object. Do not add any other text before or after the JSON. The JSON object must have these exact keys: 'title', 'authors', 'abstract'. 'authors' should be a list of strings of author names. The author names might have symbols like '*' or '†' next to them; please remove these symbols. Here is the text: --- {paper.full_text[:15000]} --- JSON_OUTPUT:"
                )
                raw_content = response.content if hasattr(response, 'content') else str(response)
            else:
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
                print(f"Raw response: {getattr(response, 'content', getattr(response, 'choices', ''))}")
            return paper

    def summarize_paper_text(self, paper_text: str, paper_title: str) -> str:
        """
        Uses an LLM to generate a concise summary of a paper's text.
        
        Args:
            paper_text: The full text of the paper to summarize.
            paper_title: The title of the paper, for context.

        Returns:
            A string containing the summary.
        """
        # Take a significant portion of the text for summarization, but not all
        # to save on tokens for very long papers.
        text_for_summary = paper_text[:25000]

        prompt = f"""
        You are a scientific summarization assistant. Your task is to provide a concise summary of the following research paper.
        Focus on these key areas:
        1.  **Problem:** What problem does the paper aim to solve?
        2.  **Methodology:** What is the core approach or method proposed?
        3.  **Key Findings:** What were the main results or conclusions?

        Do not include your own opinions or any information not present in the text.

        Paper Title: {paper_title}
        ---
        Paper Text:
        {text_for_summary}
        ---
        
        Please provide a summary based on the text.
        """
        
        print(f"Sending request to LLM for summarization of '{paper_title}'...")
        if isinstance(self.client, ChatGoogleGenerativeAI):
            response = self.client.invoke(
                f"You are a scientific summarization assistant. Your task is to provide a concise summary of the following research paper. Focus on these key areas: 1.  **Problem:** What problem does the paper aim to solve? 2.  **Methodology:** What is the core approach or method proposed? 3.  **Key Findings:** What were the main results or conclusions? Do not include your own opinions or any information not present in the text. Paper Title: {paper_title} --- Paper Text: {paper_text[:25000]} --- Please provide a summary based on the text."
            )
            summary = response.content if hasattr(response, 'content') else str(response)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            summary = response.choices[0].message.content
        if not isinstance(summary, str):
            if isinstance(summary, list):
                summary = "\n".join(str(item) for item in summary)
            else:
                summary = str(summary)
        if summary is None:
            summary = ""
        print("Summarization successful.")
        return summary

    def extract_table_from_text(self, paper_text: str, paper_title: str, topic: str) -> str:
        """Uses an LLM to find and convert a relevant table to Markdown format."""
        prompt = f"""
        From the text of the research paper titled '{paper_title}', find the most relevant table that discusses '{topic}'.
        Your task is to extract this table and convert it into a clean, well-formatted Markdown table.
        
        If no relevant table is found, respond with "No relevant table found for the specified topic."

        Here is the paper text:
        ---
        {paper_text[:30000]} 
        ---
        
        Return ONLY the Markdown table.
        """
        print(f"Sending request to LLM for table extraction on topic: '{topic}'...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        if isinstance(self.client, ChatGoogleGenerativeAI):
            response = self.client.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result = response.choices[0].message.content
        # Ensure result is always a string
        if not isinstance(result, str):
            if isinstance(result, list):
                result = "\n".join(str(item) for item in result)
            else:
                result = str(result)
        if result is None:
            result = ""
        return result

    def extract_table_as_json(self, paper_text: str, paper_title: str, topic: str) -> str:
        """Uses an LLM to find a table and convert it to a structured JSON string."""
        prompt = f"""
        From the research paper '{paper_title}', find the most relevant table discussing '{topic}'.
        Your task is to extract this table and represent it as a JSON object.
        The JSON should have two keys: "columns" (a list of column header strings) and "data" (a list of lists, where each inner list is a row of data).
        Clean the data: ensure numerical values are numbers, not strings.

        Example Output:
        {{
            "columns": ["Model", "BLEU Score", "Training Cost (FLOPs)"],
            "data": [
                ["Transformer (base)", 27.3, 3.3e18],
                ["Transformer (big)", 28.4, 1.0e19]
            ]
        }}

        Paper Text:
        ---
        {paper_text[:30000]}
        ---
        Return ONLY the valid JSON object.
        """
        print(f"Sending request to LLM for JSON table extraction on topic: '{topic}'...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        if isinstance(self.client, ChatGoogleGenerativeAI):
            response = self.client.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content
        if not isinstance(result, str):
            if isinstance(result, list):
                result = "\n".join(str(item) for item in result)
            else:
                result = str(result)
        if result is None:
            result = ""
        return result

    def extract_citations(self, text: str) -> list[str]:
        # TODO: Implement with LLM or regex
        return []

    def extract_keywords(self, paper_text: str, num_keywords: int) -> list[str]:
        """Uses an LLM to pull out the most important keywords from a text."""
        prompt = f"""
        You are an expert at scientific keyword extraction.
        From the following text, identify the {num_keywords} most important and specific keywords, concepts, and technical terms.
        Do not include generic terms like 'research', 'paper', or 'model'. Focus on specific, technical concepts.
        
        Return your answer as a single JSON object with one key, \"keywords\", which is a list of strings.

        Text:
        ---
        {paper_text[:20000]}
        ---
        
        JSON output:
        """
        print(f"Sending request to LLM for keyword extraction...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        if isinstance(self.client, ChatGoogleGenerativeAI):
            response = self.client.invoke(prompt)
            raw_content = response.content if hasattr(response, 'content') else str(response)
            # Ensure raw_content is a string
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)
            # Try to extract JSON from the response
            try:
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if not json_match:
                    return ["Error: No JSON object found in LLM response."]
                json_str = json_match.group(0)
                json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
                data = json.loads(json_str)
                return data.get("keywords", [])
            except Exception:
                return ["Error parsing keywords from LLM response."]
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            try:
                content = response.choices[0].message.content
                if content is None:
                    return ["Error: No content in OpenAI response."]
                data = json.loads(content)
                return data.get("keywords", [])
            except (json.JSONDecodeError, AttributeError):
                return ["Error parsing keywords from LLM response."]