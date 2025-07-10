
from langchain_neo4j import Neo4jGraph 
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from typing import Type, Any
from pydantic import BaseModel, Field

from .knowledge_base import KnowledgeBase
from .ingestor import Ingestor
from .extractor import Extractor

# --- Tool for Searching the Web ---
# This one is easy, we can use a pre-built tool from LangChain
web_search_tool = DuckDuckGoSearchRun()


# --- Tool for Answering Questions from the Knowledge Base ---
class KBQueryInput(BaseModel):
    """Input model for the Knowledge Base Query Tool."""
    query: str = Field(description="A detailed, specific question to ask the knowledge base.")

class KnowledgeBaseQueryTool(BaseTool):
    """A tool to answer questions about scientific papers."""
    name: str = "scientific_paper_knowledge_base_tool"  # <-- More specific name
    description: str = (
        "**This is the primary tool.** Use this tool FIRST for any question about "
        "scientific topics, research papers, methodologies, or experimental results. "
        "It uses a specialized database of academic papers."
    )
    args_schema: Type[BaseModel] = KBQueryInput
    
    # Accept any object with a run_query method
    rag_agent: Any

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.rag_agent.run_query(user_query=query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        # For now, we don't have an async implementation, so we raise an error
        # or just call the sync version.
        raise NotImplementedError("This tool does not support async yet.")


# --- Tool for Loading New Papers from ArXiv ---
class ArxivSearchInput(BaseModel):
    """Input model for the ArXiv Search Tool."""
    query: str = Field(description="The search query to send to the arXiv API.")
    max_results: int = Field(default=3, description="The maximum number of papers to fetch.")

class ArxivSearchTool(BaseTool):
    """A tool to find and add new papers from arXiv."""
    name: str = "arxiv_paper_search_and_load_tool"
    description: str = (
        "Use this tool ONLY when the user explicitly asks to find, search for, or load NEW papers "
        "on a specific topic. This adds papers to the knowledge base for later questions."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    # This tool needs access to both the Ingestor and the KnowledgeBase
    ingestor: Ingestor
    kb: KnowledgeBase

    def _run(self, query: str, max_results: int = 3) -> str:
        """Use the tool."""
        papers = self.ingestor.load_from_arxiv(query=query, max_results=max_results)
        if not papers:
            return f"No papers were found on arXiv for the query: '{query}'"
        
        self.kb.add_papers(papers)
        paper_titles = "\n- ".join([str(p.title) for p in papers if p.title])
        return f"Successfully loaded {len(papers)} papers into the knowledge base:\n- {paper_titles}"

    async def _arun(self, query: str, max_results: int) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async yet.")


# --- NEW Tool for Summarizing a Specific Paper ---
class SummarizationInput(BaseModel):
    """Input model for the Paper Summarization Tool."""
    paper_id: str = Field(description="The unique ID of the paper to summarize, e.g., its file path or arXiv ID.")

class PaperSummarizationTool(BaseTool):
    """A tool to generate a concise summary of a specific scientific paper already in the knowledge base."""
    name: str = "paper_summarization_tool"
    description: str = (
        "Use this tool when the user asks for a summary of a specific paper. "
        "You must provide the 'paper_id' of the paper you want to summarize. "
        "To get the paper_id, you might need to use another tool first to find the paper."
    )
    args_schema: Type[BaseModel] = SummarizationInput

    # This tool needs access to both the KB (to get the paper) and the Extractor (to summarize it)
    kb: KnowledgeBase
    extractor: Extractor

    def _run(self, paper_id: str) -> str:
        """Use the tool."""
        # The ChromaDB 'get' method can fetch items by their ID.
        # We need to find all chunks for a given paper_id.
        results = self.kb.collection.get(where={"paper_id": paper_id})
        
        if not results or not results.get('documents'):
            return f"Error: Could not find a paper with ID '{paper_id}' in the knowledge base."

        # Reconstruct the full text and get metadata
        documents = results.get('documents')
        if not documents:
            return f"Error: Could not find a paper with ID '{paper_id}' in the knowledge base."

        # Reconstruct the full text and get metadata
        full_text = " ".join([str(doc) for doc in documents])
        # Defensive: ensure metadatas exists and has at least one item
        metadatas = results.get('metadatas')
        if metadatas and len(metadatas) > 0:
            raw_title = metadatas[0].get('title', 'Unknown Title')
            title = str(raw_title) if raw_title is not None else 'Unknown Title'
        else:
            title = 'Unknown Title'

        # Use our new extractor method
        summary = self.extractor.summarize_paper_text(full_text, title)
        return summary

    async def _arun(self, paper_id: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")

# --- NEW Tool for Fetching a Specific Paper by ArXiv ID ---
class ArxivFetchInput(BaseModel):
    """Input model for the ArXiv Fetch Tool."""
    paper_arxiv_id: str = Field(description="The unique arXiv ID of the paper, e.g., '1706.03762'.")

class ArxivFetchTool(BaseTool):
    """
    A tool to fetch a single, specific paper from arXiv using its ID.
    Use this when you know the exact paper you need.
    """
    name: str = "arxiv_fetch_by_id_tool"
    description: str = (
        "Use this tool to fetch a specific paper from arXiv if you have its ID (e.g., '1706.03762'). "
        "This is more precise than searching by title."
    )
    args_schema: Type[ArxivFetchInput] = ArxivFetchInput

    ingestor: Ingestor
    kb: KnowledgeBase

    def _run(self, paper_arxiv_id: str) -> str:
        # The arxiv library allows fetching by ID directly
        try:
            import arxiv
            result = next(arxiv.Search(id_list=[paper_arxiv_id]).results())
            papers = self.ingestor.load_from_arxiv(query=f"id:{paper_arxiv_id}", max_results=1)
            if not papers:
                return f"Could not load paper with ID '{paper_arxiv_id}'."
            self.kb.add_papers(papers)
            return f"Successfully fetched and loaded paper '{papers[0].title}'."
        except StopIteration:
            return f"Error: No paper found with arXiv ID '{paper_arxiv_id}'."
        except Exception as e:
            return f"An error occurred: {e}"

    async def _arun(self, paper_arxiv_id: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")

# --- NEW Tool for Finding Papers in the KB ---
class PaperFinderInput(BaseModel):
    """Input model for the Paper Finder Tool."""
    query: str = Field(description="A query to find relevant papers, usually the title of the paper.")

class PaperFinderTool(BaseTool):
    """Finds papers in the knowledge base. Use this to get the 'paper_id' for other tools."""
    name: str = "paper_finder_tool"
    description: str = (
        "Use this tool to find a paper's 'paper_id'. This is a preliminary step for other tools "
        "like the summarization tool. It returns raw metadata."
    )
    args_schema: Type[PaperFinderInput] = PaperFinderInput
    
    kb: KnowledgeBase

    def _run(self, query: str) -> str:
        """Use the tool."""
        import json
        search_results = self.kb.search(query=query, n_results=3)
        if not search_results['ids'][0]:
            return json.dumps({"papers": []})
        # Return the list of metadatas as a JSON string for easy parsing by the agent
        return json.dumps({"papers": search_results['metadatas'][0]})

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")


# --- NEW Tool for Answering Questions using RAG ---
class QuestionAnsweringInput(BaseModel):
    """Input model for the Question Answering Tool."""
    question: str = Field(description="A detailed, specific question to ask about the content of the papers.")

class QuestionAnsweringTool(BaseTool):
    """A tool to answer questions using the knowledge base. This performs a search and synthesizes an answer."""
    name: str = "question_answering_tool"
    description: str = (
        "Use this tool to answer a user's question about a topic. "
        "This is a powerful tool that uses the full RAG pipeline."
    )
    args_schema: Type[QuestionAnsweringInput] = QuestionAnsweringInput
    
    rag_agent: Any # This will be our TempRAGAgent

    def _run(self, question: str) -> str:
        """Use the tool."""
        return self.rag_agent.run_query(user_query=question)

    async def _arun(self, question: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")

# --- NEW Tool for Querying the Graph Database ---
class GraphQueryInput(BaseModel):
    """Input model for the Graph Query Tool."""
    query: str = Field(description="A Cypher query to run against the Neo4j graph database.")

class GraphQueryTool(BaseTool):
    """Answers questions about relationships between papers and authors by querying the knowledge graph."""
    name: str = "graph_query_tool"
    description: str = (
        "**This is the best tool for answering questions about authors, collaborations, or connections between papers.** "
        "Use this tool when the user asks 'Who wrote X?', 'Who works with Y?', or 'What papers cite Z?'. "
        "Input must be a valid Cypher query."
    )
    args_schema: Type[GraphQueryInput] = GraphQueryInput

    graph: Neo4jGraph

    def _run(self, query: str) -> str:
        """Use the tool."""
        print(f"Running Cypher query: {query}")
        try:
            result = self.graph.query(query)
            

            # Instead of returning the raw list, format it into a sentence.
            if not result:
                return "No results found in the graph for that query."
            
            # Assuming the query returns a list of dictionaries, where each dict
            # has one key (e.g., 'author' or 'title').
            # We extract all the values from the list of dicts.
            values = [list(record.values())[0] for record in result if record]
            
            if not values:
                 return "The query ran, but returned no data."

            return f"The following items were found in the knowledge graph: {', '.join(map(str, values))}"

        except Exception as e:
            return f"Error executing Cypher query: {e}"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # We dynamically add the schema to the description, escaping the curly braces
        # so they are not treated as prompt variables.
        schema = self.graph.get_schema
        if callable(schema):
            schema = schema()
        escaped_schema = schema.replace("{", "{{").replace("}", "}}")
        self.description += f"\nGraph Schema:\n{escaped_schema}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")