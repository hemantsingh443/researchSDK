
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from typing import Type, Any
from pydantic import BaseModel, Field

from .knowledge_base import KnowledgeBase
from .ingestor import Ingestor

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