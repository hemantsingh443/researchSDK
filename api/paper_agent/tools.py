"""Tools for the Paper Agent system.

This module contains various tools that the agent can use to interact with papers,
extract information, and perform analyses. The tools are designed to be used with
the MasterAgent class.
"""

import os
import json
import re
import shutil
import tempfile
import subprocess
import logging
from typing import Type, Any, List, Dict, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field
from langchain_community.tools import BaseTool, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_neo4j import Neo4jGraph
from langchain_core.language_models.chat_models import BaseChatModel

from .knowledge_base import KnowledgeBase
from .ingestor import Ingestor
from .extractor import Extractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Standard Response Models ---

class StandardResponse(BaseModel):
    """Base response model for all tools."""
    status: str = Field(description="Indicates 'success' or 'failure'.")
    message: str = Field(description="A human-readable message about the result.")
    data: Optional[Any] = Field(default=None, description="The result data if the operation was successful.")

class SuccessResponse(StandardResponse):
    """Response model for successful operations."""
    status: str = "success"

class FailureResponse(StandardResponse):
    """Response model for failed operations."""
    status: str = "failure"

# --- Tool Factory Function ---

def get_tools(kb: KnowledgeBase, extractor: Extractor, ingestor: Ingestor, llm: BaseChatModel) -> List[BaseTool]:
    """Factory function to create and return all available tools with proper dependencies."""
    # Search and retrieval tools
    search_tools = [
        DuckDuckGoSearchRun(),
        ArxivSearchTool(ingestor=ingestor, kb=kb),
        GetPaperMetadataByTitleTool(kb=kb, llm=llm)
    ]
    
    # Content analysis tools (non-graph based)
    analysis_tools = [
        AnswerFromPapersTool(kb=kb, llm=llm),
        PaperSummarizationTool(kb=kb, extractor=extractor, llm=llm),
        KeywordExtractionTool(kb=kb, extractor=extractor, llm=llm),
        LiteratureGapTool(kb=kb, extractor=extractor, llm=llm),
    ]
    
    # Data extraction and processing tools
    extraction_tools = [
        TableExtractionTool(kb=kb, extractor=extractor, llm=llm),
        DataToCsvTool()
    ]
    
    # Visualization tools
    visualization_tools = [
        DynamicVisualizationTool(code_writing_llm=llm),
        PlotGenerationTool(llm=llm)
    ]
    
    # Initialize graph tools if Neo4j is available
    graph_tools = []
    try:
        from neo4j import GraphDatabase
        from langchain_neo4j import Neo4jGraph
        from paper_agent.config import settings
        
        print("\n=== Initializing Graph Tools ===")
        print(f"Neo4j URI: {settings.NEO4J_URI}")
        
        # Initialize Neo4j graph connection
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE or "neo4j"
        )
        
        # Test the connection
        test_query = "RETURN 1 as test_value"
        result = graph.query(test_query)
        print(f"Neo4j test query result: {result}")
        
        # Initialize graph tools
        graph_tools = [
            GraphQueryTool(graph=graph),
            RelationshipAnalysisTool(graph=graph, llm=llm),
            CitationAnalysisTool(graph=graph, paper_id="default_paper_id")
        ]
        
        print("Successfully initialized graph tools:")
        for tool in graph_tools:
            print(f"- {tool.name}: {tool.__class__.__name__}")
            
    except Exception as e:
        import traceback
        print(f"\n❌ Error initializing graph tools:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        print("\nGraph functionality will be disabled. Make sure Neo4j is running and properly configured.")
    
    # Combine all tools
    all_tools = search_tools + analysis_tools + extraction_tools + visualization_tools + graph_tools
    print(f"\nInitialized {len(all_tools)} tools: {[tool.name for tool in all_tools]}")
    
    return all_tools

# --- Tool Definitions ---

class AnswerFromPapersInput(BaseModel):
    """Input model for answering questions from papers."""
    question: str = Field(description="A detailed, specific question to ask about the content of the papers.")

class ArxivSearchInput(BaseModel):
    """Input model for the ArXiv Search Tool."""
    query: str = Field(description="The search query to find papers on arXiv (e.g., 'machine learning').")
    max_results: int = Field(default=3, description="Maximum number of results to return (default: 3).")

class AnswerFromPapersTool(BaseTool):
    """Tool to answer questions using the knowledge base of papers."""
    name: str = "answer_from_papers"
    description: str = (
        "**Use this tool FIRST** for any question about scientific topics, papers, or results. "
        "It performs a semantic search over the existing knowledge base of academic papers to find the answer."
    )
    args_schema: Type[BaseModel] = AnswerFromPapersInput
    
    kb: KnowledgeBase
    llm: BaseChatModel

    def _run(self, question: str) -> Dict[str, Any]:
        """Use the tool to answer a question using the knowledge base."""
        try:
            # Validate inputs
            if not question or not isinstance(question, str):
                return FailureResponse(
                    message="Invalid question: Please provide a non-empty question.",
                    data={"sources": []}
                ).model_dump()
            
            print(f"Answering question: '{question}' using knowledge base...")
            
            # Perform a semantic search
            search_results = self.kb.search(query=question, n_results=5)
            
            if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
                return FailureResponse(
                    message="I could not find any relevant information in the knowledge base to answer that question. Please try rephrasing your question or add more papers to the knowledge base.",
                    data={"sources": []}
                ).model_dump()
            
            # Format the context for the LLM
            context_str = ""
            sources = []
            for i, doc in enumerate(search_results['documents'][0]):
                meta = search_results['metadatas'][0][i]
                source = {
                    "title": str(meta.get('title', 'Unknown Title')),
                    "paper_id": str(meta.get('paper_id', 'N/A')),
                    "section": str(meta.get('section', 'N/A'))
                }
                sources.append(source)
                
                context_str += f"--- Context Snippet {i+1} from paper: {source['title']} ---\n"
                context_str += f"Section: {source['section']}\n"
                context_str += f"Content: {doc}\n\n"
            
            # Generate an answer using the LLM
            prompt = f"""Answer the user's question based ONLY on the provided context snippets.
            If the context doesn't contain enough information, say so. Be precise and include 
            citations like [1], [2], etc. where the number corresponds to the source index.
            
            <context>
            {context_str}
            </context>
            
            Question: {question}
            
            Answer:"""
            
            print("Generating answer with LLM...")
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return SuccessResponse(
                message="Successfully generated an answer from the knowledge base.",
                data={
                    "answer": answer,
                    "sources": sources
                }
            ).model_dump()
            
        except Exception as e:
            logger.error(f"Error in AnswerFromPapersTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"An error occurred while processing your question: {str(e)}. Please try rephrasing your question or check if the knowledge base has relevant papers."
            ).model_dump()

    async def _arun(self, question: str) -> str:
        """Async version of the tool (not implemented)."""
        raise NotImplementedError("This tool does not support async yet.")



class ArxivSearchTool(BaseTool):
    """A tool to find and add new papers from arXiv."""
    name: str = "arxiv_paper_search_and_load"
    description: str = (
        "Use this tool when the user asks to find, search for, or load NEW papers "
        "from arXiv. This downloads the papers' PDFs and adds them to the knowledge base "
        "for other tools to use."
    )
    args_schema: Type[ArxivSearchInput] = ArxivSearchInput

    ingestor: Ingestor
    kb: KnowledgeBase

    def _run(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search arXiv for papers and add them to the knowledge base."""
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                return FailureResponse(
                    message="Invalid query: Please provide a non-empty search query.",
                    data={"query": query, "max_results": max_results}
                ).model_dump()
            
            # Ensure max_results is within reasonable bounds
            max_results = max(1, min(max_results, 10))  # Limit between 1-10 results
            
            print(f"Searching arXiv for '{query}' (max {max_results} results)...")
            papers = self.ingestor.load_from_arxiv(query=query, max_results=max_results)
            
            if not papers:
                return FailureResponse(
                    message=f"No papers were found on arXiv for the query: '{query}'. Try using more specific search terms or check if the query is spelled correctly.",
                    data={"query": query, "max_results": max_results}
                ).model_dump()
            
            # Add papers to the knowledge base
            print(f"Adding {len(papers)} papers to knowledge base...")
            self.kb.add_papers(papers)
            
            # Prepare the response
            paper_details = []
            for paper in papers:
                if hasattr(paper, 'title'):
                    paper_details.append({
                        "title": str(paper.title),
                        "authors": [str(author) for author in paper.authors] if hasattr(paper, 'authors') else [],
                        "published": str(paper.published) if hasattr(paper, 'published') else None,
                        "arxiv_id": paper.entry_id.split('/')[-1] if hasattr(paper, 'entry_id') else None,
                        "paper_id": getattr(paper, 'paper_id', None)
                    })
            
            return SuccessResponse(
                message=f"Successfully loaded {len(papers)} papers into the knowledge base.",
                data={
                    "count": len(papers),
                    "papers": paper_details,
                    "query": query
                }
            ).model_dump()
            
        except Exception as e:
            logger.error(f"Error in ArxivSearchTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"An error occurred while searching arXiv: {str(e)}. Please try again with a different query or check your internet connection.",
                data={"query": query, "max_results": max_results}
            ).model_dump()

    async def _arun(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(query, max_results)


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
    args_schema: Type[SummarizationInput] = SummarizationInput

    kb: KnowledgeBase
    extractor: Extractor

    def _run(self, paper_id: str) -> Dict[str, Any]:
        """Summarize a paper by its ID."""
        try:
            # Validate inputs
            if not paper_id or not isinstance(paper_id, str):
                return FailureResponse(
                    message="Invalid paper_id: Please provide a valid paper ID.",
                    data={"paper_id": paper_id}
                ).model_dump()
            
            print(f"Summarizing paper with ID: {paper_id}")
            
            results = self.kb.collection.get(where={"paper_id": paper_id})
            
            if not results or not results.get('documents'):
                return FailureResponse(
                    message=f"Could not find a paper with ID '{paper_id}' in the knowledge base. Please check the paper ID or add the paper to the knowledge base first.",
                    data={"paper_id": paper_id}
                ).model_dump()

            import shutil, os
            pdf_filename = None
            try:
                if os.path.exists(paper_id) and paper_id.endswith('.pdf'):
                    pdf_filename = os.path.basename(paper_id)
                elif paper_id.startswith('http') and 'arxiv.org' in paper_id:
                    arxiv_id = paper_id.split('/')[-1].split('v')[0]
                    for fname in os.listdir('.'):
                        if fname.startswith(arxiv_id) and fname.endswith('.pdf'):
                            pdf_filename = fname
                            break
                    if not pdf_filename:
                        try:
                            import arxiv
                            result = next(arxiv.Search(id_list=[arxiv_id]).results())
                            pdf_path = result.download_pdf()
                            pdf_filename = os.path.basename(pdf_path)
                            print(f"Downloaded missing PDF for artifacts: {pdf_filename}")
                        except Exception as e:
                            print(f"Failed to download missing PDF for artifacts: {e}")
                if pdf_filename:
                    artifacts_dir = 'artifacts'
                    dest_path = os.path.join(artifacts_dir, pdf_filename)
                    try:
                        if not os.path.exists(artifacts_dir):
                            os.makedirs(artifacts_dir)
                        if not os.path.exists(dest_path) and os.path.exists(pdf_filename):
                            shutil.copy(pdf_filename, dest_path)
                            print(f"Copied referenced PDF to artifacts: {dest_path}")
                    except Exception as e:
                        print(f"Failed to copy referenced PDF to artifacts: {e}")
            except Exception as e:
                print(f"PDF artifact logic error: {e}")

            documents = results.get('documents')
            if not documents:
                return FailureResponse(
                    message=f"Could not find a paper with ID '{paper_id}' in the knowledge base. Please check the paper ID or add the paper to the knowledge base first.",
                    data={"paper_id": paper_id}
                ).model_dump()

            full_text = " ".join([str(doc) for doc in documents])
            if not full_text.strip():
                return FailureResponse(
                    message=f"Found paper with ID '{paper_id}' but it has no content to summarize.",
                    data={"paper_id": paper_id}
                ).model_dump()
                
            metadatas = results.get('metadatas')
            if metadatas and len(metadatas) > 0:
                raw_title = metadatas[0].get('title', 'Unknown Title')
                title = str(raw_title) if raw_title is not None else 'Unknown Title'
            else:
                title = 'Unknown Title'

            print(f"Generating summary for paper: {title}")
            summary = self.extractor.summarize_paper_text(full_text, title)
            
            return SuccessResponse(
                message=f"Successfully generated summary for paper: {title}",
                data={
                    "paper_id": paper_id,
                    "title": title,
                    "summary": summary
                }
            ).model_dump()
            
        except Exception as e:
            logger.error(f"Error in PaperSummarizationTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"An error occurred while summarizing the paper: {str(e)}. Please check the paper ID and try again.",
                data={"paper_id": paper_id}
            ).model_dump()

    async def _arun(self, paper_id: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")

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

class PaperFinderInput(BaseModel):
    """Input model for the paper metadata finder tool."""
    query: str = Field(description="A query to find relevant papers, usually the title of the paper.")

class PaperSearchInput(BaseModel):
    """Input model for searching papers by title or keywords."""
    query: str = Field(description="The title or keywords to search for in paper titles.")
    max_results: int = Field(default=5, description="Maximum number of results to return.")

class GetPaperMetadataByTitleTool(BaseTool):
    """Tool to find papers by title or keywords in the knowledge base."""
    name: str = "get_paper_metadata_by_title"
    description: str = (
        "Use this tool to find papers by their title or keywords. "
        "Returns metadata including paper_id, title, and authors for matching papers."
    )
    args_schema: Type[BaseModel] = PaperSearchInput
    
    kb: KnowledgeBase

    def _run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for papers by title or keywords in the knowledge base."""
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                return FailureResponse(
                    message="Invalid query: Please provide a non-empty search query.",
                    data={"query": query, "max_results": max_results}
                ).model_dump()
            
            # Ensure max_results is within reasonable bounds
            max_results = max(1, min(max_results, 10))  # Limit between 1-10 results
            
            print(f"Searching knowledge base for papers matching '{query}' (max {max_results} results)...")
            
            # First try to find matches using vector similarity search
            search_results = self.kb.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["metadatas", "documents"]
            )
            
            if not search_results or not search_results.get('metadatas') or not search_results['metadatas'][0]:
                print("Vector search returned no results, trying local filter...")
                # Fallback to get all and filter locally if vector search returns nothing
                all_papers = self.kb.collection.get(
                    limit=100,  # Limit to prevent memory issues
                    include=["metadatas", "documents"]
                )
                
                if not all_papers or not all_papers.get('metadatas'):
                    return FailureResponse(
                        message=f"No papers found in the knowledge base. Please add some papers first using the arxiv_paper_search_and_load tool.",
                        data={"query": query, "max_results": max_results}
                    ).model_dump()
                
                # Filter papers locally by checking if query is in title (case-insensitive)
                query_lower = query.lower()
                matching_metadatas = []
                
                for meta in all_papers['metadatas']:
                    if not meta or 'title' not in meta:
                        continue
                    title = str(meta['title']).lower()
                    if query_lower in title:
                        matching_metadatas.append(meta)
                    
                    if len(matching_metadatas) >= max_results:
                        break
                
                if not matching_metadatas:
                    return FailureResponse(
                        message=f"No papers found matching the query: '{query}'. Try using different search terms or check if the papers have been added to the knowledge base.",
                        data={"query": query, "max_results": max_results}
                    ).model_dump()
                
                results = {'metadatas': [matching_metadatas]}
            else:
                results = search_results
            
            # Process results to get unique papers by paper_id
            unique_papers = {}
            for meta in results['metadatas'][0]:
                if not meta or 'paper_id' not in meta or not meta['paper_id']:
                    continue
                
                paper_id = meta['paper_id']
                if paper_id not in unique_papers:
                    unique_papers[paper_id] = {
                        "paper_id": paper_id,
                        "title": str(meta.get('title', 'Untitled')),
                        "authors": [str(author) for author in meta.get('authors', [])],
                        "year": meta.get('year'),
                        "doi": meta.get('doi')
                    }
                
                if len(unique_papers) >= max_results:
                    break
        
            papers = list(unique_papers.values())
            
            return SuccessResponse(
                message=f"Found {len(papers)} matching papers.",
                data={
                    "count": len(papers),
                    "papers": papers,
                    "query": query
                }
            ).model_dump()
            
        except Exception as e:
            logger.error(f"Error in GetPaperMetadataByTitleTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"An error occurred while searching for papers: {str(e)}. Please try again or check if the knowledge base has papers.",
                data={"query": query, "max_results": max_results}
            ).model_dump()

    async def _arun(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(query, max_results)


class PaperSummarizationInput(BaseModel):
    """Input model for paper summarization.
    
    Either paper_id OR title must be provided. If both are provided, paper_id takes precedence.
    """
    paper_id: Optional[str] = Field(
        default=None,
        description="The unique ID of the paper to summarize. If not provided, title will be used for lookup."
    )
    title: Optional[str] = Field(
        default=None,
        description="Title of the paper to summarize. Only used if paper_id is not provided."
    )
    summary_length: str = Field(
        default="concise", 
        description="Desired length of the summary. Options: 'brief' (1-2 sentences), 'concise' (1 paragraph), or 'detailed' (multiple paragraphs)."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper_id": "1706.03762",
                "title": "Attention Is All You Need",
                "summary_length": "concise"
            }
        }

class PaperSummarizationTool(BaseTool):
    """Tool to generate summaries of scientific papers."""
    name: str = "summarize_paper"
    description: str = (
        "Use this tool to generate a summary of a specific scientific paper. "
        "You must provide the paper_id of the paper to summarize."
    )
    args_schema: Type[BaseModel] = PaperSummarizationInput
    
    kb: KnowledgeBase
    extractor: Extractor

    def _get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a paper from the knowledge base by its ID."""
        results = self.kb.collection.get(where={"paper_id": paper_id})
        if not results or 'documents' not in results or not results['documents']:
            return None
        return results
    
    def _search_paper_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Search for a paper in the knowledge base by its title."""
        # First try exact match
        results = self.kb.collection.query(
            query_texts=[title],
            n_results=1,
            where={"title": {"$eq": title}},
            include=['documents', 'metadatas']
        )
        
        # If no exact match, try semantic search
        if not results or not results.get('documents') or not results['documents'][0]:
            results = self.kb.collection.query(
                query_texts=[title],
                n_results=1,
                include=['documents', 'metadatas']
            )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return None
            
        # Reformat to match the structure of _get_paper_by_id
        return {
            'documents': results['documents'][0],
            'metadatas': results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        }
    
    def _run(self, paper_id: Optional[str] = None, title: Optional[str] = None, summary_length: str = "concise") -> Dict[str, Any]:
        """Generate a summary of the specified paper.
        
        Args:
            paper_id: The unique ID of the paper to summarize (takes precedence if both are provided)
            title: Title of the paper to summarize (used if paper_id is not provided)
            summary_length: Desired length of the summary ('brief', 'concise', or 'detailed')
            
        Returns:
            Dict containing the summary and metadata, or an error message
        """
        try:
            # Validate inputs
            if not paper_id and not title:
                return FailureResponse(
                    message="Either 'paper_id' or 'title' must be provided.",
                    data={"summary_length": summary_length}
                ).model_dump()
                
            # Validate summary length
            if summary_length not in ["brief", "concise", "detailed"]:
                summary_length = "concise"  # Default to concise if invalid
            
            # Try to get the paper by ID first
            results = None
            paper_source = None
            
            if paper_id:
                results = self._get_paper_by_id(paper_id)
                paper_source = f"ID: {paper_id}"
            
            # If not found by ID, try searching by title
            if not results and title:
                results = self._search_paper_by_title(title)
                paper_source = f"title: {title}"
            
            if not results:
                return FailureResponse(
                    message=f"Could not find paper with {paper_source} in the knowledge base.",
                    data={"paper_id": paper_id, "title": title} if paper_id else {"title": title}
                ).model_dump()
            
            # Combine all document chunks
            full_text = "\n\n".join(doc for doc in results['documents'] if doc)
            
            # Get metadata
            metadata = results['metadatas'][0] if results.get('metadatas') and results['metadatas'] else {}
            title = metadata.get('title', title or 'Unknown Title')
            paper_id = metadata.get('paper_id', paper_id or 'unknown')
            
            # Generate summary - the extractor will handle the summary length based on the prompt
            summary = self.extractor.summarize_paper_text(
                paper_text=full_text,
                paper_title=title
            )
            
            return SuccessResponse(
                message=f"Successfully generated a {summary_length} summary for '{title}'",
                data={
                    "paper_id": paper_id,
                    "title": title,
                    "summary": summary,
                    "summary_length": summary_length
                }
            ).model_dump()
            
        except Exception as e:
            logger.error(f"Error in PaperSummarizationTool: {e}")
            return FailureResponse(
                message=f"An error occurred while generating the summary: {str(e)}",
                data={"paper_id": paper_id, "summary_length": summary_length}
            ).model_dump()

    async def _arun(self, paper_id: Optional[str] = None, title: Optional[str] = None, summary_length: str = "concise") -> Dict[str, Any]:
        """Async version of the tool."""
        return self._run(paper_id=paper_id, title=title, summary_length=summary_length)

class QuestionAnsweringInput(BaseModel):
    """Input model for the Question Answering Tool."""
    question: str = Field(description="A detailed, specific question to ask about the content of the papers.")

class GraphQueryInput(BaseModel):
    """Input model for the Graph Query Tool."""
    query: str = Field(description="A Cypher query to run against the Neo4j graph database.")

class GraphQueryTool(BaseTool):
    """Answers questions about relationships between papers and authors by querying the knowledge graph.
    
    This tool executes Cypher queries against a Neo4j graph database containing paper and author relationships.
    It's particularly useful for finding connections, collaborations, and citation patterns in academic literature.
    """
    name: str = "graph_query_tool"
    description: str = (
        "**This is the best tool for answering questions about authors, collaborations, or connections between papers.** "
        "Use this tool when the user asks 'Who wrote X?', 'Who works with Y?', or 'What papers cite Z?'. "
        "Input must be a valid Cypher query."
    )
    args_schema: Type[GraphQueryInput] = GraphQueryInput
    
    graph: Neo4jGraph
    max_results: int = 50  # Limit number of results to prevent overwhelming output

    def _run(self, query: str) -> Dict[str, Any]:
        """Execute a Cypher query against the knowledge graph.
        
        Args:
            query: A valid Cypher query string
            
        Returns:
            Dict containing query results or error information
        """
        print(f"Running Cypher query: {query}")
        try:
            # Execute the query
            result = self.graph.query(query)
            
            if not result:
                return SuccessResponse(
                    message="Query executed successfully but returned no results.",
                    data={"result_count": 0}
                ).model_dump()
            
            # Process results - limit number of results to avoid huge responses
            values = [list(record.values())[0] for record in result[:self.max_results] if record]
            
            if not values:
                return SuccessResponse(
                    message="Query executed but did not return any values.",
                    data={"result_count": 0}
                ).model_dump()
            
            # For large result sets, just return a summary
            if len(result) > self.max_results:
                return SuccessResponse(
                    message=f"Query returned {len(result)} results, showing first {self.max_results}.",
                    data={
                        "result_count": len(result),
                        "results": values,
                        "truncated": True,
                        "max_results": self.max_results
                    }
                ).model_dump()
                
            return SuccessResponse(
                message=f"Query returned {len(values)} results.",
                data={
                    "result_count": len(values),
                    "results": values,
                    "truncated": False
                }
            ).model_dump()
            
        except Exception as e:
            return FailureResponse(
                message=f"Error executing Cypher query: {str(e)}",
                data={
                    "error_type": type(e).__name__,
                    "query": query[:200] + ("..." if len(query) > 200 else "")  # Include part of query for debugging
                }
            ).model_dump()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Include schema in the tool description for LLM reference
        try:
            schema = self.graph.get_schema
            if callable(schema):
                schema = schema()
            # Escape curly braces to prevent formatting issues
            escaped_schema = schema.replace("{", "{{").replace("}", "}}")
            self.description += f"\n\n**Graph Schema:**\n```\n{escaped_schema}\n```"
        except Exception as e:
            print(f"Warning: Could not load graph schema: {e}")
            self.description += "\n\n**Warning:** Could not load graph schema."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")

class TableExtractionInput(BaseModel):
    paper_id: str = Field(description="The ID of the paper from which to extract tables.")
    topic_of_interest: str = Field(description="The specific topic or type of data to look for in tables (e.g., 'model performance metrics', 'ablation study results').")

class TableExtractionTool(BaseTool):
    name: str = "table_extraction_tool"
    description: str = (
        "Use this tool to extract structured tabular data from a paper. "
        "It returns the data in a structured JSON format, NOT as a Markdown table."
    )
    args_schema: Type[TableExtractionInput] = TableExtractionInput
    kb: KnowledgeBase
    extractor: Extractor

    def _run(self, paper_id: str, topic_of_interest: str) -> Dict[str, Any]:
        """Extract tables from a paper based on a topic of interest."""
        try:
            # Validate inputs
            if not paper_id or not isinstance(paper_id, str):
                return FailureResponse(
                    message="Invalid paper_id: Please provide a valid paper ID.",
                    data={"paper_id": paper_id, "topic_of_interest": topic_of_interest}
                ).model_dump()
                
            if not topic_of_interest or not isinstance(topic_of_interest, str):
                return FailureResponse(
                    message="Invalid topic_of_interest: Please provide a valid topic.",
                    data={"paper_id": paper_id, "topic_of_interest": topic_of_interest}
                ).model_dump()
            
            print(f"Extracting table about '{topic_of_interest}' from paper ID: {paper_id}")
            
            # Get paper from knowledge base
            results = self.kb.collection.get(where={"paper_id": paper_id})
            if not results or not results.get('documents'):
                return FailureResponse(
                    message=f"Could not find paper with ID '{paper_id}' in the knowledge base. Please check the paper ID or add the paper to the knowledge base first.",
                    data={"paper_id": paper_id}
                ).model_dump()
            
            # Get paper metadata
            full_text = " ".join([str(doc) for doc in results['documents']])
            if not full_text.strip():
                return FailureResponse(
                    message=f"Found paper with ID '{paper_id}' but it has no text content to extract tables from.",
                    data={"paper_id": paper_id}
                ).model_dump()
                
            metadatas = results.get('metadatas', [{}])
            title = str(metadatas[0].get('title', 'Unknown Title'))
            
            # Try LLM-based extraction first
            try:
                print(f"Attempting LLM-based table extraction for topic: {topic_of_interest}")
                json_table_str = self.extractor.extract_table_as_json(full_text, title, topic_of_interest)
                
                # Clean and parse the response
                if "```json" in json_table_str:
                    json_table_str = json_table_str.split("```json")[1].split("```")[0].strip()
                
                parsed_data = json.loads(json_table_str)
                if isinstance(parsed_data, dict) and "columns" in parsed_data and "data" in parsed_data:
                    # Validate the extracted data
                    columns = parsed_data.get("columns", [])
                    data = parsed_data.get("data", [])
                    
                    if columns and data:
                        return SuccessResponse(
                            message=f"Successfully extracted table about '{topic_of_interest}' from paper: {title}",
                            data={
                                "columns": [str(col) for col in columns],
                                "data": [[str(cell) for cell in row] for row in data],
                                "paper_id": paper_id,
                                "title": title,
                                "extraction_method": "llm"
                            }
                        ).model_dump()
                    else:
                        print("LLM extraction returned empty table data")
                else:
                    print("LLM extraction did not return valid table format")
                    
            except json.JSONDecodeError as e:
                print(f"LLM-based table extraction failed to parse JSON: {e}")
            except Exception as e:
                print(f"LLM-based table extraction failed: {e}")
            
            # Fall back to PDF extraction if available
            try:
                import camelot
                pdf_path = self._find_pdf_path(paper_id)
                if pdf_path and os.path.exists(pdf_path):
                    print(f"Attempting PDF-based table extraction from: {pdf_path}")
                    tables = camelot.read_pdf(pdf_path, pages='all')
                    if tables and len(tables) > 0:
                        df = tables[0].df
                        if not df.empty:
                            # Convert to proper format
                            columns = [str(col) for col in df.iloc[0]]
                            data = [[str(cell) for cell in row] for row in df.iloc[1:].values.tolist()]
                            
                            table_data = {
                                "columns": columns,
                                "data": data,
                                "paper_id": paper_id,
                                "title": title,
                                "extraction_method": "pdf_camelot"
                            }
                            return SuccessResponse(
                                message=f"Extracted table from PDF using Camelot: {title}",
                                data=table_data
                            ).model_dump()
                        else:
                            print("PDF extraction returned empty table")
                    else:
                        print("PDF extraction found no tables")
                else:
                    print("PDF file not found for extraction")
            except ImportError as e:
                print(f"Camelot not installed, PDF extraction unavailable: {e}")
            except Exception as e:
                print(f"PDF extraction failed: {e}")
            
            return FailureResponse(
                message=f"Could not extract table about '{topic_of_interest}' from paper: {title}. Both LLM-based and PDF-based extraction methods failed. Please try a different topic or check if the paper contains tables.",
                data={"paper_id": paper_id, "title": title}
            ).model_dump()
            
        except Exception as e:
            logger.error(f"Error in TableExtractionTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"Error extracting table: {str(e)}. Please check the paper ID and try again.",
                data={"paper_id": paper_id, "error_type": type(e).__name__}
            ).model_dump()
    
    def _find_pdf_path(self, paper_id: str) -> Optional[str]:
        """Helper to find PDF path from paper ID."""
        import os
        
        if os.path.exists(paper_id) and paper_id.endswith('.pdf'):
            return paper_id
            
        if paper_id.startswith('http') and 'arxiv.org' in paper_id:
            arxiv_id = paper_id.split('/')[-1].split('v')[0]
            for fname in os.listdir('.'):
                if fname.startswith(arxiv_id) and fname.endswith('.pdf'):
                    return fname
        return None

class RelationshipInput(BaseModel):
    paper_a_title: str = Field(description="The title of the first paper.")
    paper_b_title: str = Field(description="The title of the second paper.")

class RelationshipAnalysisTool(BaseTool):
    """Tool for analyzing and explaining relationships between two papers in the knowledge graph.
    
    This tool finds and explains connections, citations, and other relationships between
    academic papers by querying the knowledge graph and generating natural language explanations.
    """
    name: str = "relationship_analysis_tool"
    description: str = (
            "Use this tool to explain the relationship between two papers. "
            "It queries the knowledge graph to find citation paths and other connections. "
            "Provide the titles of the two papers to analyze."
        )
    args_schema: Type[RelationshipInput] = RelationshipInput
    graph: Neo4jGraph
    llm: BaseChatModel
    max_path_length: int = 5  # Maximum path length to search for relationships

    def _run(self, paper_a_title: str, paper_b_title: str) -> Dict[str, Any]:
        """Analyze the relationship between two papers.
        
        Args:
            paper_a_title: Title or partial title of the first paper
            paper_b_title: Title or partial title of the second paper
            
        Returns:
            Dict containing the analysis results or error information
        """
        try:
            # First, find the paper nodes
            query = """
            MATCH (a:Paper), (b:Paper)
            WHERE toLower(a.title) CONTAINS toLower($title_a) AND toLower(b.title) CONTAINS toLower($title_b)
            RETURN a, b
            LIMIT 1
            """
            result = self.graph.query(query, params={"title_a": paper_a_title, "title_b": paper_b_title})
            
            if not result or len(result) == 0:
                return FailureResponse(
                    message="Could not find one or both papers in the knowledge graph.",
                    data={"paper_a_query": paper_a_title, "paper_b_query": paper_b_title}
                ).model_dump()
                
            paper_a = result[0]['a']
            paper_b = result[0]['b']
            
            # Find the shortest path between them
            path_query = f"""
            MATCH path = shortestPath((a:Paper)-[*..{self.max_path_length}]-(b:Paper))
            WHERE a.title = $title_a AND b.title = $title_b
            RETURN path
            """
            path_result = self.graph.query(path_query, params={"title_a": paper_a['title'], "title_b": paper_b['title']})
            
            if not path_result or len(path_result) == 0:
                return SuccessResponse(
                    message=f"No direct relationship found between the papers.",
                    data={
                        "paper_a": paper_a['title'],
                        "paper_b": paper_b['title'],
                        "relationship_found": False
                    }
                ).model_dump()
            
            # Format the path for display and analysis
            path = path_result[0]['path']
            path_description = self._format_path(path)
            
            # Generate a natural language explanation
            explanation = self._generate_explanation(paper_a, paper_b, path_description)
            
            return SuccessResponse(
                message="Relationship analysis completed successfully.",
                data={
                    "paper_a": paper_a['title'],
                    "paper_b": paper_b['title'],
                    "relationship_found": True,
                    "path_length": len(path) - 1,  # Number of edges
                    "path_description": path_description,
                    "explanation": explanation
                }
            ).model_dump()
            
        except Exception as e:
            return FailureResponse(
                message=f"Error analyzing relationship: {str(e)}",
                data={
                    "error_type": type(e).__name__,
                    "paper_a_query": paper_a_title,
                    "paper_b_query": paper_b_title
                }
            ).model_dump()

class CitationAnalysisInput(BaseModel):
    analysis_type: str = Field(description="The type of citation analysis to perform. Options: 'most_cited', 'hottest_papers'.")
    limit: int = Field(default=5, description="The number of papers to return.")

class CitationAnalysisTool(BaseTool):
    """Performs citation analysis on the knowledge graph. 
    
    This tool can identify:
    - Most cited papers (highest incoming citations)
    - Most connected papers (most outgoing references)
    - Foundational papers (highly cited and highly connected)
    - Citation network for a specific paper
    
    Results are returned in a structured format with detailed metadata.
    """
    name: str = "citation_analysis_tool"
    description: str = (
        "Use this to analyze citation patterns and find influential papers. "
        "Analysis types:\n"
        "- 'most_cited': Papers with the most incoming citations\n"
        "- 'hottest_papers': Papers that cite many others (broad literature review)\n"
        "- 'foundational': Highly cited papers that also cite many works\n"
        "- 'paper_network': Citation network for a specific paper"
    )
    args_schema: Type[CitationAnalysisInput] = CitationAnalysisInput
    graph: Neo4jGraph
    max_results: int = 20  # Increased default max results
    llm: Optional[BaseChatModel] = None  # Optional LLM for enhanced analysis
    
    class Config:
        arbitrary_types_allowed = True  # Allow Neo4jGraph type

    def _run(self, analysis_type: str, limit: int = 5, paper_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform citation analysis on the knowledge graph.
        
        Args:
            analysis_type: Type of analysis to perform 
                         ('most_cited', 'hottest_papers', 'foundational', 'paper_network')
            limit: Maximum number of results to return (capped at max_results)
            paper_id: Optional paper ID for paper-specific analysis
            
        Returns:
            Dict containing analysis results or error information
        """
        # Validate input
        if not analysis_type or not isinstance(analysis_type, str):
            return FailureResponse(
                message="Invalid analysis_type: Please provide a valid analysis type.",
                data={
                    "valid_types": ["most_cited", "hottest_papers", "foundational", "paper_network"],
                    "provided_type": analysis_type,
                    "paper_id_provided": bool(paper_id)
                }
            )
        
        limit = min(max(1, limit), self.max_results)  # Ensure limit is reasonable
        
        try:
            print(f"Performing citation analysis: {analysis_type} (limit: {limit})")
            
            # Check if graph is connected
            if not self._check_graph_connection():
                return FailureResponse(
                    message="Failed to connect to the knowledge graph. Please check the connection. Make sure Neo4j is running and properly configured.",
                    data={"analysis_type": analysis_type}
                )
            
            # Route to appropriate analysis method
            if analysis_type == "most_cited":
                result = self._find_most_cited(limit)
                if result.get('status') == 'success' and result.get('data', {}).get('result_count', 0) == 0:
                    result['message'] += " This may be because no papers have been added to the knowledge base yet, or no citation relationships exist in the graph."
                return result
            elif analysis_type == "hottest_papers":
                result = self._find_hottest_papers(limit)
                if result.get('status') == 'success' and result.get('data', {}).get('result_count', 0) == 0:
                    result['message'] += " This may be because no papers have been added to the knowledge base yet, or no citation relationships exist in the graph."
                return result
            elif analysis_type == "foundational":
                result = self._find_foundational_papers(limit)
                if result.get('status') == 'success' and result.get('data', {}).get('result_count', 0) == 0:
                    result['message'] += " This may be because no papers have been added to the knowledge base yet, or no citation relationships exist in the graph."
                return result
            elif analysis_type == "paper_network" and paper_id:
                result = self._analyze_paper_network(paper_id, limit)
                if result.get('status') == 'success' and result.get('data', {}).get('result_count', 0) == 0:
                    result['message'] += " This may be because the specified paper was not found or has no citation relationships in the graph."
                return result
            else:
                return FailureResponse(
                    message=f"Invalid analysis type or missing paper_id: {analysis_type}. Valid types are: most_cited, hottest_papers, foundational, paper_network.",
                    data={
                        "valid_types": ["most_cited", "hottest_papers", "foundational", "paper_network"],
                        "provided_type": analysis_type,
                        "paper_id_provided": bool(paper_id)
                    }
                )
                
        except Exception as e:
            error_msg = f"Error performing {analysis_type} analysis"
            error_data = {
                "error_type": type(e).__name__,
                "analysis_type": analysis_type,
                "limit": limit,
                "paper_id": paper_id
            }
            
            # Add more context for specific error types
            if "ConnectionError" in str(e) or "ServiceUnavailable" in str(e):
                error_msg = "Failed to connect to the knowledge graph. The service may be down or unreachable."
                error_data["suggestion"] = "Please check your Neo4j connection and try again."
            
            return FailureResponse(
                message=f"{error_msg}: {str(e)}. Please check your Neo4j connection and try again.",
                data=error_data
            )
    
    def _check_graph_connection(self, max_retries: int = 2) -> bool:
        """Check if the graph database is accessible with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                # Simple query to test connection
                self.graph.query("RETURN 1 AS test")
                print("Graph connection successful")
                return True
            except Exception as e:
                logger.warning(f"Graph connection check failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    import time
                    time.sleep(1)  # Brief delay before retry
                else:
                    logger.error(f"Graph connection failed after {max_retries + 1} attempts: {e}")
                    print(f"Graph connection failed after {max_retries + 1} attempts. Please check if Neo4j is running and properly configured.")
        return False
            
    def _find_most_cited(self, limit: int) -> Dict[str, Any]:
        """Find the most cited papers in the knowledge graph."""
        try:
            query = """
            MATCH (p:Paper)<-[:CITES]-(cited_by)
            WITH p, count(cited_by) AS citation_count
            RETURN p.title AS title, 
                   p.year AS year,
                   p.authors AS authors,
                   citation_count
            ORDER BY citation_count DESC
            LIMIT $limit
            """
            result = self.graph.query(query, params={"limit": limit})
            
            if not result:
                return SuccessResponse(
                    message="No citation data found in the knowledge graph.",
                    data={"result_count": 0}
                )
                
            # Process results to ensure proper data types
            processed_papers = []
            for paper in result:
                processed_paper = {
                    "title": str(paper.get('title', 'Unknown Title')),
                    "year": paper.get('year'),
                    "authors": [str(author) for author in paper.get('authors', [])] if paper.get('authors') else [],
                    "citation_count": paper.get('citation_count', 0)
                }
                processed_papers.append(processed_paper)
                
            return SuccessResponse(
                message=f"Found {len(result)} most cited papers.",
                data={
                    "analysis_type": "most_cited",
                    "papers": processed_papers,
                    "result_count": len(result)
                }
            )
        except Exception as e:
            logger.error(f"Error in _find_most_cited: {e}", exc_info=True)
            return FailureResponse(
                message=f"Error finding most cited papers: {str(e)}",
                data={"result_count": 0}
            )
        
    def _find_hottest_papers(self, limit: int) -> Dict[str, Any]:
        """Find papers with the most outgoing references."""
        query = """
        MATCH (p:Paper)-[:CITES]->(cited)
        WITH p, count(cited) AS reference_count
        RETURN p.title AS title,
               p.year AS year,
               p.authors AS authors,
               reference_count
        ORDER BY reference_count DESC
        LIMIT $limit
        """
        result = self.graph.query(query, params={"limit": limit})
        
        if not result:
            return SuccessResponse(
                message="No reference data found in the knowledge graph.",
                data={"result_count": 0}
            )
            
        return SuccessResponse(
            message=f"Found {len(result)} papers with the most references.",
            data={
                "analysis_type": "hottest_papers",
                "papers": [dict(paper) for paper in result],
                "result_count": len(result)
            }
        )
        
    def _find_foundational_papers(self, limit: int) -> Dict[str, Any]:
        """Find papers that are both highly cited and reference many other works."""
        query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)<-[:CITES]-(cited_by)
        WITH p, count(DISTINCT cited_by) AS citation_count
        OPTIONAL MATCH (p)-[:CITES]->(ref)
        WITH p, citation_count, count(DISTINCT ref) AS reference_count,
             (citation_count * 1.0) / NULLIF((citation_count + reference_count), 0) AS citation_ratio
        WHERE citation_count > 0 AND reference_count > 0
        RETURN p.title AS title,
               p.year AS year,
               p.authors AS authors,
               citation_count,
               reference_count,
               citation_ratio
        ORDER BY citation_ratio DESC, citation_count DESC
        LIMIT $limit
        """
        result = self.graph.query(query, params={"limit": limit})
        
        if not result:
            return SuccessResponse(
                message="No foundational papers found with both incoming and outgoing citations.",
                data={"result_count": 0}
            )
            
        return SuccessResponse(
            message=f"Found {len(result)} foundational papers (highly cited with many references).",
            data={
                "analysis_type": "foundational",
                "papers": [dict(paper) for paper in result],
                "result_count": len(result)
            }
        )

    paper_id: str = Field(description="The ID of the paper from which to extract keywords.")
    num_keywords: int = Field(default=10, description="The number of keywords to extract.")

class KeywordExtractionInput(BaseModel):
    """Input model for the Keyword Extraction Tool."""
    paper_id: str = Field(description="The ID of the paper to analyze.")
    num_keywords: int = Field(default=10, description="Number of keywords to extract (default: 10).")


class LiteratureGapInput(BaseModel):
    """Input model for the Literature Gap Analysis Tool."""
    paper_ids: List[str] = Field(description="List of paper IDs to analyze for gaps.")
    query: Optional[str] = Field(
        default=None, 
        description="Optional query to focus the gap analysis (e.g., 'machine learning in healthcare')."
    )


class LiteratureGapTool(BaseTool):
    """Identifies potential gaps or limitations in the existing literature.
    
    This tool analyzes a set of papers to find areas that are underexplored, 
    contradictory findings, or opportunities for future research. It's particularly 
    useful for identifying novel research directions or areas needing further investigation.
    """
    name: str = "literature_gap_analysis"
    description: str = (
        "Use this to identify gaps, contradictions, or underexplored areas in the literature. "
        "Provide a list of paper_ids and an optional query to focus the analysis."
    )
    args_schema: Type[BaseModel] = LiteratureGapInput
    
    kb: KnowledgeBase
    extractor: Extractor
    max_keywords: int = 50  # Safety limit to prevent excessive processing

class KeywordExtractionTool(BaseTool):
    """Extracts the most important keywords or concepts from a given paper.
    
    This tool analyzes the content of a paper and identifies the most significant
    terms and concepts using statistical analysis of term frequency and other
    linguistic features. It's useful for quickly understanding the main topics
    and focus areas of a research paper.
    """
    name: str = "keyword_extraction_tool"
    description: str = (
        "Use this to identify the main topics, concepts, or keywords of a specific paper. "
        "Provide the paper_id and optionally the number of keywords to extract (default: 10)."
    )
    args_schema: Type[KeywordExtractionInput] = KeywordExtractionInput
    
    kb: KnowledgeBase
    extractor: Extractor
    max_keywords: int = 50  # Safety limit to prevent excessive processing

    def _run(self, paper_id: str, num_keywords: int = 10) -> Dict[str, Any]:
        """Extract keywords from a paper.
        
        Args:
            paper_id: The ID of the paper to analyze
            num_keywords: Number of keywords to extract (capped at max_keywords)
            
        Returns:
            Dict containing the extracted keywords and metadata
        """
        # Validate input
        num_keywords = min(max(1, num_keywords), self.max_keywords)
        
        try:
            # Get the paper content from the knowledge base
            results = self.kb.collection.get(
                where={"paper_id": paper_id}, 
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get('documents'):
                return FailureResponse(
                    message=f"Could not find paper with ID '{paper_id}' in the knowledge base.",
                    data={"paper_id": paper_id, "available_papers": self._get_available_papers()}
                )
            
            # Combine all text chunks for the paper
            paper_text = "\n\n".join(results['documents'])
            if not paper_text.strip():
                return FailureResponse(
                    message=f"Paper with ID '{paper_id}' has no text content to analyze.",
                    data={"paper_id": paper_id}
                )
            
            # Extract keywords using the extractor
            try:
                keywords = self.extractor.extract_keywords(paper_text, num_keywords=num_keywords)
                
                # Convert keywords to a consistent format (list of strings)
                if isinstance(keywords, list):
                    if all(isinstance(kw, (list, tuple)) and len(kw) >= 2 for kw in keywords):
                        # Handle case where keywords are returned as list of (keyword, score) tuples
                        keyword_list = [kw[0] for kw in keywords]
                        scores = [float(kw[1]) for kw in keywords]
                    else:
                        # Handle case where keywords are returned as flat list of strings
                        keyword_list = [str(kw) for kw in keywords]
                        scores = [1.0] * len(keyword_list)  # Default score of 1.0 if not provided
                else:
                    # Handle case where keywords is not a list
                    keyword_list = [str(keywords)] if keywords else []
                    scores = [1.0] * len(keyword_list)
                    
                # Prepare the result with consistent format
                paper_metadata = results['metadatas'][0] if results.get('metadatas') else {}
                result_data = {
                    "paper_id": paper_id,
                    "title": paper_metadata.get('title', 'Unknown'),
                    "authors": paper_metadata.get('authors', []),
                    "year": paper_metadata.get('year'),
                    "num_keywords_found": len(keyword_list),
                    "keywords": []
                }
                
                # Add keywords with scores if available
                if scores and len(scores) == len(keyword_list):
                    min_score = min(scores) if scores else 0
                    max_score = max(scores) if scores else 1.0
                    score_range = max_score - min_score if max_score > min_score else 1.0
                    
                    for kw, score in zip(keyword_list, scores):
                        normalized_score = (float(score) - min_score) / score_range if score_range > 0 else 1.0
                        result_data["keywords"].append({
                            "keyword": kw,
                            "score": float(score),
                            "normalized_score": normalized_score
                        })
                else:
                    # If no scores, just add keywords with default score of 1.0
                    result_data["keywords"] = [{"keyword": kw, "score": 1.0, "normalized_score": 1.0} 
                                             for kw in keyword_list]
                
            except Exception as e:
                return FailureResponse(
                    message=f"Error during keyword extraction: {str(e)}",
                    data={
                        "error_type": type(e).__name__,
                        "paper_id": paper_id,
                        "text_length": len(paper_text),
                        "error_details": str(e)
                    }
                )
            
            return SuccessResponse(
                message=f"Successfully extracted {len(keywords)} keywords from paper: {result_data['title']}",
                data=result_data
            )
            
        except Exception as e:
            return FailureResponse(
                message=f"Unexpected error extracting keywords: {str(e)}",
                data={
                    "error_type": type(e).__name__,
                    "paper_id": paper_id
                }
            )
    
    def _get_available_papers(self) -> List[Dict[str, Any]]:
        """Get a list of available papers in the knowledge base."""
        try:
            # Get a sample of papers (limit to 10 for performance)
            results = self.kb.collection.get(
                limit=10,
                include=["metadatas"]
            )
            
            if not results or not results.get('metadatas'):
                return []
                
            return [
                {"id": meta.get('paper_id', 'unknown'), "title": meta.get('title', 'Untitled')}
                for meta in results['metadatas']
                if meta and 'paper_id' in meta
            ]
        except Exception:
            return []

class PlottingInput(BaseModel):
    json_data: str = Field(description="A JSON string containing the table data, with 'columns' and 'data' keys.")
    chart_type: str = Field(description="The type of chart to generate (e.g., 'bar', 'line').")
    title: str = Field(description="The title for the chart.")
    filename: str = Field(description="The filename to save the plot to (e.g., 'performance_chart.png').")

class PlotGenerationTool(BaseTool):
    name: str = "plot_generation_tool"
    description: str = "Use this tool to generate a plot from structured JSON data and save it as an image file."
    args_schema: Type[PlottingInput] = PlottingInput

    def _run(self, json_data: str, chart_type: str, title: str, filename: str) -> Dict[str, Any]:
        """Generate a plot from JSON data and save it to a file.
        
        Args:
            json_data: JSON string with 'columns' and 'data' keys
            chart_type: Type of chart to generate (e.g., 'line', 'bar', 'scatter')
            title: Title for the plot
            filename: Output filename for the plot
            
        Returns:
            Dict containing success/failure status and detailed information
        """
        # Validate inputs
        if not json_data or not isinstance(json_data, str):
            return FailureResponse(
                message="Invalid json_data: Please provide valid JSON data with 'columns' and 'data' keys.",
                data={"chart_type": chart_type, "title": title, "filename": filename}
            )
            
        if not chart_type or not isinstance(chart_type, str):
            return FailureResponse(
                message="Invalid chart_type: Please provide a valid chart type (e.g., 'line', 'bar', 'scatter').",
                data={"chart_type": chart_type, "title": title, "filename": filename}
            )
            
        if not title or not isinstance(title, str):
            return FailureResponse(
                message="Invalid title: Please provide a valid title for the plot.",
                data={"chart_type": chart_type, "title": title, "filename": filename}
            )
            
        if not filename or not isinstance(filename, str):
            return FailureResponse(
                message="Invalid filename: Please provide a valid filename for the plot.",
                data={"chart_type": chart_type, "title": title, "filename": filename}
            )
        
        try:
            print(f"Generating {chart_type} plot with title: {title}")
            
            # Ensure artifacts directory exists
            artifacts_dir = os.getenv('ARTIFACTS_DIR', 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Ensure filename is in artifacts directory
            if not filename.startswith(artifacts_dir + os.sep):
                filename = os.path.join(artifacts_dir, filename)
            
            # Parse JSON data
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return FailureResponse(
                    message=f"Invalid JSON data: {str(e)}",
                    data={
                        "error_type": "JSONDecodeError",
                        "chart_type": chart_type, 
                        "title": title, 
                        "filename": filename
                    }
                )
            
            # Validate data structure
            if not isinstance(data, dict) or 'columns' not in data or 'data' not in data:
                return FailureResponse(
                    message="Invalid data format: Expected JSON with 'columns' and 'data' keys.",
                    data={
                        "received_keys": list(data.keys()) if isinstance(data, dict) else "Invalid JSON",
                        "chart_type": chart_type, 
                        "title": title, 
                        "filename": filename
                    }
                )
            
            # Create DataFrame
            try:
                df = pd.DataFrame(data['data'], columns=data['columns'])
                if df.empty:
                    return FailureResponse(
                        message="Empty data: No data provided to plot.",
                        data={"chart_type": chart_type, "title": title, "filename": filename}
                    )
            except Exception as e:
                return FailureResponse(
                    message=f"Error creating DataFrame: {str(e)}",
                    data={
                        "error_type": type(e).__name__,
                        "chart_type": chart_type, 
                        "title": title, 
                        "filename": filename
                    }
                )
            
            # Generate plot
            try:
                # Set index for plotting
                if len(df.columns) > 0:
                    df.set_index(df.columns[0], inplace=True)
                
                # Create plot
                ax = df.plot(kind=chart_type, title=title, figsize=(10, 6))
                plt.ylabel("Value")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save plot
                plt.savefig(filename)
                plt.close()
                
                # Create web-accessible path
                web_path = f"/artifacts/{os.path.basename(filename)}"
                
                return SuccessResponse(
                    message=f"Successfully generated and saved {chart_type} plot to '{filename}'.",
                    data={
                        "output_path": filename,
                        "web_path": web_path,
                        "filename": os.path.basename(filename),
                        "chart_type": chart_type,
                        "title": title,
                        "num_data_points": len(df),
                        "columns": list(df.columns)
                    }
                )
                
            except Exception as e:
                # Clean up any partially created plot
                plt.close()
                return FailureResponse(
                    message=f"Error generating plot: {str(e)}",
                    data={
                        "error_type": type(e).__name__,
                        "chart_type": chart_type, 
                        "title": title, 
                        "filename": filename
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in PlotGenerationTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"Unexpected error generating plot: {str(e)}",
                data={
                    "error_type": type(e).__name__,
                    "chart_type": chart_type, 
                    "title": title, 
                    "filename": filename
                }
            )

class SmartPlottingInput(BaseModel):
    json_data: str = Field(description="A JSON string with 'columns' and 'data' keys.")
    title: str = Field(description="The title for the chart.")
    filename: str = Field(description="The filename to save the plot to (e.g., 'performance_chart.png').")
    analysis_goal: str = Field(description="A short sentence describing what the plot should show or compare. E.g., 'Compare the error metrics (RMSE, MAE) of the models'.")

class SmartPlotGenerationTool(BaseTool):
    name: str = "smart_plot_generation_tool"
    description: str = (
        "An intelligent plotting tool. It takes structured JSON data and an analysis goal, "
        "and automatically chooses the best way to visualize the data to meet that goal. "
        "Use this to create insightful charts from extracted tables."
    )
    args_schema: Type[SmartPlottingInput] = SmartPlottingInput

    def _run(self, json_data: str, title: str, filename: str, analysis_goal: str) -> Dict[str, Any]:
        """Generate an insightful plot from JSON data based on an analysis goal and save it to a file.
        
        Args:
            json_data: JSON string with 'columns' and 'data' keys
            title: Title for the plot
            filename: Output filename for the plot
            analysis_goal: Description of what to analyze/visualize
            
        Returns:
            Dict containing success/failure status and detailed information
        """
        # Validate inputs
        if not json_data or not isinstance(json_data, str):
            return FailureResponse(
                message="Invalid json_data: Please provide valid JSON data with 'columns' and 'data' keys.",
                data={"title": title, "filename": filename, "analysis_goal": analysis_goal}
            )
            
        if not title or not isinstance(title, str):
            return FailureResponse(
                message="Invalid title: Please provide a valid title for the plot.",
                data={"title": title, "filename": filename, "analysis_goal": analysis_goal}
            )
            
        if not filename or not isinstance(filename, str):
            return FailureResponse(
                message="Invalid filename: Please provide a valid filename for the plot.",
                data={"title": title, "filename": filename, "analysis_goal": analysis_goal}
            )
            
        if not analysis_goal or not isinstance(analysis_goal, str):
            return FailureResponse(
                message="Invalid analysis_goal: Please provide a valid analysis goal.",
                data={"title": title, "filename": filename, "analysis_goal": analysis_goal}
            )
        
        try:
            print(f"Generating smart plot with title: {title} for goal: {analysis_goal}")
            
            # Ensure artifacts directory exists
            artifacts_dir = os.getenv('ARTIFACTS_DIR', 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Ensure filename is in artifacts directory
            if not filename.startswith(artifacts_dir + os.sep):
                filename = os.path.join(artifacts_dir, filename)
            
            # Parse JSON data
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return FailureResponse(
                    message=f"Invalid JSON data: {str(e)}",
                    data={
                        "error_type": "JSONDecodeError",
                        "title": title, 
                        "filename": filename, 
                        "analysis_goal": analysis_goal
                    }
                )
            
            # Validate data structure
            if not isinstance(data, dict) or 'columns' not in data or 'data' not in data:
                return FailureResponse(
                    message="Invalid data format: Expected JSON with 'columns' and 'data' keys.",
                    data={
                        "received_keys": list(data.keys()) if isinstance(data, dict) else "Invalid JSON",
                        "title": title, 
                        "filename": filename, 
                        "analysis_goal": analysis_goal
                    }
                )
            
            # Create DataFrame
            try:
                df = pd.DataFrame(data['data'], columns=data['columns'])
                if df.empty:
                    return FailureResponse(
                        message="Empty data: No data provided to plot.",
                        data={"title": title, "filename": filename, "analysis_goal": analysis_goal}
                    )
                
                # Set index for plotting
                if len(df.columns) > 0:
                    df.set_index(df.columns[0], inplace=True)
                    
            except Exception as e:
                return FailureResponse(
                    message=f"Error creating DataFrame: {str(e)}",
                    data={
                        "error_type": type(e).__name__,
                        "title": title, 
                        "filename": filename, 
                        "analysis_goal": analysis_goal
                    }
                )
            
            # Generate smart plot
            try:
                numeric_cols = df.select_dtypes(include=np.number).columns
                
                if len(numeric_cols) == 0:
                    return FailureResponse(
                        message="No numeric columns found in data. Cannot generate plot.",
                        data={
                            "columns": list(df.columns),
                            "numeric_columns": list(numeric_cols),
                            "title": title, 
                            "filename": filename, 
                            "analysis_goal": analysis_goal
                        }
                    )
                
                if len(numeric_cols) > 1:
                    max_vals = df[numeric_cols].max()
                    if len(max_vals) > 1 and max_vals.max() / max_vals.min() > 10:
                        # Use dual axis for data with very different scales
                        fig, ax1 = plt.subplots(figsize=(12, 7))
                        ax2 = ax1.twinx() # Create a second y-axis
                        
                        largest_col = max_vals.idxmax()
                        other_cols = [c for c in numeric_cols if c != largest_col]
                        
                        df[other_cols].plot(kind='bar', ax=ax1, position=0, width=0.4)
                        df[[largest_col]].plot(kind='bar', ax=ax2, color='red', position=1, width=0.4)

                        ax1.set_ylabel('Error Metrics')
                        ax2.set_ylabel(str(largest_col), color='red')
                        ax1.set_title(title)
                        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
                        
                        plot_type = "dual_axis_bar"

                    else: # All values are on a similar scale
                        df[numeric_cols].plot(kind='bar', title=title, figsize=(10, 6), rot=45)
                        plot_type = "single_axis_bar"
                else: # Only one numeric column
                    df[numeric_cols[0]].plot(kind='bar', title=title, figsize=(10, 6), rot=45)
                    plot_type = "single_column_bar"
                
                plt.tight_layout()
                plt.savefig(filename)
                plt.close()
                
                # Create web-accessible path
                web_path = f"/artifacts/{os.path.basename(filename)}"
                
                return SuccessResponse(
                    message=f"Successfully generated an insightful {plot_type} plot and saved it to '{filename}'.",
                    data={
                        "output_path": filename,
                        "web_path": web_path,
                        "filename": os.path.basename(filename),
                        "plot_type": plot_type,
                        "title": title,
                        "analysis_goal": analysis_goal,
                        "num_data_points": len(df),
                        "numeric_columns": list(numeric_cols),
                        "total_columns": len(df.columns)
                    }
                )
                
            except Exception as e:
                # Clean up any partially created plot
                plt.close()
                return FailureResponse(
                    message=f"Error generating smart plot: {str(e)}",
                    data={
                        "error_type": type(e).__name__,
                        "title": title, 
                        "filename": filename, 
                        "analysis_goal": analysis_goal
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in SmartPlotGenerationTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"Unexpected error generating smart plot: {str(e)}",
                data={
                    "error_type": type(e).__name__,
                    "title": title, 
                    "filename": filename, 
                    "analysis_goal": analysis_goal
                }
            )

class DynamicVisualizationInput(BaseModel):
    json_data: str = Field(description="A JSON string with 'columns' and 'data' keys, representing the data to be visualized.")
    analysis_goal: str = Field(description="A clear, natural language description of what the visualization should show. E.g., 'Compare the performance of each model', 'Show the trend of training cost over time', 'Show the distribution of different model types'.")
    filename: str = Field(description="The filename to save the plot to (e.g., 'visualization.png').")
    chart_type: str | None = Field(default=None, description="Optional. The type of chart to generate (e.g., 'bar', 'line', 'scatter', 'box', 'violin', 'hist'). If not specified, infer from the data and analysis goal.")

class DynamicVisualizationTool(BaseTool):
    """An advanced data visualization tool. It takes structured JSON data (optionally with a 'paper_id' or 'paper_name' column for multi-paper comparison) and a high-level goal,
    and then generates and executes Python code to create the best possible visualization.
    It can create bar charts, line plots, scatter plots, pie charts, violin plots, box plots, histograms, and more."""
    name: str = "dynamic_visualization_tool"
    description: str = (
            "Use this tool to create insightful data visualizations from structured JSON data. "
            "Provide the data and a clear goal for the analysis. Optionally, specify a chart_type ('bar', 'line', 'scatter', 'box', 'violin', 'hist', etc.). "
            "If the data includes a 'paper_id' or 'paper_name' column, the tool will group and compare across papers/models."
        )
    args_schema: Type[DynamicVisualizationInput] = DynamicVisualizationInput
    code_writing_llm: BaseChatModel
    
    def _run(self, json_data: str, analysis_goal: str, filename: str, chart_type: str = "") -> Dict[str, Any]:
        """Generate a visualization from structured data.
        
        Args:
            json_data: JSON string with 'columns' and 'data' keys
            analysis_goal: Natural language description of the visualization goal
            filename: Where to save the generated plot
            chart_type: Optional chart type (e.g., 'bar', 'line', 'scatter')
            
        Returns:
            Dict with status, message, and data containing the path to the saved plot
        """
        try:
            # Parse input JSON
            data = json.loads(json_data)
            if not all(k in data for k in ['columns', 'data']):
                return FailureResponse(
                    message="Invalid data format: expected 'columns' and 'data' keys in JSON",
                    data={"received_keys": list(data.keys())}
                )
                
            # Create DataFrame
            df = pd.DataFrame(data['data'], columns=data['columns'])
            
            # Generate code for visualization
            code = self._generate_plotting_code(df.head().to_string(), analysis_goal, chart_type or None)
            if not code:
                return FailureResponse(
                    message="Failed to generate plotting code",
                    data={"analysis_goal": analysis_goal, "chart_type": chart_type}
                )
                
            # Execute the code to generate the plot
            execution_result = self.execute_python_code(code, df)
            
            if not execution_result.get('success', False):
                return FailureResponse(
                    message=f"Failed to generate visualization: {execution_result.get('error', 'Unknown error')}",
                    data={"error_details": execution_result}
                )
                
            # Ensure the plot was saved
            if not os.path.exists(filename):
                return FailureResponse(
                    message="Plot generation completed but file was not found",
                    data={"expected_path": os.path.abspath(filename)}
                ).model_dump()
                
            return SuccessResponse(
                message=f"Successfully generated visualization: {filename}",
                data={
                    "filename": filename,
                    "absolute_path": os.path.abspath(filename),
                    "analysis_goal": analysis_goal,
                    "chart_type": chart_type or "auto-detected"
                }
            ).model_dump()
            
        except json.JSONDecodeError as e:
            return FailureResponse(
                message="Invalid JSON data provided",
                data={"error": str(e)}
            ).model_dump()
        except Exception as e:
            return FailureResponse(
                message=f"Error generating visualization: {str(e)}",
                data={"error_type": type(e).__name__}
            ).model_dump()

    def _generate_plotting_code(self, df_head: str, analysis_goal: str, chart_type: str | None = None) -> str:
        """Uses an LLM to write robust, safe Python code to generate a plot."""
        chart_type_instruction = f"The user requested a '{chart_type}' plot. Generate code for this plot type." if chart_type else ""
        prompt = f"""
        You are an expert Python data scientist. Your task is to write a snippet of Python code to generate a single, insightful Matplotlib visualization.

        You are given a pandas DataFrame named `df`. Its `df.index` is set to the categorical labels for the x-axis. The first few rows are:
        ---
        {df_head}
        ---
        The user's analysis goal is: "{analysis_goal}"
        {chart_type_instruction}

        **CRITICAL INSTRUCTIONS:**
        1.  Your code will be executed with a pre-existing DataFrame named `df`.
        2.  **DO NOT, under any circumstances, redeclare, redefine, or create the `df` variable.** Do not write `df = ...`, `df = pd.DataFrame(...)`, or any similar statement. Do not include any sample data or import statements.
        3.  **Your code MUST directly use the existing `df` object.**
        4.  You may use advanced plot types such as violin plots, box plots, histograms, or scatter plots if the analysis goal or chart_type suggests it.
        5.  Use the robust syntax: `ax.bar(df.index, df['Column'])`, `ax.scatter(df['ColA'], df['ColB'])`, `ax.boxplot(...)`, `ax.violinplot(...)`, etc. Do NOT use the high-level `df.plot()` wrapper.
        6.  Always create a figure and axes object first: `fig, ax = plt.subplots(figsize=(12, 7))`. Plot on the `ax` object.
        7.  To rotate x-axis labels, use `ax.tick_params(axis='x', rotation=45)`. Do NOT use `plt.xticks(rotation=...)`.
        8.  The code MUST save the plot to a file named 'plot.png' using `fig.savefig('plot.png', bbox_inches='tight')`.
        9.  The code MUST call `plt.close(fig)` at the end to close the specific figure.
        10. Make the plot professional: use `ax.set_title()`, `ax.set_xlabel()`, and `ax.set_ylabel()`.
        11. Return ONLY the Python code snippet. Do not add any other text, markdown wrappers, import statements, or DataFrame creation.
        12. **WARNING:** If you include any import statements, DataFrame creation, or redefinition of `df`, your code will be automatically stripped of those lines and only the plotting code will be executed.

        **Examples:**
        # Violin plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.violinplot([df['Metric1'], df['Metric2']], showmeans=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Metric1', 'Metric2'])
        ax.set_title('Distribution of Metrics')
        fig.savefig('plot.png', bbox_inches='tight')
        plt.close(fig)

        # Box plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot([df['Metric1'], df['Metric2']])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Metric1', 'Metric2'])
        ax.set_title('Boxplot of Metrics')
        fig.savefig('plot.png', bbox_inches='tight')
        plt.close(fig)

        # Histogram:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['Metric1'], bins=10)
        ax.set_title('Histogram of Metric1')
        fig.savefig('plot.png', bbox_inches='tight')
        plt.close(fig)
        """
        
        print("Master Data-Viz tool is writing robust plotting code...")
        code_generation_response = self.code_writing_llm.invoke(prompt)
        # Ensure the return is always a string
        if isinstance(code_generation_response.content, str):
            return code_generation_response.content
        elif isinstance(code_generation_response.content, list):
            return "\n".join([str(x) for x in code_generation_response.content])
        else:
            return str(code_generation_response.content)

    def clean_generated_code(self, code: str) -> str:
        """Removes forbidden lines (imports, df creation, pd/plt redefinitions, empty lines, markdown, non-printable chars) from generated code. Injects 'import numpy as np' if needed."""
        import re
        lines = code.splitlines()
        cleaned = []
        np_needed = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('import'):
                if 'numpy' in stripped and 'as np' in stripped:
                    np_needed = False  # Already present
                continue
            if stripped.startswith('df ='):
                continue
            if stripped.startswith('pd =') or stripped.startswith('plt ='):
                continue
            if 'DataFrame' in stripped and ('=' in stripped):
                continue
            # Remove markdown code block markers
            if stripped.startswith('```') or stripped.endswith('```'):
                continue
            if 'np.' in stripped:
                np_needed = True
            cleaned.append(line)
        cleaned_code = '\n'.join(cleaned)
        # Remove non-printable and non-ASCII characters
        cleaned_code = re.sub(r'[^\x20-\x7E\n\t]', '', cleaned_code)
        # Inject import if needed
        if np_needed and 'import numpy as np' not in cleaned_code:
            cleaned_code = 'import numpy as np\n' + cleaned_code
        return cleaned_code

    def execute_python_code(self, code: str, df: pd.DataFrame) -> str:
        """
        Executes Python code in a controlled environment.
        Captures stdout, stderr, and any exceptions.
        Always provides 'np' (numpy) in the local scope.
        """
     
        buffer = StringIO()
        
  
        local_scope = {
            'df': df,
            'pd': pd,
            'plt': plt,
            'np': np
        }
        
        try:
           
            from contextlib import redirect_stdout
            with redirect_stdout(buffer):
                exec(code, {"__builtins__": __builtins__}, local_scope)
            
        
            stdout = buffer.getvalue()
            return f"Code executed successfully. Captured output:\n{stdout}"
        except Exception as e:
            stderr = buffer.getvalue()
            return f"An error occurred during code execution:\n{e}\nCaptured output:\n{stderr}"

    def _run(self, json_data: str, analysis_goal: str, filename: str, chart_type: str = "") -> str:
        try:
            if not filename.startswith("artifacts/"):
                filename = f"artifacts/{filename}"

            input_obj = json.loads(json_data)
            if input_obj.get("status") != "success":
                reason = input_obj.get("reason", "The previous tool failed to provide data.")
                return f"Error: Cannot visualize because the required data was not provided. Reason: {reason}"

            table_data = input_obj.get("data")
            if not table_data:
                return "Error: Cannot visualize because the previous tool succeeded but returned no data."
            
            df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
            group_col = None
            for col in df.columns:
                if col.lower() in ["paper_id", "paper_name", "model", "model_name"]:
                    group_col = col
                    break
            if group_col:
                if df.index.name != group_col:
                    df.set_index(group_col, inplace=True)
            else:
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].nunique() == len(df):
                        df.set_index(col, inplace=True)
                        break
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            generated_code = self._generate_plotting_code(df.head().to_string(), analysis_goal, chart_type)
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("````")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```",1)[1].split("```",1)[0].strip()
            cleaned_code = self.clean_generated_code(generated_code)
            print(f"--- Cleaned Plotting Code (repr) ---\n{repr(cleaned_code)}\n-----------------------------")
            try:
                execution_result = self.execute_python_code(cleaned_code, df)
            except SyntaxError as e:
                print(f"SyntaxError in full code: {e}. Trying line-by-line execution...")
                lines = cleaned_code.split('\n')
                for i, line in enumerate(lines):
                    try:
                        exec(line, {'df': df, 'plt': __import__('matplotlib.pyplot'), 'pd': __import__('pandas')})
                    except Exception as le:
                        print(f"Line {i+1} failed: {repr(line)}\nError: {le}")
                execution_result = "SyntaxError encountered. See above for details."
            print(f"--- Execution Result ---\n{execution_result}\n------------------------")
            temp_filenames = ['plot.png', filename.split('/')[-1]]
            moved = False
            for temp_file in temp_filenames:
                if os.path.exists(temp_file):
                    if os.path.exists(filename):
                        os.remove(filename)
                    shutil.move(temp_file, filename)
                    moved = True
                    break
            if moved:
                return f"Successfully generated and saved visualization to '{filename}'."
            else:
                return f"Error: Code executed but did not produce a plot file. Details: {execution_result}"
        except Exception as e:
            return f"An error occurred in the tool's main logic: {e}"

class ContradictionInput(BaseModel):
    paper_a_id: str = Field(description="The ID of the first paper for comparison.")
    paper_b_id: str = Field(description="The ID of the second paper for comparison.")
    topic: str = Field(description="The specific topic, dataset, or claim to check for contradictions (e.g., 'performance on the SQuAD 2.0 dataset').")

class ConflictingResultsTool(BaseTool):
    """
    Analyzes two papers to find contradictory or conflicting results on a specific topic.
    This is a powerful tool for critical analysis.
    """
    name: str = "conflicting_results_tool"
    description: str = (
        "Use this tool to find and explain conflicting or contradictory findings between two specific papers. "
        "You must provide the two paper IDs and the topic of conflict."
    )
    args_schema: Type[ContradictionInput] = ContradictionInput
    kb: KnowledgeBase
    extractor: Extractor # Uses an LLM for the analysis

    def _run(self, paper_a_id: str, paper_b_id: str, topic: str) -> Dict[str, Any]:
        """Analyze two papers to find conflicting results on a specific topic.
        
        Args:
            paper_a_id: ID of the first paper
            paper_b_id: ID of the second paper
            topic: The specific topic to analyze for conflicts
            
        Returns:
            Dict containing the analysis results or error information
        """
        # Validate inputs
        if not paper_a_id or not isinstance(paper_a_id, str):
            return FailureResponse(
                message="Invalid paper_a_id: Please provide a valid paper ID for the first paper.",
                data={"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic}
            )
            
        if not paper_b_id or not isinstance(paper_b_id, str):
            return FailureResponse(
                message="Invalid paper_b_id: Please provide a valid paper ID for the second paper.",
                data={"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic}
            )
            
        if not topic or not isinstance(topic, str):
            return FailureResponse(
                message="Invalid topic: Please provide a valid topic to analyze for conflicts.",
                data={"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic}
            )
        
        try:
            print(f"Analyzing conflicting results between papers {paper_a_id} and {paper_b_id} on topic: {topic}")
            
            # Retrieve papers from knowledge base
            result_a = self.kb.collection.get(where={"paper_id": paper_a_id})
            result_b = self.kb.collection.get(where={"paper_id": paper_b_id})
            
            # Process first paper
            documents_a = (result_a['documents'] if result_a and result_a.get('documents') else []) or []
            if not documents_a:
                return FailureResponse(
                    message=f"Could not find paper with ID '{paper_a_id}' in the knowledge base. Please verify the paper ID or add the paper to the knowledge base first.",
                    data={"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic}
                )
                
            text_a = " ".join(str(doc) for doc in documents_a if doc is not None)
            if not text_a.strip():
                return FailureResponse(
                    message=f"Paper with ID '{paper_a_id}' has no text content to analyze.",
                    data={"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic}
                )
                
            metadatas_a = (result_a['metadatas'] if result_a and result_a.get('metadatas') else []) or []
            meta_a = metadatas_a[0] if metadatas_a else {}
            title_a = str(meta_a.get('title')) if meta_a.get('title') is not None else str(paper_a_id)
            
            # Process second paper
            documents_b = (result_b['documents'] if result_b and result_b.get('documents') else []) or []
            if not documents_b:
                return FailureResponse(
                    message=f"Could not find paper with ID '{paper_b_id}' in the knowledge base. Please verify the paper ID or add the paper to the knowledge base first.",
                    data={"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic}
                )
                
            text_b = " ".join(str(doc) for doc in documents_b if doc is not None)
            if not text_b.strip():
                return FailureResponse(
                    message=f"Paper with ID '{paper_b_id}' has no text content to analyze.",
                    data={"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "topic": topic}
                )
                
            metadatas_b = (result_b['metadatas'] if result_b and result_b.get('metadatas') else []) or []
            meta_b = metadatas_b[0] if metadatas_b else {}
            title_b = str(meta_b.get('title')) if meta_b.get('title') is not None else str(paper_b_id)
            
        except (IndexError, TypeError, KeyError) as e:
            logger.error(f"Error retrieving papers for conflict analysis: {e}", exc_info=True)
            return FailureResponse(
                message="Error retrieving papers from the knowledge base. Please verify the paper IDs.",
                data={
                    "error_type": type(e).__name__,
                    "paper_a_id": paper_a_id, 
                    "paper_b_id": paper_b_id, 
                    "topic": topic
                }
            )
        
        # Analyze for contradictions
        try:
            print(f"Performing contradiction analysis on topic: {topic}")
            analysis = self.extractor.find_contradictions(text_a, title_a, text_b, title_b, topic)
            
            return SuccessResponse(
                message=f"Successfully analyzed conflicting results between '{title_a}' and '{title_b}' on topic: {topic}",
                data={
                    "paper_a": {"id": paper_a_id, "title": title_a},
                    "paper_b": {"id": paper_b_id, "title": title_b},
                    "topic": topic,
                    "analysis": analysis
                }
            )
            
        except Exception as e:
            logger.error(f"Error during contradiction analysis: {e}", exc_info=True)
            return FailureResponse(
                message=f"Error analyzing conflicting results: {str(e)}",
                data={
                    "error_type": type(e).__name__,
                    "paper_a_id": paper_a_id, 
                    "paper_b_id": paper_b_id, 
                    "topic": topic
                }
            )

    topic: str = Field(description="The central research topic to analyze for gaps.")
    num_papers_to_analyze: int = Field(default=5, description="The number of top papers on the topic to include in the analysis.")

class LiteratureGapTool(BaseTool):
    """Analyzes a collection of top papers on a given topic to identify potential
    gaps in the literature and suggest future research directions.
    
    This tool performs a comprehensive analysis of the current state of research
    on a given topic by examining multiple papers. It identifies trends, common
    methodologies, and most importantly, areas that have not been sufficiently
    explored in the existing literature.
    """
    name: str = "literature_gap_tool"
    description: str = (
        "A powerful research tool. Use this to analyze the current state of a research field "
        "and get suggestions for novel future work. Operates on a collection of papers. "
        "Provide a topic and optionally the number of top papers to analyze (default: 5)."
    )
    args_schema: Type[LiteratureGapInput] = LiteratureGapInput
    
    kb: KnowledgeBase
    extractor: Extractor
    llm: BaseChatModel
    max_papers: int = 10  # Maximum number of papers to analyze
    min_papers: int = 2   # Minimum number of papers needed for meaningful analysis

    def _run(self, paper_ids: List[str], query: Optional[str] = None, num_papers_to_analyze: int = 5) -> Dict[str, Any]:
        """Analyze papers on a topic to find research gaps and opportunities.
        
        Args:
            paper_ids: List of paper IDs to analyze for gaps
            query: Optional query to focus the gap analysis
            num_papers_to_analyze: Number of top papers to include in the analysis
            
        Returns:
            Dict containing the analysis results or error information
        """
        # Validate input
        num_papers_to_analyze = min(max(self.min_papers, num_papers_to_analyze), self.max_papers)
        
        try:
            # If paper_ids are provided, retrieve those papers directly
            if paper_ids:
                # Get papers by IDs from knowledge base
                papers_data = []
                for paper_id in paper_ids[:num_papers_to_analyze]:  # Limit to num_papers_to_analyze
                    paper = self.kb.get_paper_by_id(paper_id)
                    if paper:
                        papers_data.append({
                            "paper_id": paper_id,
                            "title": paper.get('title', f'Paper {paper_id}'),
                            "content": paper.get('content', '')[:2000],  # Limit content length
                            "metadata": paper.get('metadata', {})
                        })
                
                if not papers_data:
                    return FailureResponse(
                        message="No valid papers found for the provided IDs.",
                        data={"paper_ids": paper_ids}
                    )
            else:
                # Search for relevant papers using query
                search_results = self.kb.search(query=query or "research", n_results=num_papers_to_analyze)
                
                if not search_results or not search_results.get('documents'):
                    return FailureResponse(
                        message=f"No papers found on topic: {query}",
                        data={"topic": query, "num_papers_searched": num_papers_to_analyze}
                    )
                
                # Check if we have enough papers for meaningful analysis
                if len(search_results['documents']) < self.min_papers:
                    return FailureResponse(
                        message=f"Found only {len(search_results['documents'])} papers, but need at least {self.min_papers} for analysis.",
                        data={
                            "topic": query,
                            "papers_found": len(search_results['documents']),
                            "min_papers_required": self.min_papers
                        }
                    )
                
                # Prepare paper data for analysis
                papers_data = []
                for i, doc in enumerate(search_results['documents']):
                    papers_data.append({
                        "paper_id": search_results['ids'][i],
                        "title": doc.get('title', f'Paper {i+1}'),
                        "content": doc.get('content', '')[:2000],  # Limit content length
                        "metadata": search_results['metadatas'][i] if 'metadatas' in search_results and i < len(search_results['metadatas']) else {}
                    })
            
            # Generate structured analysis
            analysis = self._analyze_papers(query or "research", papers_data)
            
            return SuccessResponse(
                message=f"Analysis completed for {len(papers_data)} papers on topic: {query}",
                data={
                    "topic": query,
                    "papers_analyzed": [{"id": p["paper_id"], "title": p["title"]} for p in papers_data],
                    "analysis": analysis,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            return FailureResponse(
                message=f"Error analyzing literature gaps: {str(e)}",
                data={
                    "error_type": type(e).__name__,
                    "topic": topic,
                    "num_papers_requested": num_papers_to_analyze
                }
            )
    
    def _analyze_papers(self, topic: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform the actual analysis of papers to identify gaps."""
        # Prepare context for the LLM
        papers_context = "\n\n".join([
            f"PAPER {i+1}: {p['title']}\n"
            f"Content: {p['content'][:1500]}..."
            for i, p in enumerate(papers)
        ])
        
        # Generate analysis prompt
        prompt = f"""
        You are a research analyst examining the current state of knowledge on: {topic}
        
        Below are {len(papers)} research papers on this topic:
        
        {papers_context}
        
        Please analyze these papers and provide a structured response with the following sections:
        
        1. CURRENT STATE OF RESEARCH:
           - Main themes and approaches
           - Common methodologies used
           - Key findings and conclusions
           
        2. IDENTIFIED GAPS:
           - Limitations in current research
           - Contradictions between studies
           - Underexplored aspects of the topic
           - Methodological weaknesses
           
        3. FUTURE RESEARCH DIRECTIONS:
           - Specific questions that remain unanswered
           - Novel approaches that could be taken
           - Potential interdisciplinary connections
           - Methodological improvements
           
        4. MOST PROMISING OPPORTUNITIES:
           - Rank 3-5 most promising research directions
           - For each, explain why it's valuable and feasible
        
        Be specific, critical, and creative in your analysis. Support your points with evidence from the papers.
        """
        
        try:
            # Get analysis from LLM
            response = self.llm.invoke(prompt)
            
            # Parse the response into a structured format
            return self._parse_analysis_response(response.content)
            
        except Exception as e:
            # If parsing fails, return the raw response
            return {
                "raw_analysis": str(response.content) if 'response' in locals() else str(e),
                "parse_error": True if 'response' in locals() else False
            }
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM's response into a structured format."""
        # This is a simplified parser - in practice, you might want to use more sophisticated parsing
        # or ask the LLM to return structured JSON directly
        sections = {
            "current_state": "",
            "identified_gaps": "",
            "future_directions": "",
            "promising_opportunities": []
        }
        
        current_section = None
        opportunities = []
        
        for line in response_text.split('\n'):
            line = line.strip()
            
            # Detect section headers
            if line.upper().startswith(('1. CURRENT', 'CURRENT STATE')):
                current_section = "current_state"
                continue
            elif line.upper().startswith(('2. IDENTIFIED', 'IDENTIFIED GAPS')):
                current_section = "identified_gaps"
                continue
            elif line.upper().startswith(('3. FUTURE', 'FUTURE RESEARCH')):
                current_section = "future_directions"
                continue
            elif line.upper().startswith(('4. MOST', 'MOST PROMISING')):
                current_section = "promising_opportunities"
                continue
                
            # Add content to current section
            if current_section and line:
                if current_section == "promising_opportunities":
                    # Try to extract ranked opportunities
                    if line.strip().startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.\s', line):
                        opportunities.append(line.lstrip('-*• 1234567890. '))
                else:
                    sections[current_section] += line + "\n"
        
        # Clean up sections
        for key in sections:
            if isinstance(sections[key], str):
                sections[key] = sections[key].strip()
        
        # Add opportunities if we found any
        if opportunities:
            sections["promising_opportunities"] = [
                {"rank": i+1, "description": opp}
                for i, opp in enumerate(opportunities[:5])  # Limit to top 5
            ]
        
        return sections

class CsvExportInput(BaseModel):
    """Input model for the CSV Export Tool."""
    json_data: str = Field(description="A JSON string with 'columns' and 'data' keys, representing the table data.")
    filename: str = Field(description="The filename to save the CSV to (e.g., 'performance_data.csv').")

class DataToCsvTool(BaseTool):
    """A utility tool to save structured JSON data into a CSV file.
    This is useful for exporting extracted tables for further analysis in other programs."""
    name: str = "data_to_csv_tool"
    description: str = (
            "Use this tool to save structured table data (in JSON format) to a CSV file. "
            "This is the final step after you have extracted a table."
        )
    args_schema: Type[CsvExportInput] = CsvExportInput

    def _run(self, json_data: str, filename: str) -> Dict[str, Any]:
        """Save structured JSON data to a CSV file in the artifacts directory.
        
        Args:
            json_data: A JSON string with 'columns' and 'data' keys
            filename: The output filename for the CSV (will be saved in ARTIFACTS_DIR)
            
        Returns:
            Dict with status, message, and data containing file info
        """
        try:
            # Get the artifacts directory path from the environment or use a default
            artifacts_dir = os.getenv('ARTIFACTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Ensure filename has .csv extension
            if not filename.lower().endswith('.csv'):
                filename = f"{filename}.csv"
                
            # Create full path to save the file
            filepath = os.path.join(artifacts_dir, filename)
            
            data = json.loads(json_data)
            if not all(k in data for k in ['columns', 'data']):
                return FailureResponse(
                    message="Invalid data format: expected 'columns' and 'data' keys in JSON",
                    data={"received_keys": list(data.keys())}
                )

            # Create and save DataFrame
            df = pd.DataFrame(data['data'], columns=data['columns'])
            df.to_csv(filepath, index=False)
            
            # Verify file was created
            if not os.path.exists(filepath):
                return FailureResponse(
                    message="CSV file was not created successfully",
                    data={"expected_path": filepath}
                ).model_dump()
                
            # Create a web-accessible URL for the file
            web_path = f"/artifacts/{filename}"
                
            return SuccessResponse(
                message=f"Successfully saved data to {filename}",
                data={
                    "filename": filename,
                    "absolute_path": filepath,
                    "web_path": web_path,
                    "num_rows": len(df),
                    "columns": list(df.columns),
                    "file_size_bytes": os.path.getsize(filepath)
                }
            ).model_dump()
            
        except json.JSONDecodeError as e:
            return FailureResponse(
                message="Invalid JSON data provided",
                data={"error": str(e)}
            )
        except PermissionError as e:
            return FailureResponse(
                message="Permission denied when trying to write CSV file",
                data={"error": str(e), "filename": filename}
            ).model_dump()
        except Exception as e:
            return FailureResponse(
                message=f"Error saving to CSV: {str(e)}",
                data={"error_type": type(e).__name__, "filename": filename}
            ).model_dump()

    async def _arun(self, json_data: str, filename: str) -> str:
        raise NotImplementedError("This tool does not support async yet.")


class ArchitectureDiagramInput(BaseModel):
    diagram_code: str = Field(description="A Mermaid or Graphviz DOT string describing the architecture.")
    filename: str = Field(description="The filename to save the diagram to (e.g., 'transformer_architecture.png').")
    engine: str = Field(default="mermaid", description="Diagram engine: 'mermaid' or 'graphviz'.")

class ArchitectureDiagramTool(BaseTool):
    name: str = "architecture_diagram_tool"
    description: str = (
        "Generate a model architecture diagram from Mermaid or Graphviz code and save it as a PNG in the artifacts directory. "
        "Input: diagram_code (Mermaid or DOT), filename, engine ('mermaid' or 'graphviz')."
    )
    args_schema: Type[ArchitectureDiagramInput] = ArchitectureDiagramInput

    def _sanitize_mermaid_code(self, diagram_code: str) -> str:
        """Sanitize Mermaid code to prevent syntax errors."""
        import re
        
        if "```mermaid" in diagram_code:
            diagram_code = diagram_code.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in diagram_code:
            diagram_code = diagram_code.split("```")[1].split("```")[0].strip()
        
        diagram_code = re.sub(r'([A-Za-z0-9_]+)\(([^)]*[()][^)]*)\)', r'\1["\2"]', diagram_code)
        
        diagram_code = re.sub(r'(\w+)\s*-->\s*(\w+)', r'\1 --> \2', diagram_code)
        
        diagram_code = diagram_code.replace('(', '[').replace(')', ']')
        
        return diagram_code

    def _run(self, diagram_code: str, filename: str, engine: str = "auto") -> Dict[str, Any]:
        """Generate a diagram from code and save it to a file.
        
        Args:
            diagram_code: The diagram code (Mermaid or Graphviz DOT format)
            filename: The output filename
            engine: The rendering engine to use ("auto", "mermaid", or "graphviz")
            
        Returns:
            Dict with success/failure status and detailed information
        """
        import os, logging, subprocess, tempfile
        
        # Validate inputs
        if not diagram_code or not isinstance(diagram_code, str):
            return FailureResponse(
                message="Invalid diagram_code: Please provide valid diagram code.",
                data={"filename": filename, "engine": engine}
            )
            
        if not filename or not isinstance(filename, str):
            return FailureResponse(
                message="Invalid filename: Please provide a valid filename.",
                data={"filename": filename, "engine": engine}
            )
        
        try:
            # Ensure artifacts directory exists
            artifacts_dir = os.getenv('ARTIFACTS_DIR', 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Ensure filename is in artifacts directory
            if not filename.startswith(artifacts_dir + os.sep):
                filename = os.path.join(artifacts_dir, filename)
            
            # Sanitize the diagram code
            sanitized_code = self._sanitize_mermaid_code(diagram_code)
            
            # Determine diagram type
            is_dot = sanitized_code.strip().lower().startswith("digraph") or "->" in sanitized_code or "graph" in sanitized_code.lower()
            is_mermaid = any(keyword in sanitized_code.lower() for keyword in ["graph lr", "graph td", "sequenceDiagram", "classDiagram", "stateDiagram", "erDiagram"])
            
            print(f"Generating diagram with engine: {engine}")
            print(f"Detected diagram types - Mermaid: {is_mermaid}, Graphviz DOT: {is_dot}")
            
            # Try Graphviz first if specified or if it's a DOT diagram
            if engine == "graphviz" or (engine == "auto" and is_dot and not is_mermaid):
                try:
                    import graphviz
                    print("Attempting Graphviz rendering")
                    dot = graphviz.Source(sanitized_code)
                    dot.format = "png"
                    output_path = dot.render(filename=filename, cleanup=True)
                    return SuccessResponse(
                        message=f"Graphviz diagram saved successfully",
                        data={
                            "output_path": output_path,
                            "filename": filename + ".png",
                            "engine_used": "graphviz",
                            "diagram_type": "dot"
                        }
                    )
                except ImportError:
                    logging.warning("Graphviz library not installed")
                    print("Graphviz library not installed, trying Mermaid")
                except Exception as e:
                    logging.error(f"Graphviz error: {e}")
                    print(f"Graphviz rendering failed: {e}")
                    
            # Try Mermaid if specified or as fallback
            if engine == "mermaid" or (engine == "auto" and (is_mermaid or is_dot)):
                try:
                    print("Attempting Mermaid rendering")
                    # Create temporary file with diagram code
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as tmpfile:
                        tmpfile.write(sanitized_code)
                        tmpfile_path = tmpfile.name
                    
                    # Ensure puppeteer config exists
                    puppeteer_config_path = os.path.join(os.path.dirname(__file__), 'puppeteer-config.json')
                    if not os.path.exists(puppeteer_config_path):
                        with open(puppeteer_config_path, 'w') as f:
                            f.write('{\n  "args": ["--no-sandbox", "--disable-setuid-sandbox"]\n}')
                    
                    # Run mermaid CLI
                    cmd = [
                        "mmdc", "-i", tmpfile_path, "-o", filename,
                        "--puppeteerConfigFile", puppeteer_config_path
                    ]
                    
                    print(f"Running command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    os.remove(tmpfile_path)
                    
                    if result.returncode == 0:
                        return SuccessResponse(
                            message="Mermaid diagram saved successfully",
                            data={
                                "output_path": filename,
                                "filename": filename,
                                "engine_used": "mermaid",
                                "diagram_type": "mermaid"
                            }
                        )
                    else:
                        error_msg = f"Mermaid CLI error: {result.stderr}"
                        logging.error(error_msg)
                        print(error_msg)
                        
                        # Try simplified diagram as fallback
                        print("Trying simplified diagram as fallback")
                        simple_code = "graph LR\nA[Start] --> B[Process] --> C[End]"
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as tmpfile2:
                            tmpfile2.write(simple_code)
                            tmpfile_path2 = tmpfile2.name
                        
                        cmd2 = [
                            "mmdc", "-i", tmpfile_path2, "-o", filename,
                            "--puppeteerConfigFile", puppeteer_config_path
                        ]
                        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
                        os.remove(tmpfile_path2)
                        
                        if result2.returncode == 0:
                            return SuccessResponse(
                                message="Mermaid diagram (simplified) saved successfully",
                                data={
                                    "output_path": filename,
                                    "filename": filename,
                                    "engine_used": "mermaid",
                                    "diagram_type": "mermaid",
                                    "simplified": True
                                }
                            )
                        else:
                            raise Exception(f"Original error: {result.stderr}. Simplified diagram also failed: {result2.stderr}")
                
                except FileNotFoundError:
                    error_msg = "Mermaid CLI (mmdc) is not installed. Please install it with 'npm install -g @mermaid-js/mermaid-cli'."
                    print(error_msg)
                    return FailureResponse(
                        message=error_msg,
                        data={
                            "filename": filename,
                            "engine": engine,
                            "missing_dependency": "mmdc"
                        }
                    )
                except subprocess.TimeoutExpired:
                    error_msg = "Mermaid diagram generation timed out. Please try a simpler diagram."
                    print(error_msg)
                    return FailureResponse(
                        message=error_msg,
                        data={
                            "filename": filename,
                            "engine": engine,
                            "timeout": True
                        }
                    )
                except Exception as e:
                    error_msg = f"Mermaid rendering failed: {str(e)}"
                    logging.error(error_msg)
                    print(error_msg)
                    
                    # Try Graphviz as final fallback
                    try:
                        import graphviz
                        print("Trying Graphviz as final fallback")
                        dot = graphviz.Source(sanitized_code)
                        dot.format = "png"
                        output_path = dot.render(filename=filename, cleanup=True)
                        return SuccessResponse(
                            message=f"Mermaid failed but Graphviz fallback succeeded",
                            data={
                                "output_path": output_path,
                                "filename": filename + ".png",
                                "engine_used": "graphviz",
                                "diagram_type": "dot",
                                "original_error": str(e)
                            }
                        )
                    except Exception as e2:
                        # Both failed, save as Markdown
                        md_fallback = filename.rsplit('.', 1)[0] + "_diagram.md"
                        with open(md_fallback, 'w') as f:
                            f.write(f"# Diagram Code (Fallback)\n\n```mermaid\n")
                            f.write(sanitized_code)
                            f.write("\n```")
                        return FailureResponse(
                            message=f"Both Mermaid and Graphviz failed. Diagram code saved as Markdown to {md_fallback}.",
                            data={
                                "filename": md_fallback,
                                "engine": engine,
                                "mermaid_error": str(e),
                                "graphviz_error": str(e2)
                            }
                        )
            
            # If we get here, no engine worked
            return FailureResponse(
                message="Unable to generate diagram. Please check the diagram code format and ensure either Mermaid CLI or Graphviz is installed.",
                data={
                    "filename": filename,
                    "engine": engine,
                    "diagram_code_preview": sanitized_code[:200] + "..." if len(sanitized_code) > 200 else sanitized_code
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ArchitectureDiagramTool: {e}", exc_info=True)
            return FailureResponse(
                message=f"Error generating diagram: {str(e)}",
                data={
                    "filename": filename,
                    "engine": engine,
                    "error_type": type(e).__name__
                }
            )