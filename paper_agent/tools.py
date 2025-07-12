from langchain_neo4j import Neo4jGraph 
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from typing import Type, Any
from pydantic import BaseModel, Field

import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import json 
import numpy as np
import os
from langchain_core.language_models.chat_models import BaseChatModel 

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
    name: str = "scientific_paper_knowledge_base_tool" 
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

# --- THE NEW, SUPERIOR FINDER TOOL ---
class PaperFinderInput(BaseModel):
    """Input model for the Graph Paper Finder Tool."""
    query: str = Field(description="A query to find relevant papers, usually the title of the paper.")

class GraphPaperFinderTool(BaseTool):
    """
    Finds a specific paper in the knowledge graph by its title and returns ALL its metadata,
    including its exact 'paper_id' (which could be an arXiv ID or a local path).
    """
    name: str = "graph_paper_finder_tool"
    description: str = (
        "Use this tool as the VERY FIRST STEP to find a paper's 'paper_id' and other metadata. "
        "It performs an exact, case-insensitive search on the paper's title in the knowledge graph. "
        "This is the most reliable way to find a specific paper that is already in the database."
    )
    args_schema: Type[PaperFinderInput] = PaperFinderInput

    graph: Neo4jGraph

    def _run(self, query: str) -> str:
        """Use the tool."""
        cypher = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($query)
        RETURN p.id as paper_id, p.title as title
        LIMIT 5
        """
        try:
            result = self.graph.query(cypher, params={"query": query})
            if not result:
                return f"No paper found in the Knowledge Graph with a title containing '{query}'."
            return str(result)
        except Exception as e:
            return f"Error executing graph query: {e}"

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

    def _run(self, paper_id: str, topic_of_interest: str) -> str:
        # Fetch the paper text from the knowledge base
        results = self.kb.collection.get(where={"paper_id": paper_id})
        if not results or not results['documents']:
            return f"Error: Could not find paper with ID {paper_id}."
        
        full_text = " ".join(results['documents'])
        metadatas = results.get('metadatas')
        if metadatas and len(metadatas) > 0:
            raw_title = metadatas[0].get('title', 'Unknown Title')
            title = str(raw_title) if raw_title is not None else 'Unknown Title'
        else:
            title = 'Unknown Title'

        # --- NEW JSON OUTPUT LOGIC ---
        # We call a modified extractor method
        json_table = self.extractor.extract_table_as_json(full_text, title, topic_of_interest)
        return json_table # Return the JSON string directly

class RelationshipInput(BaseModel):
    paper_a_title: str = Field(description="The title of the first paper.")
    paper_b_title: str = Field(description="The title of the second paper.")

class RelationshipAnalysisTool(BaseTool):
    name: str = "relationship_analysis_tool"
    description: str = (
        "Use this tool to explain the relationship between two papers. "
        "It queries the knowledge graph to find citation paths and other connections."
    )
    args_schema: Type[RelationshipInput] = RelationshipInput
    graph: Neo4jGraph
    llm: Any  # Accept any LLM with an .invoke method

    def _run(self, paper_a_title: str, paper_b_title: str) -> str:
        # A Cypher query to find the shortest path between two papers
        cypher_query = f"""
        MATCH (p1:Paper), (p2:Paper), path = shortestPath((p1)-[:CITES*]-(p2))
        WHERE toLower(p1.title) CONTAINS toLower('{paper_a_title}')
        AND toLower(p2.title) CONTAINS toLower('{paper_b_title}')
        RETURN path
        """
        print(f"Running relationship query: {cypher_query}")
        graph_data = str(self.graph.query(cypher_query))

        if not graph_data or graph_data == '[]':
            return "No direct citation path found between the two papers in the knowledge graph."

        # Use an LLM to explain the path
        synthesis_prompt = f"""
        A user wants to know the relationship between '{paper_a_title}' and '{paper_b_title}'.
        A knowledge graph query returned the following connection path:
        ---
        {graph_data}
        ---
        Based on this data, please explain the relationship in a clear, concise paragraph. For example, you might say 'Paper B cites Paper A' or 'Paper A and Paper B both cite a common paper C'.
        """
        response = self.llm.invoke(synthesis_prompt)
        return response.content

class CitationAnalysisInput(BaseModel):
    analysis_type: str = Field(description="The type of citation analysis to perform. Options: 'most_cited', 'hottest_papers'.")
    limit: int = Field(default=5, description="The number of papers to return.")

class CitationAnalysisTool(BaseTool):
    """
    Performs citation analysis on the knowledge graph. Can find the most cited papers
    or the papers with the most outgoing citations ('hottest papers' or foundational work).
    """
    name: str = "citation_analysis_tool"
    description: str = (
        "Use this to find influential papers. 'most_cited' finds papers that many others reference. "
        "'hottest_papers' finds papers that reference many others."
    )
    args_schema: Type[CitationAnalysisInput] = CitationAnalysisInput
    graph: Neo4jGraph

    def _run(self, analysis_type: str, limit: int = 5) -> str:
        if analysis_type == 'most_cited':
            # This query counts incoming CITES relationships
            query = f"""
            MATCH (p:Paper)<-[r:CITES]-()
            RETURN p.title AS paper, count(r) AS citations
            ORDER BY citations DESC
            LIMIT {limit}
            """
        elif analysis_type == 'hottest_papers':
            # This query counts outgoing CITES relationships
            query = f"""
            MATCH (p:Paper)-[r:CITES]->()
            RETURN p.title AS paper, count(r) AS citations_made
            ORDER BY citations_made DESC
            LIMIT {limit}
            """
        else:
            return "Error: Invalid analysis_type. Must be 'most_cited' or 'hottest_papers'."

        print(f"Running citation analysis query: {analysis_type}")
        try:
            result = self.graph.query(query)
            if not result:
                return "No citation data found in the graph to perform analysis."
            return str(result)
        except Exception as e:
            return f"Error during citation analysis: {e}"

class KeywordExtractionInput(BaseModel):
    paper_id: str = Field(description="The ID of the paper from which to extract keywords.")
    num_keywords: int = Field(default=10, description="The number of keywords to extract.")

class KeywordExtractionTool(BaseTool):
    """Extracts the most important keywords or concepts from a given paper."""
    name: str = "keyword_extraction_tool"
    description: str = "Use this to identify the main topics, concepts, or keywords of a specific paper."
    args_schema: Type[KeywordExtractionInput] = KeywordExtractionInput
    kb: KnowledgeBase
    extractor: Extractor # Uses an LLM to find the keywords

    def _run(self, paper_id: str, num_keywords: int = 10) -> str:
        results = self.kb.collection.get(where={"paper_id": paper_id})
        if not results or not results['documents']:
            return f"Error: Could not find paper with ID {paper_id}."
        
        full_text = " ".join(results['documents'])
        # Call a new method on our extractor
        keywords = self.extractor.extract_keywords(full_text, num_keywords)
        return f"The key concepts are: {', '.join(keywords)}"

class PlottingInput(BaseModel):
    # The input is now a JSON string, not markdown
    json_data: str = Field(description="A JSON string containing the table data, with 'columns' and 'data' keys.")
    chart_type: str = Field(description="The type of chart to generate (e.g., 'bar', 'line').")
    title: str = Field(description="The title for the chart.")
    filename: str = Field(description="The filename to save the plot to (e.g., 'performance_chart.png').")

class PlotGenerationTool(BaseTool):
    name: str = "plot_generation_tool"
    description: str = "Use this tool to generate a plot from structured JSON data and save it as an image file."
    args_schema: Type[PlottingInput] = PlottingInput

    def _run(self, json_data: str, chart_type: str, title: str, filename: str) -> str:
        try:
            # Parse the structured JSON input
            data = json.loads(json_data)
            df = pd.DataFrame(data['data'], columns=data['columns'])

            # Plotting logic is now much cleaner
            # Assume the first column is the x-axis (index)
            df.set_index(df.columns[0], inplace=True)
            
            ax = df.plot(kind=chart_type, title=title, figsize=(10, 6))
            plt.ylabel("Value")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return f"Successfully generated and saved plot to '{filename}'."
        except Exception as e:
            return f"Error generating plot: {e}"

class SmartPlottingInput(BaseModel):
    json_data: str = Field(description="A JSON string with 'columns' and 'data' keys.")
    # The agent now decides the best chart type!
    title: str = Field(description="The title for the chart.")
    filename: str = Field(description="The filename to save the plot to (e.g., 'performance_chart.png').")
    # A new field to guide the LLM
    analysis_goal: str = Field(description="A short sentence describing what the plot should show or compare. E.g., 'Compare the error metrics (RMSE, MAE) of the models'.")

class SmartPlotGenerationTool(BaseTool):
    name: str = "smart_plot_generation_tool"
    description: str = (
        "An intelligent plotting tool. It takes structured JSON data and an analysis goal, "
        "and automatically chooses the best way to visualize the data to meet that goal. "
        "Use this to create insightful charts from extracted tables."
    )
    args_schema: Type[SmartPlottingInput] = SmartPlottingInput

    def _run(self, json_data: str, title: str, filename: str, analysis_goal: str) -> str:
        try:
            data = json.loads(json_data)
            df = pd.DataFrame(data['data'], columns=data['columns'])
            df.set_index(df.columns[0], inplace=True)
            
            # --- THE NEW INTELLIGENCE ---
            # Heuristic: If one column's values are much larger than the others,
            # plot it on a secondary y-axis.
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 1:
                max_vals = df[numeric_cols].max()
                # If the max of one column is > 10x the max of another...
                if max_vals.max() / max_vals.min() > 10:
                    # Plot the largest column on a secondary axis
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

                else: # All values are on a similar scale
                    df.plot(kind='bar', title=title, figsize=(10, 6), rot=45)
            else: # Only one numeric column
                df.plot(kind='bar', title=title, figsize=(10, 6), rot=45)
            
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return f"Successfully generated an insightful plot and saved it to '{filename}'."

        except Exception as e:
            return f"Error generating smart plot: {e}"

class DynamicVisualizationInput(BaseModel):
    json_data: str = Field(description="A JSON string with 'columns' and 'data' keys, representing the data to be visualized.")
    analysis_goal: str = Field(description="A clear, natural language description of what the visualization should show. E.g., 'Compare the performance of each model', 'Show the trend of training cost over time', 'Show the distribution of different model types'.")
    filename: str = Field(description="The filename to save the plot to (e.g., 'visualization.png').")
    chart_type: str | None = Field(default=None, description="Optional. The type of chart to generate (e.g., 'bar', 'line', 'scatter', 'box', 'violin', 'hist'). If not specified, infer from the data and analysis goal.")

class DynamicVisualizationTool(BaseTool):
    """
    An advanced data visualization tool. It takes structured JSON data and a high-level goal,
    and then generates and executes Python code to create the best possible visualization.
    It can create bar charts, line plots, scatter plots, pie charts, violin plots, box plots, histograms, and more.
    """
    name: str = "dynamic_visualization_tool"
    description: str = (
        "Use this tool to create insightful data visualizations from structured JSON data. "
        "Provide the data and a clear goal for the analysis. Optionally, specify a chart_type ('bar', 'line', 'scatter', 'box', 'violin', 'hist', etc.)."
    )
    args_schema: Type[DynamicVisualizationInput] = DynamicVisualizationInput
    
    # This tool needs its own LLM to write the plotting code
    code_writing_llm: BaseChatModel

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
        """Removes forbidden lines (imports, df creation, pd/plt redefinitions, empty lines, markdown, non-printable chars) from generated code."""
        import re
        lines = code.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('import'):
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
            cleaned.append(line)
        cleaned_code = '\n'.join(cleaned)
        # Remove non-printable and non-ASCII characters
        cleaned_code = re.sub(r'[^\x20-\x7E\n\t]', '', cleaned_code)
        return cleaned_code

    # --- NEW HELPER FUNCTION for safe code execution ---
    def execute_python_code(self, code: str, df: pd.DataFrame) -> str:
        """
        Executes Python code in a controlled environment.
        Captures stdout, stderr, and any exceptions.
        """
     
        buffer = StringIO()
        
  
        local_scope = {
            'df': df,
            'pd': pd,
            'plt': plt
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
            data = json.loads(json_data)
            df = pd.DataFrame(data['data'], columns=data['columns'])
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
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            cleaned_code = self.clean_generated_code(generated_code)
            print(f"--- Cleaned Plotting Code (repr) ---\n{repr(cleaned_code)}\n-----------------------------")

            # Try executing the whole code, if syntax error, try line by line
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
            
            temp_filename = 'plot.png'
            if os.path.exists(temp_filename):
                os.rename(temp_filename, filename)
                return f"Successfully generated and saved visualization to '{filename}'."
            else:
                return f"Error: Code executed but did not produce '{temp_filename}'. Details: {execution_result}"

        except Exception as e:
            return f"An error occurred in the tool's main logic: {e}"