from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Any
from langchain_neo4j import Neo4jGraph 

from .knowledge_base import KnowledgeBase
from .ingestor import Ingestor
from .tools import (
    GetPaperMetadataByTitleTool, 
    AnswerFromPapersTool,   
    ArxivSearchTool,
    ArxivFetchTool,
    PaperSummarizationTool,
    web_search_tool,
    GraphQueryTool,         
    TableExtractionTool,   
    RelationshipAnalysisTool,
    DynamicVisualizationTool,
    ArchitectureDiagramTool,  
    CitationAnalysisTool,   
    KeywordExtractionTool,  
    ConflictingResultsTool,
    LiteratureGapTool,
    DataToCsvTool 
)
from .extractor import Extractor
from langchain_google_genai import ChatGoogleGenerativeAI 
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models import ChatOllama

ROBUST_JSON_PROMPT_TEMPLATE = """
You are a helpful scientific research assistant. Your goal is to answer the user's question by thinking step-by-step.
You must respond with a JSON object containing either an 'action' and 'action_input' key, or a 'action' and 'action_input' key with a final answer.

You have access to the following tools:
{tools}

The available tool names are: {tool_names}

Here are some important rules to follow:
1.  **Analyze Observations:** After using a tool, check the 'status' of the JSON observation. If `status: "failure"`, you MUST NOT repeat the same action. Analyze the 'reason' and try a different approach.
2.  **Paper Loading:** If the user asks you to find or load papers, your job is done once you have successfully used the `arxiv_paper_search_and_load_tool`. Your final answer should just confirm the papers were loaded.
3.  **Getting Paper IDs:** The `get_paper_metadata_by_title` tool returns a JSON object like `{"status": "success", "data": [{"paper_id": "...", "title": "...", "authors": [...]}]}`. To get the ID for a subsequent tool (like `paper_summarization_tool`), you must extract it from the **first item in the `data` list**. For example: `observation['data'][0]['paper_id']`.
4.  **Data Flow:** The `table_extraction_tool` outputs structured JSON with a `status` and `data` key. The `dynamic_visualization_tool` and `data_to_csv_tool` require this entire JSON object as their `json_data` input.
5.  **Cypher Queries:** When matching on string properties like 'title' in a Cypher query, ALWAYS use `toLower()` and `CONTAINS`. For example: `WHERE toLower(p.title) CONTAINS 'attention'`.
6.  **Tool Arguments:** ALWAYS provide ALL required arguments with the EXACT argument names specified in the tool's description.
7.  **Avoid Loops:** If a tool fails, try a different tool or conclude that you cannot answer the question. Do not get stuck.
8.  **Final Answer:** Once you have a satisfactory answer, you MUST provide the final answer. If you cannot find an answer after 3-4 steps, provide a best-effort summary and state what you couldn't find.
9.  **Author/Collaboration Questions:** For questions like 'Who wrote X?' or 'Who collaborated with Y?', always use the `graph_query_tool` with a Cypher query. Do not use `get_paper_metadata_by_title`.

Use the following format for each step:

Thought: The user's question is about X. I should use tool Y to find the answer.
Action:
```json
{{
  "action": "tool_name",
  "action_input": {{"argument_name": "argument value"}}
}}
```

# Examples (using the correct, current tool names):

## Get Paper Metadata Example:
```json
{{
  "action": "get_paper_metadata_by_title",
  "action_input": {{"query": "Attention is All You Need"}}
}}
```

## Answering a Question from Papers (RAG) Example:
```json
{{
  "action": "answer_from_papers",
  "action_input": {{"question": "What is the Transformer architecture?"}}
}}
```

## ArXiv Search Example:
```json
{{
  "action": "arxiv_paper_search_and_load_tool",
  "action_input": {{"query": "Mixture of Experts", "max_results": 5}}
}}
```

## Graph Query (for Authors/Relationships) Example:
```json
{{
  "action": "graph_query_tool",
  "action_input": {{"query": "MATCH (p:Paper)<-[:AUTHORED]-(a:Author) WHERE toLower(p.title) CONTAINS 'attention' RETURN a.name AS author LIMIT 10"}}
}}
```

## Full Paper Summarization Chain Example:
Thought: The user wants a summary of a specific paper. First, I need to get its ID using `get_paper_metadata_by_title`.
Action:
```json
{{
  "action": "get_paper_metadata_by_title",
  "action_input": {{"query": "Attention is All You Need"}}
}}
```
Observation: {{"status": "success", "data": [{{"paper_id": "http://arxiv.org/abs/1706.03762v5", "title": "Attention Is All You Need", "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"]}}]}}
Thought: The tool succeeded and I have the paper_id: "http://arxiv.org/abs/1706.03762v5". Now I can use `paper_summarization_tool`.
Action:
```json
{{
  "action": "paper_summarization_tool",
  "action_input": {{"paper_id": "http://arxiv.org/abs/1706.03762v5"}}
}}
```
Observation: "This paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms..."
Thought: I have gathered enough information. I will now provide the final answer.
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "Summary of 'Attention Is All You Need': This paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms..."
}}
```

Begin!
Question: {input}
Thought:{agent_scratchpad}
"""


class PaperAgent:
    """
    An agentic system that can reason and use a suite of tools
    to accomplish complex research tasks.
    """
    def __init__(self, llm_provider: str = "google"):
        """
        Initializes the agent, its tools, and the agent executor.
        """
        print("--- Initializing Agentic System ---")

        if llm_provider == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        elif llm_provider == "local":
            llm = ChatOllama(
                model="llama3:8b-instruct-q4_K_M",
                temperature=0.0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        print(f"LLM Initialized: {getattr(llm, 'model', 'unknown')}")

        self.ingestor = Ingestor()
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        chroma_db_path = os.getenv("CHROMA_DB_PATH", "./paper_db")
        self.kb = KnowledgeBase(db_path=chroma_db_path, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password)
        if llm_provider == "google":
            extractor_model = "gemini-1.5-flash"
            extractor_api_type = "google"
        else:
            extractor_model = "llama3:8b-instruct-q4_K_M"
            extractor_api_type = "local"
        self.extractor = Extractor(api_type=extractor_api_type, model=extractor_model)
        
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)

        rag_sub_agent = self._create_rag_pipeline(llm)

        self.tools = [
            web_search_tool,
            GetPaperMetadataByTitleTool(graph=graph),
            AnswerFromPapersTool(rag_agent=rag_sub_agent),
            GraphQueryTool(graph=graph), 
            ArxivSearchTool(ingestor=self.ingestor, kb=self.kb),
            ArxivFetchTool(ingestor=self.ingestor, kb=self.kb),
            PaperSummarizationTool(kb=self.kb, extractor=self.extractor),
            TableExtractionTool(kb=self.kb, extractor=self.extractor),
            RelationshipAnalysisTool(graph=graph, llm=llm),
            CitationAnalysisTool(graph=graph),             
            KeywordExtractionTool(kb=self.kb, extractor=self.extractor), 
            DynamicVisualizationTool(code_writing_llm=llm),
            ArchitectureDiagramTool(),
            ConflictingResultsTool(
                name="conflicting_results_tool",
                description="Use this tool to find and explain conflicting or contradictory findings between two specific papers. You must provide the two paper IDs and the topic of conflict.",
                kb=self.kb,
                extractor=self.extractor
            ),
            LiteratureGapTool(
                name="literature_gap_tool",
                description="A powerful research tool. Use this to analyze the current state of a research field and get suggestions for novel future work. Operates on a collection of papers.",
                kb=self.kb,
                extractor=self.extractor
            ),
            DataToCsvTool() 
        ]
        tool_names = ", ".join([str(t.name) for t in self.tools if t.name])
        print(f"Tools Initialized: [{tool_names}]")

        prompt = ChatPromptTemplate.from_template(ROBUST_JSON_PROMPT_TEMPLATE)
        agent = create_json_chat_agent(llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3, 
            return_intermediate_steps=False,
            early_stopping_method="force"
        )
        print("--- Agentic System Ready ---")


    def _create_rag_pipeline(self, llm):
        """Helper to create a temporary object to pass to the KB tool."""
        class TempRAGAgent:
            def __init__(self, kb, llm_client, model_name):
                self.kb = kb
                self.llm = llm_client
                self.model = model_name
            
            def run_query(self, user_query):
                context_str = ""
                search_results = self.kb.search(query=user_query, n_results=3)
                for i in range(len(search_results['ids'][0])):
                    doc = search_results['documents'][0][i]
                    meta = search_results['metadatas'][0][i]
                    context_str += f"--- Context Snippet {i+1} ---\n"
                    context_str += f"Source Metadata: {meta}\n"
                    context_str += f"Content: {doc}\n\n"
                
                prompt = f"Answer the user's question based ONLY on the provided context, which includes metadata and content for each snippet:\n<context>\n{context_str}\n</context>\n\nQuestion: {user_query}"
                try:
                    response = self.llm.invoke(prompt)
                except Exception as e:
                    err_str = str(e).lower()
                    if "quota" in err_str or "429" in err_str or "resourceexhausted" in err_str:
                        print("Quota hit or rate limited in RAG. Switching to local Llama 3 model...")
                        self.llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.0)
                        response = self.llm.invoke(prompt)
                    else:
                        raise
                return response.content
        
        return TempRAGAgent(self.kb, llm, getattr(llm, "model", "unknown"))


    def run(self, user_query: str):
        """Runs the full agentic loop for a single, simple task."""
        print(f"\n--- Worker Agent Executing Task: '{user_query[:100]}...' ---")
        response = self.agent_executor.invoke({"input": user_query})
        return response

    def run_single_tool(self, tool_name: str, tool_input: Any) -> str:
        """
        Executes a single tool directly without the full agentic loop.
        This is used by the MasterAgent to control the worker precisely.
        """
        if tool_name not in [tool.name for tool in self.tools]:
            return f"Error: Tool '{tool_name}' not found."
        try:
            tool_to_run = next(t for t in self.tools if t.name == tool_name)
            observation = tool_to_run.run(tool_input)
            return str(observation)
        except Exception as e:
            return f"Error running tool '{tool_name}': {e}"