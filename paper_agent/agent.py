from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Any
from langchain_neo4j import Neo4jGraph 

from .knowledge_base import KnowledgeBase
from .ingestor import Ingestor
from .tools import (
    GraphPaperFinderTool, 
    QuestionAnsweringTool,   
    ArxivSearchTool,
    ArxivFetchTool,
    PaperSummarizationTool,
    web_search_tool,
    GraphQueryTool,         
    TableExtractionTool,   
    RelationshipAnalysisTool,
    DynamicVisualizationTool 
)
from .extractor import Extractor
from langchain_google_genai import ChatGoogleGenerativeAI # <-- NEW IMPORT
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
1. After using a tool and getting an Observation, THINK about whether you have enough information to answer the user's question.
2. If the user asks you to find or load papers, your job is done once you have successfully used the 'arxiv_paper_search_and_load_tool'. Your final answer should just confirm that the papers were loaded.
3. The 'paper_summarization_tool' and 'arxiv_fetch_by_id_tool' require a unique 'paper_id'. The 'paper_id' can be found in the 'Source Metadata' from the 'paper_finder_tool'. You MUST extract the 'paper_id' from the metadata and use it as the input for these tools. Do NOT use the paper's title as the ID.
4. **Data Flow:** The 'table_extraction_tool' outputs structured JSON. The 'plot_generation_tool' takes this JSON as its 'json_data' input. Ensure your plan connects these tools correctly.
5. **CRITICAL RULE: When matching on string properties like 'title' in a Cypher query, ALWAYS use a case-insensitive search with 'toLower()' and 'CONTAINS'. For example: `WHERE toLower(p.title) CONTAINS 'attention'`. Do NOT use '=' for string matching.**
6. When using a tool, ALWAYS provide ALL required arguments, using the EXACT argument names as specified in the tool's description/schema. Never invent argument names. If you are unsure, check the tool's schema.
7. Do not get stuck in a loop. If a tool is not giving you the information you need, try a different tool or conclude that you cannot answer the question.
8. Once you have a satisfactory answer, you MUST provide the final answer to the user in the specified format.
9. Never call paper_summarization_tool or arxiv_fetch_by_id_tool with a null or missing paper_id. Always extract a valid string value from the metadata. If you cannot find a valid paper_id, do not call the tool and consider using another tool or asking the user for clarification.
10. If, after 2 or 3 tool uses, you cannot find a direct answer to the user's question, provide a best-effort summary of what you did find, or explicitly state that the answer is not available in the knowledge base. Always provide a final answer, even if it is "I could not find an answer to your question in the current knowledge base."
11. For questions about authors, collaborations, or relationships between papers (e.g., 'Who wrote X?', 'Who collaborated with Y?', 'What papers cite Z?'), always use the graph_query_tool. Do not use paper_finder_tool for these questions.
12. When you receive a JSON list of papers from paper_finder_tool, extract the paper_id from the first paper and use it as input to paper_summarization_tool.

Use the following format for each step:

Thought: The user's question is about X. I should use tool Y to find the answer.
Action:
```json
{{
  "action": "tool_name",
  "action_input": {{"argument_name": "argument value"}}
}}
```

# Examples (always use the exact argument names from the tool schema):

## paper_finder_tool (requires 'query' argument, not for author/collaboration questions):
```json
{{
  "action": "paper_finder_tool",
  "action_input": {{"query": "Attention is All You Need"}}
}}
```

## question_answering_tool (requires 'question' argument):
```json
{{
  "action": "question_answering_tool",
  "action_input": {{"question": "What is Mixture of Experts?"}}
}}
```

## arxiv_paper_search_and_load_tool (requires 'query' and optionally 'max_results'):
```json
{{
  "action": "arxiv_paper_search_and_load_tool",
  "action_input": {{"query": "Mixture of Experts", "max_results": 5}}
}}
```

## graph_query_tool (for author/collaboration/relationship questions):
```json
{{
  "action": "graph_query_tool",
  "action_input": {{"query": "MATCH (p:Paper)<-[:AUTHORED]-(a:Author) WHERE toLower(p.title) CONTAINS 'attention' RETURN a.name AS author LIMIT 10"}}
}}
```

## paper summarization chain example:
Thought: The user wants a summary of a specific paper. I should use paper_finder_tool to get the paper_id.
Action:
```json
{{
  "action": "paper_finder_tool",
  "action_input": {{"query": "Attention is All You Need"}}
}}
```
Observation: {{"papers": [{{"paper_id": "attention.pdf", "title": "Attention Is All You Need", "authors": "..."}}]}}
Thought: I have found the paper_id. I should now use paper_summarization_tool with this paper_id.
Action:
```json
{{
  "action": "paper_summarization_tool",
  "action_input": {{"paper_id": "attention.pdf"}}
}}
```
Observation: "This paper introduces the Transformer architecture, which relies entirely on attention mechanisms..."
Thought: I have gathered enough information. I will now provide the final answer.
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "Summary of 'Attention Is All You Need': This paper introduces the Transformer architecture, which relies entirely on attention mechanisms..."
}}
```

// INCORRECT: Do NOT do this!
```json
{{
  "action": "paper_summarization_tool",
  "action_input": {{"paper_id": null}}
}}
```

Observation: The result from using the tool.
... (this Thought/Action/Observation can repeat)
Thought: I have gathered enough information. I will now provide the final answer.
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "The final, comprehensive answer to the user's original question."
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
    def __init__(self, db_path: str = "./paper_db", llm_provider: str = "google", neo4j_uri = "neo4j://172.20.128.55:7687", neo4j_user="neo4j", neo4j_password="password"):
        """
        Initializes the agent, its tools, and the agent executor.
        """
        print("--- Initializing Agentic System ---")

        # 1. Initialize the LLM based on the provider
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

        # 2. Initialize our core components
        self.ingestor = Ingestor()
        # The KnowledgeBase now takes the Neo4j credentials
        self.kb = KnowledgeBase(db_path=db_path, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password)
        if llm_provider == "google":
            extractor_model = "gemini-1.5-flash"
            extractor_api_type = "google"
        else:
            extractor_model = "llama3:8b-instruct-q4_K_M"
            extractor_api_type = "local"
        self.extractor = Extractor(api_type=extractor_api_type, model=extractor_model)
        
        # --- NEW: Initialize the Neo4jGraph utility ---
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)

        # 3. Create the RAG "sub-agent" for the KB tool
        # This feels a bit circular, but it's a clean way to reuse our RAG logic
        # We create a temporary RAG agent to pass to the tool.
        rag_sub_agent = self._create_rag_pipeline(llm)

        # 4. Initialize the tool suite, including the new graph tool
        self.tools = [
            web_search_tool,
            GraphPaperFinderTool(graph=graph), # <-- REPLACE THE OLD FINDER
            QuestionAnsweringTool(rag_agent=rag_sub_agent), # <-- The "smart" RAG tool
            GraphQueryTool(graph=graph), # <-- ADD NEW TOOL
            ArxivSearchTool(ingestor=self.ingestor, kb=self.kb),
            ArxivFetchTool(ingestor=self.ingestor, kb=self.kb),
            PaperSummarizationTool(kb=self.kb, extractor=self.extractor),
            TableExtractionTool(kb=self.kb, extractor=self.extractor),
            RelationshipAnalysisTool(graph=graph, llm=llm),
            # Replace the old tool with the new one, passing it an LLM to use
            DynamicVisualizationTool(code_writing_llm=llm)
        ]
        tool_names = ", ".join([str(t.name) for t in self.tools if t.name])
        print(f"Tools Initialized: [{tool_names}]")

        # 5. Create the Agent using our NEW, more robust prompt
        prompt = ChatPromptTemplate.from_template(ROBUST_JSON_PROMPT_TEMPLATE)
        agent = create_json_chat_agent(llm, self.tools, prompt)
        # 6. Create the Agent Executor, which runs the main loop (add max_iterations)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3, # <-- Let's reduce this slightly
            # --- THE FINAL FIX ---
            # This tells the agent that if the LLM outputs something that isn't a tool
            # or a final answer, just return that output directly. This stops loops.
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
                
                # --- THE TWEAK ---
                # Add the metadata, including the crucial paper_id, to the context string.
                for i in range(len(search_results['ids'][0])):
                    doc = search_results['documents'][0][i]
                    meta = search_results['metadatas'][0][i]
                    context_str += f"--- Context Snippet {i+1} ---\n"
                    context_str += f"Source Metadata: {meta}\n" # <-- ADD THIS LINE
                    context_str += f"Content: {doc}\n\n"
                
                prompt = f"Answer the user's question based ONLY on the provided context, which includes metadata and content for each snippet:\n<context>\n{context_str}\n</context>\n\nQuestion: {user_query}"
                response = self.llm.invoke(prompt)
                return response.content
        
        return TempRAGAgent(self.kb, llm, getattr(llm, "model", "unknown"))


    def run(self, user_query: str):
        """Runs the full agentic loop for a single, simple task."""
        print(f"\n--- Worker Agent Executing Task: '{user_query[:100]}...' ---")
        response = self.agent_executor.invoke({"input": user_query})
        return response

    # --- NEW HELPER METHOD ---
    def run_single_tool(self, tool_name: str, tool_input: Any) -> str:
        """
        Executes a single tool directly without the full agentic loop.
        This is used by the MasterAgent to control the worker precisely.
        """
        if tool_name not in [tool.name for tool in self.tools]:
            return f"Error: Tool '{tool_name}' not found."
        try:
            # Find the tool in our list
            tool_to_run = next(t for t in self.tools if t.name == tool_name)
            # Use the tool's .run() method directly
            observation = tool_to_run.run(tool_input)
            return str(observation)
        except Exception as e:
            return f"Error running tool '{tool_name}': {e}"