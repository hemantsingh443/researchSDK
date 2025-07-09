from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

from .knowledge_base import KnowledgeBase
from .ingestor import Ingestor
from .tools import KnowledgeBaseQueryTool, ArxivSearchTool, web_search_tool

ROBUST_JSON_PROMPT_TEMPLATE = """
You are a helpful scientific research assistant. Your goal is to answer the user's question by thinking step-by-step.
You must respond with a JSON object containing either an 'action' and 'action_input' key, or a 'action' and 'action_input' key with a final answer.

You have access to the following tools:
{tools}

The available tool names are: {tool_names}

Here are some rules to follow:
1. After using a tool and getting an Observation, THINK about whether you have enough information to answer the user's question.
2. If the user asks you to find or load papers, your job is done once you have successfully used the 'arxiv_paper_search_and_load_tool'. Your final answer should just confirm that the papers were loaded.
3. Do not get stuck in a loop. If a tool is not giving you the information you need, try a different tool or conclude that you cannot answer the question.
4. Once you have a satisfactory answer, you MUST provide the final answer to the user in the specified format.

Use the following format:

Thought: The user's question is about X. I should use tool Y to find the answer.
Action:
```json
{{
  "action": "tool_name",
  "action_input": "the input for the tool"
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
    def __init__(self, db_path: str = "./paper_db"):
        """
        Initializes the agent, its tools, and the agent executor.
        """
        print("--- Initializing Agentic System ---")

        # 1. Initialize the LLM (we'll use LangChain's wrapper)
        # This points to our local Ollama server
        llm = ChatOpenAI(
            base_url='http://localhost:11434/v1',
            model='llama3:8b-instruct-q4_K_M',
            temperature=0.0
        )
        print(f"LLM Initialized: {llm.model_name}")

        # 2. Initialize our core components
        self.ingestor = Ingestor()
        self.kb = KnowledgeBase(db_path=db_path)
        
        # 3. Create the RAG "sub-agent" for the KB tool
        # This feels a bit circular, but it's a clean way to reuse our RAG logic
        # We create a temporary RAG agent to pass to the tool.
        rag_sub_agent = self._create_rag_pipeline(llm)

        # 4. Initialize the tool suite
        self.tools = [
            web_search_tool,
            KnowledgeBaseQueryTool(rag_agent=rag_sub_agent),
            ArxivSearchTool(ingestor=self.ingestor, kb=self.kb)
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
            verbose=True, # Set to True to see the agent's thoughts!
            handle_parsing_errors=True, # Crucial for local models
            max_iterations=5 # <-- IMPORTANT GUARDRAIL
        )
        print("--- Agentic System Ready ---")


    def _create_rag_pipeline(self, llm):
        """Helper to create a temporary object to pass to the KB tool."""
        # This is a bit of a hack to reuse our previous RAG logic
        # without a full refactor.
        class TempRAGAgent:
            def __init__(self, kb, llm_client, model_name):
                self.kb = kb
                self.llm = llm_client
                self.model = model_name
            
            def run_query(self, user_query):
                # This is the RAG logic from our old agent.py
                context_str = ""
                search_results = self.kb.search(query=user_query, n_results=3)
                for i, doc in enumerate(search_results['documents'][0]):
                    source = search_results['metadatas'][0][i]['title']
                    context_str += f"--- Context Snippet {i+1} (from paper: {source}) ---\n{doc}\n\n"
                
                prompt = f"Answer the user's question based ONLY on the provided context:\n<context>\n{context_str}\n</context>\n\nQuestion: {user_query}"
                response = self.llm.invoke(prompt)
                return response.content
        
        return TempRAGAgent(self.kb, llm, llm.model_name)


    def run(self, user_query: str):
        """
        Runs the agent with a user query.
        """
        print(f"\n--- Executing Agent with Query: '{user_query}' ---")
        response = self.agent_executor.invoke({"input": user_query})
        return response