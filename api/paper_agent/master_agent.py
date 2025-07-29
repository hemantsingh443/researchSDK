import json
import re
import os
import datetime
from typing import Dict, Any, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

from .tools import get_tools
from .knowledge_base import KnowledgeBase
from .ingestor import Ingestor
from .extractor import Extractor

# Load environment variables
load_dotenv()

class MasterAgent:
    """
    A powerful, stateful agent that can reason, plan, and execute complex research tasks.
    It operates in a loop, maintains a state, and can self-correct its plan.
    This class merges the original Master/Worker/Planner roles into one cohesive unit.
    """
    def __init__(self, llm_provider: str = "google", max_loops: int = 15):
        """Initialize the MasterAgent with the specified LLM provider.
        
        Args:
            llm_provider: The LLM provider to use ("google" or "local")
            max_loops: Maximum number of reasoning loops before termination
        """
        print("--- Initializing Master Agent ---")
        self.max_loops = max_loops
        
        # Initialize LLM based on provider
        self.llm = self._initialize_llm(llm_provider)
        
        # Initialize components with proper configuration
        self.ingestor = Ingestor()
        
        # Load configuration from environment variables
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        chroma_db_path = os.getenv("CHROMA_DB_PATH", "./paper_db")
        
        self.kb = KnowledgeBase(
            db_path=chroma_db_path,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
        
        # Initialize extractor with appropriate model based on provider
        extractor_model = "gemini-1.5-flash" if llm_provider == "google" else "llama3:8b-instruct-q4_K_M"
        self.extractor = Extractor(api_type=llm_provider, model=extractor_model)
        
        # Initialize tools with proper dependencies
        self.tools = get_tools(self.kb, self.extractor, self.ingestor, self.llm)
        self.tool_manifest = self._create_tool_manifest()
        self.tool_dict = {tool.name: tool for tool in self.tools}
        
        # Initialize RAG pipeline for question answering
        self.rag_agent = self._create_rag_pipeline()
        
        print(f"LLM Provider: {llm_provider}")
        print(f"Tools Initialized: {[tool.name for tool in self.tools]}")
        print("--- Master Agent Ready ---")

    def _create_tool_manifest(self) -> str:
        """Creates a string representation of the available tools for the planner prompt."""
        return "\n\n".join([f"Tool Name: {tool.name}\nDescription: {tool.description}\nArguments: {tool.args}" for tool in self.tools])

    def _clean_and_parse_json(self, json_string: str) -> Dict[str, Any]:
        """Robustly cleans and parses a JSON string from an LLM response."""
        # Find the first '{' and the last '}' to extract the JSON object
        match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON object found in the LLM response.")
        
        json_str = match.group(0)
        # Remove trailing commas that would make the JSON invalid
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        
        return json.loads(json_str)

    def _create_prompt(self, state: Dict[str, Any]) -> str:
        """Creates the dynamic prompt for the master LLM."""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.datetime.now().year
        return f"""
You are the Master Agent, a hyper-intelligent AI directing research. Your job is to analyze the user's goal and the history of actions taken, then decide the single best next action to solve the user's request.

**TOOLS OVERVIEW:**
---
{self.tool_manifest}
---

**PROJECT STATE (History of steps taken and data collected):**
---
{self._serialize_state(state)}
---

**YOUR TASK:**
Based on the project state, what is the single best next action to take?

**RULES OF REASONING:**
1.  **Analyze Observations:** Look at the `observation` from the last step. Did it succeed or fail? Does it contain the information you need?
2.  **Self-Correct:** If a tool failed, DO NOT try the same tool again with the same inputs. Analyze the reason for failure.
    - If you need a piece of information (e.g., a `paper_id`), and the tool to get it failed, try a different tool or approach to get that information.
    - If a downstream tool failed because an upstream tool didn't provide data (e.g., visualization failed because table extraction found nothing), DO NOT try the downstream tool again. Acknowledge the missing data and either find another way to get it or conclude that it's unavailable.
    - When searching for papers, if the initial search doesn't return the exact paper you're looking for, try:
        * Using more specific search terms
        * Using the exact title in quotes
        * Trying different search tools
        * Using DuckDuckGo search to find the arXiv ID first, then using that ID with other tools
        * Be aware that papers from the future are not yet available. Today's date is {current_date} and the current year is {current_year}. If searching for papers from a specific year, consider that the current year is {current_year}. If asked about papers from a future year (e.g., {current_year + 1}), explain that they are not yet available.

**SEARCH BEST PRACTICES:**
- When searching for a specific paper and the results don't match what you expected:
  1. Check if the paper title in the results is different from what you were looking for
  2. Try using more specific search terms or the exact title in quotes
  3. Use DuckDuckGo search to find the exact arXiv ID first, then use that ID with other tools
  4. If multiple related papers are found, analyze which one is most relevant to the user's request
  5. If no relevant papers are found, consider informing the user and asking for more specific information

**EXAMPLES OF GOOD SEARCH STRATEGIES:**
- Instead of searching for "Attention", search for "Attention is All You Need"
- If that doesn't work, try "\"Attention is All You Need\"" (with quotes)
- If still no success, use DuckDuckGo to search for "arxiv Attention is All You Need" to find the ID

**SCRATCHPAD USAGE:**
- The scratchpad contains data from previous successful tool executions
- Use data in the scratchpad to inform your next action (e.g., if paper_metadata exists, you can use the paper_id)
- The scratchpad also stores search results for later reference

**Example Response:**
{{
    "thought": "The user wants to compare two papers. I have successfully found the ID for the first paper and stored it in the scratchpad. Now I need to find the ID for the second paper, 'BERT'.",
    "action": {{
        "name": "get_paper_metadata_by_title",
        "input": {{"query": "BERT"}}
    }}
}}
"""

    async def run(self, user_query: str, websocket_callback=None):
        """The main loop of the Master Agent."""
        state = {
            "original_request": user_query,
            "completed_steps": [],
            "scratchpad": {} # A place to store important data like IDs, tables, etc.
        }

        for i in range(self.max_loops):
            loop_info = {
                "loop": i + 1,
                "max_loops": self.max_loops,
                "thought": "",
                "action": {},
                "observation": ""
            }
            
            print(f"\n{'='*20} MASTER AGENT LOOP {i+1}/{self.max_loops} {'='*20}")
            
            # Send loop start message if callback is provided
            if websocket_callback:
                import asyncio
                if asyncio.iscoroutinefunction(websocket_callback):
                    await websocket_callback({
                        'type': 'loop_start',
                        'loop': i + 1,
                        'max_loops': self.max_loops
                    })
                else:
                    websocket_callback({
                        'type': 'loop_start',
                        'loop': i + 1,
                        'max_loops': self.max_loops
                    })
            
            # 1. Create the prompt based on the current state
            master_prompt = self._create_prompt(state)
            
            # 2. Get the LLM's decision
            print("MasterAgent is thinking...")
            if websocket_callback:
                import asyncio
                if asyncio.iscoroutinefunction(websocket_callback):
                    await websocket_callback({
                        'type': 'loop_thinking',
                        'loop': i + 1,
                        'message': 'MasterAgent is thinking...'
                    })
                else:
                    websocket_callback({
                        'type': 'loop_thinking',
                        'loop': i + 1,
                        'message': 'MasterAgent is thinking...'
                    })
            
            llm_response_str = self.llm.invoke(master_prompt).content
            
            try:
                decision = self._clean_and_parse_json(llm_response_str)
                thought = decision.get("thought", "No thought provided.")
                action = decision.get("action", {})
                action_name = action.get("name")
                action_input = action.get("input")
                print(f"Thought: {thought}")
                
                # Update loop info
                loop_info["thought"] = thought
                loop_info["action"] = action
                
                # Send thought to frontend
                if websocket_callback:
                    import asyncio
                    if asyncio.iscoroutinefunction(websocket_callback):
                        await websocket_callback({
                            'type': 'loop_thought',
                            'loop': i + 1,
                            'thought': thought,
                            'action': action
                        })
                    else:
                        websocket_callback({
                            'type': 'loop_thought',
                            'loop': i + 1,
                            'thought': thought,
                            'action': action
                        })
            except Exception as e:
                error_msg = f"MasterAgent failed to generate a valid JSON decision. Error: {e}"
                print(error_msg)
                print(f"Raw Response was: {llm_response_str}")
                state['completed_steps'].append({"step": i+1, "thought": "Error in parsing LLM decision.", "action_taken": "None", "observation": str(e)})
                
                # Send error to frontend
                if websocket_callback:
                    websocket_callback({
                        'type': 'loop_error',
                        'loop': i + 1,
                        'error': error_msg
                    })
                continue

            # 3. Check for termination
            if action_name == "Final Answer":
                print("MasterAgent has decided the task is complete.")
                state["completed_steps"].append({"step": i + 1, "thought": thought, "action_taken": "Final Answer", "observation": action_input})
                
                # Send completion message
                if websocket_callback:
                    import asyncio
                    if asyncio.iscoroutinefunction(websocket_callback):
                        await websocket_callback({
                            'type': 'loop_final_answer',
                            'loop': i + 1,
                            'message': 'MasterAgent has decided the task is complete.'
                        })
                    else:
                        websocket_callback({
                            'type': 'loop_final_answer',
                            'loop': i + 1,
                            'message': 'MasterAgent has decided the task is complete.'
                        })
                break

            # 4. Execute the chosen tool
            if action_name in self.tool_dict:
                print(f"MasterAgent executes: {action_name} with input: {action_input}")
                if websocket_callback:
                    import asyncio
                    if asyncio.iscoroutinefunction(websocket_callback):
                        await websocket_callback({
                            'type': 'loop_action',
                            'loop': i + 1,
                            'action_name': action_name,
                            'action_input': action_input,
                            'message': f"MasterAgent executes: {action_name}"
                        })
                    else:
                        websocket_callback({
                            'type': 'loop_action',
                            'loop': i + 1,
                            'action_name': action_name,
                            'action_input': action_input,
                            'message': f"MasterAgent executes: {action_name}"
                        })
                
                tool_to_run = self.tool_dict[action_name]
                try:
                    # Use .invoke() for LangChain tools which handles argument mapping
                    observation = tool_to_run.invoke(action_input)
                    print(f"Observation: {observation}")
                    
                    # Update loop info
                    loop_info["observation"] = str(observation)
                    
                    # Send observation to frontend
                    if websocket_callback:
                        import asyncio
                        if asyncio.iscoroutinefunction(websocket_callback):
                            await websocket_callback({
                                'type': 'loop_observation',
                                'loop': i + 1,
                                'observation': str(observation)
                            })
                        else:
                            websocket_callback({
                                'type': 'loop_observation',
                                'loop': i + 1,
                                'observation': str(observation)
                            })
                except Exception as e:
                    observation = f"Error executing tool '{action_name}': {e}"
                    print(f"ERROR: {observation}")
                    
                    # Send error to frontend
                    if websocket_callback:
                        import asyncio
                        if asyncio.iscoroutinefunction(websocket_callback):
                            await websocket_callback({
                                'type': 'loop_error',
                                'loop': i + 1,
                                'error': observation
                            })
                        else:
                            websocket_callback({
                                'type': 'loop_error',
                                'loop': i + 1,
                                'error': observation
                            })
                
                # Update state
                state["completed_steps"].append({"step": i + 1, "thought": thought, "action_taken": action, "observation": observation})
                
                # --- STATEFUL SCRATCHPAD UPDATE ---
                # A more robust way to pass context
                if isinstance(observation, dict) and observation.get('status') == 'success':
                    if action_name == 'get_paper_metadata_by_title':
                        # Check if we found papers and store them
                        if 'data' in observation and 'papers' in observation['data']:
                            found_papers = observation['data']['papers']
                            state['scratchpad']['paper_metadata'] = observation['data']
                            # If we're looking for a specific paper, check if we found it
                            if 'query' in observation['data']:
                                query = observation['data']['query']
                                # Store the search results for potential later use
                                if 'paper_search_results' not in state['scratchpad']:
                                    state['scratchpad']['paper_search_results'] = {}
                                state['scratchpad']['paper_search_results'][query] = found_papers
                    elif action_name == 'table_extraction_tool':
                        state['scratchpad']['extracted_table'] = observation['data']
                    elif action_name == 'arxiv_paper_search_and_load':
                        # Store the loaded papers for later reference
                        if 'data' in observation and 'papers' in observation['data']:
                            loaded_papers = observation['data']['papers']
                            if 'loaded_papers' not in state['scratchpad']:
                                state['scratchpad']['loaded_papers'] = []
                            state['scratchpad']['loaded_papers'].extend(loaded_papers)

            else:
                error_msg = f"Error: Chosen tool '{action_name}' not found."
                print(error_msg)
                state['completed_steps'].append({"step": i + 1, "thought": thought, "action_taken": action, "observation": f"Tool '{action_name}' does not exist."})
                
                # Send error to frontend
                if websocket_callback:
                    websocket_callback({
                        'type': 'loop_error',
                        'loop': i + 1,
                        'error': error_msg
                    })
        
        else:
            print("MasterAgent reached maximum loops.")
            if websocket_callback:
                import asyncio
                if asyncio.iscoroutinefunction(websocket_callback):
                    await websocket_callback({
                        'type': 'loop_max_reached',
                        'message': 'MasterAgent reached maximum loops.'
                    })
                else:
                    websocket_callback({
                        'type': 'loop_max_reached',
                        'message': 'MasterAgent reached maximum loops.'
                    })

        # 5. Final Synthesis
        print("\n" + "="*20 + " GENERATING FINAL REPORT " + "="*20)
        if websocket_callback:
            import asyncio
            if asyncio.iscoroutinefunction(websocket_callback):
                await websocket_callback({
                    'type': 'loop_final_synthesis',
                    'message': 'Generating final report...'
                })
            else:
                websocket_callback({
                    'type': 'loop_final_synthesis',
                    'message': 'Generating final report...'
                })
        
        final_report = self._synthesize_final_report(state)
        
        # Return the final report and the thought process for the API
        return final_report, state["completed_steps"]

    def _initialize_llm(self, provider: str) -> BaseChatModel:
        """Initialize the appropriate LLM based on the provider."""
        if provider == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        elif provider == "local":
            return ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.0)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _serialize_state(self, state: Dict[str, Any]) -> str:
        """Safely serialize the state dictionary, handling custom response objects.
        
        Args:
            state: The state dictionary to serialize
            
        Returns:
            A JSON string representation of the state
        """
        def default_serializer(obj):
            if hasattr(obj, 'dict'):
                return obj.dict()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
            
        try:
            return json.dumps(state, indent=2, default=default_serializer)
        except Exception as e:
            print(f"Error serializing state: {e}")
            # Fallback to a simplified representation
            return json.dumps({
                k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                for k, v in state.items()
            }, indent=2)

    def _create_rag_pipeline(self):
        """Create a RAG pipeline for question answering."""
        class TempRAGAgent:
            def __init__(self, kb, llm):
                self.kb = kb
                self.llm = llm
            
            def run_query(self, user_query: str) -> str:
                """Run a query against the RAG pipeline."""
                context_str = ""
                search_results = self.kb.search(query=user_query, n_results=3)
                
                if not search_results or 'ids' not in search_results or not search_results['ids']:
                    return "No relevant documents found in the knowledge base."
                    
                for i in range(len(search_results['ids'][0])):
                    doc = search_results['documents'][0][i]
                    meta = search_results['metadatas'][0][i]
                    context_str += f"--- Context Snippet {i+1} ---\n"
                    context_str += f"Source Metadata: {meta}\n"
                    context_str += f"Content: {doc}\n\n"
                
                prompt = (
                    "Answer the user's question based ONLY on the provided context, "
                    "which includes metadata and content for each snippet:\n"
                    f"<context>\n{context_str}\n</context>\n\n"
                    f"Question: {user_query}"
                )
                
                try:
                    response = self.llm.invoke(prompt)
                    return response.content
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(x in error_msg for x in ["quota", "429", "resourceexhausted"]):
                        print("Quota hit or rate limited. Falling back to local model...")
                        # Fall back to local model if available
                        if not isinstance(self.llm, ChatOllama):
                            self.llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.0)
                            return self.run_query(user_query)
                    return f"Error in RAG pipeline: {str(e)}"
        
        return TempRAGAgent(self.kb, self.llm)
    def run_single_tool(self, tool_name: str, tool_input: dict) -> Union[str, dict]:
        """
        Execute a single tool directly without the full agentic loop.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters for the tool
            
        Returns:
            The tool's output as a string or dict
        """
        if tool_name not in self.tool_dict:
            return f"Error: Tool '{tool_name}' not found."
            
        try:
            tool = self.tool_dict[tool_name]
            if hasattr(tool, 'invoke'):
                return tool.invoke(tool_input)
            elif hasattr(tool, 'run'):
                return tool.run(tool_input)
            else:
                return f"Error: Tool '{tool_name}' has no callable method."
        except Exception as e:
            return f"Error running tool '{tool_name}': {str(e)}"

    def _synthesize_final_report(self, state: Dict[str, Any]) -> str:
        """Generates the final comprehensive report for the user."""
        try:
            # Use _serialize_state to properly handle JSON serialization
            serialized_state = self._serialize_state(state)
            
            synthesis_prompt = f"""
You are a brilliant scientific analyst. The project to answer '{state.get('original_request', 'the user query')}' is complete.
Synthesize all information from the 'PROJECT STATE' into a single, comprehensive Markdown report.

**Structure your report as follows:**
1.  **Executive Summary:** A single paragraph directly answering the user's core request.
2.  **Key Findings:** A bulleted list of the most important insights.
3.  **Detailed Analysis:** In-depth explanation of the steps taken and information found. If some steps failed, explain why.
4.  **Visualizations:** Reference any plots or artifacts created and explain them.
5.  **Conclusion & Limitations:** Conclude the report and mention any limitations.

**PROJECT STATE (The full history of your work):**
---
{serialized_state}
---

Write the final report in Markdown format.
"""
            final_report = self.llm.invoke(synthesis_prompt).content
            return final_report
            
        except Exception as e:
            return f"Error generating final report: {str(e)}\n\nRaw state data (simplified):\n{self._serialize_state(state)}"