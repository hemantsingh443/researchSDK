from .agent import PaperAgent as WorkerAgent # Our ReAct agent is now the worker
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any
import json
import re 

class MasterAgent:
    """
    A stateful, self-correcting agent that can reason, re-plan, and
    execute complex tasks by orchestrating a worker agent and its tools.
    """
    def __init__(self, worker_agent: WorkerAgent, llm: BaseChatModel, max_loops: int = 20):
        self.worker_agent = worker_agent
        self.llm = llm
        self.max_loops = max_loops
        self.tool_manifest = self._create_tool_manifest()
        print("MasterAgent initialized. The true brain is online.")

    def _create_tool_manifest(self) -> str:
        """Creates a string representation of the available tools for the planner prompt."""
        manifest = ""
        for tool in self.worker_agent.tools:
            manifest += f"Tool Name: {tool.name}\n"
            manifest += f"Tool Description: {tool.description}\n"
            manifest += f"Tool Arguments: {tool.args}\n\n"
        return manifest

    def _clean_and_parse_json(self, json_string: str) -> Dict[str, Any]:
        """
        A robust function to clean and parse a JSON string from an LLM response.
        It handles markdown code blocks and other common LLM artifacts.
        """
        # Remove markdown code block wrappers (```json ... ```)
        if json_string.startswith("```json"):
            json_string = json_string[7:] # Remove ```json
            if json_string.endswith("```"):
                json_string = json_string[:-3] # Remove ```
        # Sometimes models just use ``` ... ```
        if json_string.startswith("```"):
            json_string = json_string[3:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
        # Strip any leading/trailing whitespace
        json_string = json_string.strip()
        # Now, try to parse the cleaned string
        return json.loads(json_string)

    def run(self, user_query: str):
        """The main loop of the Master Agent."""
        
        # Initialize the state (the "scratchpad")
        state: Dict[str, Any] = {
            "original_request": user_query,
            "completed_steps": []
        }

        for i in range(self.max_loops):
            print(f"\n{'='*20} MASTER AGENT LOOP {i+1}/{self.max_loops} {'='*20}")

            # 1. THINK & RE-PLAN
            # Create the prompt for the master LLM to decide the next action
            master_prompt = self._create_master_prompt(state)
            
            print("MasterAgent is thinking...")
            llm_response_str = self.llm.invoke(master_prompt).content
            
            try:
                # The LLM should respond with a JSON object containing its thought and the next action
                response_str = llm_response_str
                if not isinstance(response_str, str):
                    # If it's a list, join elements as string
                    if isinstance(response_str, list):
                        response_str = "\n".join([str(x) for x in response_str])
                    else:
                        response_str = str(response_str)
                # --- USE THE NEW ROBUST PARSER ---
                decision = self._clean_and_parse_json(response_str)
                thought = decision.get("thought", "No thought provided.")
                action_json = decision.get("action", {})
            except (json.JSONDecodeError, AttributeError) as e: # Catch more potential errors
                print(f"MasterAgent failed to generate valid JSON. Error: {e}")
                print(f"Raw Response was: {llm_response_str}")
                thought = "Error in parsing LLM response. Attempting to recover."
                # We can make the recovery smarter later, for now, we stop.
                action_json = {"action": "Final Answer", "action_input": "I encountered an internal error and cannot proceed."}

            print(f"Thought: {thought}")

            # 2. ACT
            if isinstance(action_json, dict):
                action_name = action_json.get("action")
                action_input = action_json.get("action_input")
            elif isinstance(action_json, str):
                action_name = action_json
                action_input = None
            else:
                action_name = None
                action_input = None

            # Check for completion
            if action_name == "Final Answer":
                print("MasterAgent has decided the task is complete.")
                return action_input

            # Execute the chosen action using the worker agent
            if not isinstance(action_name, str):
                print("Invalid or missing action name. Skipping this step.")
                break
            worker_response = self.worker_agent.run_single_tool(action_name, action_input)
            
            # 3. OBSERVE & UPDATE STATE
            # Add the result of this loop to our state
            state["completed_steps"].append({
                "step": i + 1,
                "thought": thought,
                "action_taken": action_json,
                "observation": worker_response
            })

        print("MasterAgent reached maximum loops.")
        return "I could not complete the request within the allowed number of steps."

    def _create_master_prompt(self, state: Dict[str, Any]) -> str:
        """Creates the dynamic prompt for the master LLM."""
        return f"""
You are the Master Agent, a hyper-intelligent AI that directs a worker agent to solve complex user requests.
Your job is to analyze the user's goal and the history of actions taken so far, then decide the single best next action.

**TOOLS OVERVIEW:**
---
{self.tool_manifest}
---

**PROJECT STATE:**
---
{json.dumps(state, indent=2)}
---

**YOUR TASK:**
Based on the project state, what is the single best next action to take to progress towards the original request?

**RULES:**
1.  **Analyze the observations:** Look at the 'observation' from the last completed step. Did it succeed? Did it fail? Does it contain the information you need?
2.  **Self-Correct:** If a previous action failed or didn't provide the right information, change the plan! Do not repeat the same failed action. Try a different tool or different inputs.
3.  **Conclude:** If the last observation contains enough information to answer the original request, your next action MUST be 'Final Answer'.

Respond with a single JSON object with two keys: "thought" (your reasoning for the decision) and "action" (the action to take).

Example of a valid response:
{{
    "thought": "The previous step failed to find the paper by title. I will now try to find it using its known arXiv ID, which is a more reliable method.",
    "action": {{
        "action": "arxiv_fetch_by_id_tool",
        "action_input": {{"paper_arxiv_id": "1706.03762"}}
    }}
}}
"""