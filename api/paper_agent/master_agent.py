from .agent import PaperAgent as WorkerAgent # Our ReAct agent is now the worker
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Dict, Any
import json
from langchain_community.chat_models import ChatOllama

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
        It handles markdown code blocks, extra text, and other common LLM artifacts.
        """
        import re
        # Remove markdown code block wrappers (```json ... ```)
        if json_string.startswith("```json"):
            json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
        if json_string.startswith("```"):
            json_string = json_string[3:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
        json_string = json_string.strip()
        # Use regex to extract the first JSON object
        match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON object found in LLM response.")

    def run(self, user_query: str):
        """The main loop of the Master Agent, now with a final synthesis step and explicit planning."""
        state: Dict[str, Any] = {
            "original_request": user_query,
            "completed_steps": []
        }
        plan_prompt = f"""
You are a world-class AI project planner. Your job is to break down the following user request into the SHORTEST, most EFFICIENT sequence of tool-based steps, using the available tools:
---\n{self.tool_manifest}\n---\nUser Request: '{user_query}'\n---\nRules:\n1. Each step must use a specific tool and specify its required arguments.\n2. Chain steps so outputs from one can be used as inputs for the next.\n3. Be explicit and efficient.\n4. Output ONLY a numbered list of steps, no extra text.\n"""
        plan_response = self.llm.invoke(plan_prompt).content
        # Ensure plan_response is always a list for robust iteration
        if not isinstance(plan_response, list):
            plan_response = [plan_response]
        plan_lines = []
        for item in plan_response:
            item_str = str(item)
            plan_lines.extend(item_str.split('\n'))
        plan = []
        for line in plan_lines:
            line = line.strip()
            if line and line[0].isdigit():
                step = line.split('.', 1)[-1].lstrip(" )-")
                plan.append(step)
        state["plan"] = plan
        print("--- Generated Plan ---")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")
        # --- END PLAN GENERATION ---
        for i in range(self.max_loops):
            print(f"\n{'='*20} MASTER AGENT LOOP {i+1}/{self.max_loops} {'='*20}")
            master_prompt = self._create_master_prompt(state)
            print("MasterAgent is thinking...")
            try:
                llm_response_str = self.llm.invoke(master_prompt).content
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "429" in err_str or "resourceexhausted" in err_str:
                    print("Quota hit or rate limited. Switching to local Llama 3 model...")
                    from langchain_community.chat_models import ChatOllama
                    self.llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.0)
                    llm_response_str = self.llm.invoke(master_prompt).content
                else:
                    raise
            try:
                response_str = llm_response_str
                if not isinstance(response_str, str):
                    if isinstance(response_str, list):
                        response_str = "\n".join([str(x) for x in response_str])
                    else:
                        response_str = str(response_str)
                decision = self._clean_and_parse_json(response_str)
                thought = decision.get("thought", "No thought provided.")
                action_json = decision.get("action", {})
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"MasterAgent failed to generate valid JSON. Error: {e}")
                print(f"Raw Response was: {llm_response_str}")
                break
            print(f"Thought: {thought}")
            if isinstance(action_json, dict):
                action_name = action_json.get("action")
                action_input = action_json.get("action_input")
            elif isinstance(action_json, str):
                action_name = action_json
                action_input = None
            else:
                action_name = None
                action_input = None
            if action_name == "Final Answer":
                print("MasterAgent has decided the task is complete. Moving to final report generation.")
                state["completed_steps"].append({
                    "step": i + 1,
                    "thought": thought,
                    "action_taken": "Final Answer",
                    "observation": action_input
                })
                break
            if isinstance(action_name, str) and action_name:
                print(f"MasterAgent directs worker to execute: {action_name}")
                worker_response = self.worker_agent.run_single_tool(action_name, action_input)
                state["completed_steps"].append({
                    "step": i + 1,
                    "thought": thought,
                    "action_taken": action_json,
                    "observation": worker_response
                })
                # --- CONTEXT PASSING LOGIC ---
                try:
                    if action_name == "table_extraction_tool":
                        import json
                        paper_id = action_input.get("paper_id") if isinstance(action_input, dict) else None
                        if worker_response and worker_response.strip().startswith("{"):
                            table_json = json.loads(worker_response)
                            if paper_id:
                                state[f"table_{paper_id}"] = table_json
                                print(f"[Context Passing] Stored table for {paper_id}")
                    if action_name == "paper_summarization_tool":
                        paper_id = action_input.get("paper_id") if isinstance(action_input, dict) else None
                        if paper_id:
                            state[f"summary_{paper_id}"] = worker_response
                            print(f"[Context Passing] Stored summary for {paper_id}")
                    # Add similar logic for other tools as needed
                except Exception as e:
                    print(f"[Context Passing Error]: {e}")
                # --- END CONTEXT PASSING LOGIC ---
            else:
                print("Invalid or missing action name. Skipping this step.")
                break
        else:
            print("MasterAgent reached maximum loops.")
        print("\n\n" + "="*20 + " GENERATING FINAL REPORT " + "="*20)
        final_synthesis_prompt = f"""
        You are a brilliant scientific analyst and writer. The project to answer '{state['original_request']}' is complete.
        Your task is to synthesize all the information gathered in the 'Project State' into a single, comprehensive report for the user.

        **Structure your report as follows:**
        1.  **Executive Summary:** A single paragraph that directly answers the user's core request.
        2.  **Key Findings:** A bulleted list of the most important insights discovered.
        3.  **Detailed Analysis:** A more in-depth explanation of the steps taken and the information found.
        4.  **Visualizations:** Reference any plots that were created and explain what they show.
        5.  **Conclusion & Limitations:** Conclude the report and mention any steps that could not be completed and why (e.g., "The requested table could not be extracted due to...").
        6.  **Plan:** Include the explicit step-by-step plan that was followed.

        **PROJECT STATE (The full history of your work):**
        ---
        {json.dumps(state, indent=2)}
        ---

        Write the final report in clear, professional language using Markdown formatting.
        """
        final_report = self.llm.invoke(final_synthesis_prompt).content
        return final_report

    def run_with_thoughts(self, user_query: str):
        """
        Runs the main loop and returns both the final report and the completed_steps (thought process).
        """
        state: Dict[str, Any] = {
            "original_request": user_query,
            "completed_steps": []
        }
        # --- NEW: Generate a step-by-step plan using the LLM ---
        plan_prompt = f"""
You are a world-class AI project planner. Your job is to break down the following user request into the SHORTEST, most EFFICIENT sequence of tool-based steps, using the available tools:
---\n{self.tool_manifest}\n---\nUser Request: '{user_query}'\n---\nRules:\n1. Each step must use a specific tool and specify its required arguments.\n2. Chain steps so outputs from one can be used as inputs for the next.\n3. Be explicit and efficient.\n4. Output ONLY a numbered list of steps, no extra text.\n"""
        plan_response = self.llm.invoke(plan_prompt).content
        # Ensure plan_response is always a list for robust iteration
        if not isinstance(plan_response, list):
            plan_response = [plan_response]
        plan_lines = []
        for item in plan_response:
            item_str = str(item)
            plan_lines.extend(item_str.split('\n'))
        plan = []
        for line in plan_lines:
            line = line.strip()
            if line and line[0].isdigit():
                step = line.split('.', 1)[-1].lstrip(" )-")
                plan.append(step)
        state["plan"] = plan
        print("--- Generated Plan ---")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")
        # --- END PLAN GENERATION ---
        for i in range(self.max_loops):
            print(f"\n{'='*20} MASTER AGENT LOOP {i+1}/{self.max_loops} {'='*20}")
            master_prompt = self._create_master_prompt(state)
            print("MasterAgent is thinking...")
            try:
                llm_response_str = self.llm.invoke(master_prompt).content
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "429" in err_str or "resourceexhausted" in err_str:
                    print("Quota hit or rate limited. Switching to local Llama 3 model...")
                    from langchain_community.chat_models import ChatOllama
                    self.llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.0)
                    llm_response_str = self.llm.invoke(master_prompt).content
                else:
                    raise
            try:
                response_str = llm_response_str
                if not isinstance(response_str, str):
                    if isinstance(response_str, list):
                        response_str = "\n".join([str(x) for x in response_str])
                    else:
                        response_str = str(response_str)
                decision = self._clean_and_parse_json(response_str)
                thought = decision.get("thought", "No thought provided.")
                action_json = decision.get("action", {})
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"MasterAgent failed to generate valid JSON. Error: {e}")
                print(f"Raw Response was: {llm_response_str}")
                break
            print(f"Thought: {thought}")
            if isinstance(action_json, dict):
                action_name = action_json.get("action")
                action_input = action_json.get("action_input")
            elif isinstance(action_json, str):
                action_name = action_json
                action_input = None
            else:
                action_name = None
                action_input = None
            if action_name == "Final Answer":
                print("MasterAgent has decided the task is complete. Moving to final report generation.")
                state["completed_steps"].append({
                    "step": i + 1,
                    "thought": thought,
                    "action_taken": "Final Answer",
                    "observation": action_input
                })
                break
            if isinstance(action_name, str) and action_name:
                print(f"MasterAgent directs worker to execute: {action_name}")
                worker_response = self.worker_agent.run_single_tool(action_name, action_input)
                state["completed_steps"].append({
                    "step": i + 1,
                    "thought": thought,
                    "action_taken": action_json,
                    "observation": worker_response
                })
                # --- CONTEXT PASSING LOGIC ---
                try:
                    if action_name == "table_extraction_tool":
                        import json
                        paper_id = action_input.get("paper_id") if isinstance(action_input, dict) else None
                        if worker_response and worker_response.strip().startswith("{"):
                            table_json = json.loads(worker_response)
                            if paper_id:
                                state[f"table_{paper_id}"] = table_json
                                print(f"[Context Passing] Stored table for {paper_id}")
                    if action_name == "paper_summarization_tool":
                        paper_id = action_input.get("paper_id") if isinstance(action_input, dict) else None
                        if paper_id:
                            state[f"summary_{paper_id}"] = worker_response
                            print(f"[Context Passing] Stored summary for {paper_id}")
                    # Add similar logic for other tools as needed
                except Exception as e:
                    print(f"[Context Passing Error]: {e}")
                # --- END CONTEXT PASSING LOGIC ---
            else:
                print("Invalid or missing action name. Skipping this step.")
                break
        else:
            print("MasterAgent reached maximum loops.")
        print("\n\n" + "="*20 + " GENERATING FINAL REPORT " + "="*20)
        final_synthesis_prompt = f"""
        You are a brilliant scientific analyst and writer. The project to answer '{state['original_request']}' is complete.
        Your task is to synthesize all the information gathered in the 'Project State' into a single, comprehensive report for the user.

        **Structure your report as follows:**
        1.  **Executive Summary:** A single paragraph that directly answers the user's core request.
        2.  **Key Findings:** A bulleted list of the most important insights discovered.
        3.  **Detailed Analysis:** A more in-depth explanation of the steps taken and the information found.
        4.  **Visualizations:** Reference any plots that were created and explain what they show.
        5.  **Conclusion & Limitations:** Conclude the report and mention any steps that could not be completed and why (e.g., "The requested table could not be extracted due to...").
        6.  **Plan:** Include the explicit step-by-step plan that was followed.

        **PROJECT STATE (The full history of your work):**
        ---
        {json.dumps(state, indent=2)}
        ---

        Write the final report in clear, professional language using Markdown formatting.
        """
        final_report = self.llm.invoke(final_synthesis_prompt).content
        return final_report, state["completed_steps"]

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