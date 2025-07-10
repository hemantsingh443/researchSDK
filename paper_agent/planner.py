from .agent import PaperAgent as ReActAgent 
from langchain_openai import ChatOpenAI
from typing import List, Dict

class PlanningAgent:
    """
    A planning agent that deconstructs a complex goal into a sequence of
    executable steps, then uses an execution agent to perform each step.
    Supports both Google Gemini (ChatGoogleGenerativeAI) and local Ollama (ChatOpenAI) LLMs.
    Pass the correct LLM instance as planner_llm.
    """
    def __init__(self, react_agent: ReActAgent, planner_llm):
        """
        Initializes the Planning Agent.

        Args:
            react_agent: An instance of our existing ReAct agent, which will act as the executor.
            planner_llm: An LLM to use for generating the plan. Can be ChatGoogleGenerativeAI or ChatOpenAI.
        """
        self.executor_agent = react_agent
        self.planner_llm = planner_llm
        print("PlanningAgent initialized. The 'manager' is ready.")

    def _create_plan(self, user_query: str) -> List[str]:
        """Uses the LLM to create a step-by-step plan by first reflecting on the available tools."""
        
        # --- THE KEY UPGRADE: Create a detailed tool manifest for the planner ---
        tool_manifest = ""
        for tool in self.executor_agent.tools:
            tool_manifest += f"Tool Name: {tool.name}\n"
            tool_manifest += f"Tool Description: {tool.description}\n"
            tool_manifest += f"Tool Arguments: {tool.args}\n\n"

        prompt = f"""
        You are a hyper-efficient AI project planner. Your job is to create the SHORTEST POSSIBLE, most EFFICIENT step-by-step plan to accomplish the user's request.

        **Analyze the following tools very carefully:**
        ---
        {tool_manifest}
        ---

        **Based on your analysis of the tools, create a plan to resolve the following user request:**
        ---
        User Request: "{user_query}"
        ---

        **Rules for creating the plan:**
        1.  **Be Efficient:** Do not use a tool if a more direct tool exists. For example, to find authors, use the `graph_query_tool` directly instead of finding and summarizing papers first.
        2.  **Chain Inputs:** Think about the outputs of each step. If a later step needs a 'paper_id', a previous step must use a tool like `paper_finder_tool` that provides it.
        3.  **Simple Steps:** Each step in the plan must be a simple, single instruction for the worker agent.

        Your final output must be ONLY the numbered list of steps. Do not add any other text.
        """
        
        print("\n--- Generating Plan (with reflection) ---")
        response = self.planner_llm.invoke(prompt)
        plan_str = response.content
        
        # More robust parsing of the plan
        plan = []
        for line in plan_str.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove the leading number and period
                plan.append(line.split('. ', 1)[1])
        
        print("--- Plan Created ---")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")
            
        return plan

    def run(self, user_query: str) -> str:
        """Creates a plan and executes it step by step."""
        plan = self._create_plan(user_query)
        
        if not plan:
            return "I could not create a plan to address your request."

        # --- THE STATEFUL SCRATCHPAD ---
        scratchpad: Dict[str, str] = {"original_request": user_query}
        print("\n--- Executing Plan ---")

        for i, step in enumerate(plan):
            print(f"\n[Executing Step {i+1}/{len(plan)}]: {step}")
            
            # Inject the current state (the scratchpad) into the prompt for the worker.
            # This gives the worker memory of what has been learned so far.
            worker_prompt = f"""
            You are a diligent worker agent. Here is the current state of our project:
            ---
            {str(scratchpad)}
            ---
            Your single, specific task is to: {step}
            
            Based on the project state, please execute this task.
            """
            
            response = self.executor_agent.run(worker_prompt)
            result = response.get('output', 'Step completed with no textual output.')
            
            # Update the scratchpad with the result of the current step.
            scratchpad[f"result_of_step_{i+1}"] = result
            print(f"[Step {i+1} Complete]. Scratchpad updated.")

        # Final synthesis step, now with a rich scratchpad of all results.
        final_synthesis_prompt = f"""
        The plan is complete. The original request was: '{user_query}'
        
        Here is the full project scratchpad containing all results:
        ---
        {str(scratchpad)}
        ---
        
        Synthesize all of this information into a final, comprehensive answer for the user.
        """
        
        print("\n--- Generating Final Synthesis ---")
        final_answer = self.planner_llm.invoke(final_synthesis_prompt).content
        if isinstance(final_answer, list):
            final_answer = "\n".join(
                str(item) if not isinstance(item, str) else item
                for item in final_answer
            )
        return final_answer