from .agent import PaperAgent as ReActAgent 
from langchain_openai import ChatOpenAI
from typing import List, Dict

class PlanningAgent:
    """
    A planning agent that deconstructs a complex goal into a sequence of
    executable steps, then uses an execution agent to perform each step.
    """
    def __init__(self, react_agent: ReActAgent, planner_llm: ChatOpenAI):
        """
        Initializes the Planning Agent.

        Args:
            react_agent: An instance of our existing ReAct agent, which will act as the executor.
            planner_llm: An LLM to use for generating the plan.
        """
        self.executor_agent = react_agent
        self.planner_llm = planner_llm
        print("PlanningAgent initialized. The 'manager' is ready.")

    def _create_plan(self, user_query: str) -> List[str]:
        """Uses the LLM to create a step-by-step plan."""
        
        # Get the names of the available tools from the executor agent
        available_tools = ", ".join([t.name for t in self.executor_agent.tools])

        # Let's improve the planning prompt to be more direct.
        prompt = f"""
        Create a concise, step-by-step plan to accomplish the user's request.
        User request: "{user_query}"
        Available tools: [{available_tools}]
        
        The plan should be a numbered list of simple actions. Focus on efficiency.
        For example:
        1. Find 2 papers on 'Quantum Computing' using 'arxiv_paper_search_and_load_tool'.
        2. Find the paper_id for the first paper using 'paper_finder_tool' with the paper's title.
        3. Summarize the first paper using 'paper_summarization_tool' with its paper_id.
        4. Repeat steps 2 and 3 for the second paper.
        5. Synthesize the summaries into a final answer.
        
        Output ONLY the numbered list.
        """
        
        print("\n--- Generating Plan ---")
        response = self.planner_llm.invoke(prompt)
        plan_str = response.content
        if isinstance(plan_str, list):
            plan_str = "\n".join(str(item) for item in plan_str)
        plan = [step.strip().split('. ', 1)[1] for step in plan_str.split('\n') if step.strip() and step[0].isdigit()]
        
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