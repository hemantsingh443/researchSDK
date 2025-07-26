from .agent import PaperAgent as ReActAgent 
from typing import List, Dict
from langchain_community.chat_models import ChatOllama

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
        try:
            response = self.planner_llm.invoke(prompt)
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "429" in err_str or "resourceexhausted" in err_str:
                print("Quota hit or rate limited in planner. Switching to local Llama 3 model...")
                self.planner_llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.0)
                response = self.planner_llm.invoke(prompt)
            else:
                raise
        plan_str = response.content
        
        plan = []
        if isinstance(plan_str, str):
            lines = plan_str.split('\n')
        elif isinstance(plan_str, list):
            lines = plan_str
        else:
            lines = [str(plan_str)]
        for line in lines:
            if not isinstance(line, str):
                line = str(line)
            line = line.strip()
            if line and line[0].isdigit():
                split_line = line.split('.', 1)
                if len(split_line) > 1:
                    step = split_line[1].lstrip(" )-")
                    plan.append(step)
                else:
                    plan.append(line[1:].lstrip(" )-."))
        
        print("--- Plan Created ---")
        for i, step in enumerate(plan):
            print(f"{i+1}. {step}")
            
        return plan

    def run(self, user_query: str) -> str:
        """Creates a plan and executes it step by step."""
        plan = self._create_plan(user_query)
        
        if not plan:
            return "I could not create a plan to address your request."

        scratchpad: Dict[str, str] = {"original_request": user_query}
        print("\n--- Executing Plan ---")

        for i, step in enumerate(plan):
            print(f"\n[Executing Step {i+1}/{len(plan)}]: {step}")
            
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
            
            scratchpad[f"result_of_step_{i+1}"] = result
            print(f"[Step {i+1} Complete]. Scratchpad updated.")

        final_synthesis_prompt = f"""
You are a brilliant scientific analyst and writer. The project to answer '{user_query}' is complete.
Here is the full project scratchpad containing all the data, summaries, and artifacts you have generated:
---
{str(scratchpad)}
---

Your final task is to synthesize all of this information into a single, comprehensive, and insightful report for the user.

**Structure your report as follows:**
1.  **Executive Summary:** A single paragraph that directly answers the user's core request.
2.  **Key Findings:** A bulleted list of the most important insights you discovered. This could include which model performed best, a key difference between two papers, or a surprising result from the data.
3.  **Detailed Analysis:** A more in-depth explanation, incorporating the summaries and comparisons you made.
4.  **Visualizations:** Reference any plots you created (e.g., "A visual comparison of performance is available in the chart 'rita_performance.png'.").
5.  **Further Questions:** Based on your analysis, propose one or two interesting follow-up questions for future research.

Write the report in clear, professional language.
"""
        
        print("\n--- Generating Final Synthesis ---")
        try:
            final_answer = self.planner_llm.invoke(final_synthesis_prompt).content
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "429" in err_str or "resourceexhausted" in err_str:
                print("Quota hit or rate limited in final synthesis. Switching to local Llama 3 model...")
                self.planner_llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0.0)
                final_answer = self.planner_llm.invoke(final_synthesis_prompt).content
            else:
                raise
        if isinstance(final_answer, list):
            final_answer = "\n".join(
                str(item) if not isinstance(item, str) else item
                for item in final_answer
            )
        return final_answer