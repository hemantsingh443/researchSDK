# examples/08_run_planning_agent.py

from paper_agent.agent import PaperAgent as ReActAgent
from paper_agent.planner import PlanningAgent
from langchain_openai import ChatOpenAI
import sys
import os

def run_planning_example():
    # 1. Initialize the "worker bee" - our existing ReAct agent
    react_agent = ReActAgent(db_path="./paper_db")

    # 2. Initialize the "manager" LLM - this can be the same model
    planner_llm = ChatOpenAI(
        base_url='http://localhost:11434/v1',
        api_key=None,
        model='llama3:8b-instruct-q4_K_M',
        temperature=0.0
    )

    # 3. Create the master Planning Agent
    planning_agent = PlanningAgent(react_agent=react_agent, planner_llm=planner_llm)

    # 4. Define a complex, multi-step goal
    complex_query = """
    Create a brief literature review on 'Mixture of Experts' for me.
    First, find 2 recent papers on the topic and add them to the library.
    Then, summarize each of those two papers.
    Finally, write a one-paragraph synthesis of the summaries that explains what Mixture of Experts is.
    """

    # 5. Run the planner
    final_response = planning_agent.run(complex_query)

    print("\n\n" + "="*50)
    print("PLANNING AGENT'S FINAL, SYNTHESIZED ANSWER")
    print("="*50)
    print(final_response)

if __name__ == "__main__":
    if not os.path.exists("./paper_db"):
        print("Knowledge base not found. Please run a setup script first.")
        sys.exit(1)
    
    run_planning_example()