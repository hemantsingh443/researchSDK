from paper_agent.agent import PaperAgent as ReActAgent
from paper_agent.planner import PlanningAgent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import os

def run_planning_example():
    # 1. Initialize the "worker bee"  existing ReAct agent
    react_agent = ReActAgent(db_path="./paper_db", llm_provider="google")

    # 2. Initialize the "manager" LLM - use Google Gemini
    planner_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

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