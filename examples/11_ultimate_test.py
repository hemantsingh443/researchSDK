
from paper_agent.agent import PaperAgent as WorkerAgent
from paper_agent.master_agent import MasterAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import os

def run_ultimate_test():
    # --- Setup ---
    # Ensure all components are initialized with the latest provider
    worker_agent = WorkerAgent(db_path="./paper_db", llm_provider="google")
    master_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    master_agent = MasterAgent(worker_agent=worker_agent, llm=master_llm)

    # --- The Ultimate Query ---
    ultimate_query = """
    I'm a researcher interested in the latest advancements in language models.
    I've heard about a very new paper called 'RITA: Group Attention is All You Need for Timeseries Analytics'.

    Please do the following:
    1. First, check if you already have this 'RITA' paper. If not, find it on arXiv and add it to my library.
    2. Provide a summary of the 'RITA' paper, focusing on what problem it solves and its main contribution.
    3. Compare 'RITA' to the foundational 'Attention is All You Need' paper. What is the core difference in their approach, based on the problem they are trying to solve? Use the graph database to see if there's any direct connection.
    4. The 'RITA' paper should have a table comparing its performance to other models on the ETT dataset. Extract this table.
    5. Finally, create a bar chart from that extracted table, title it 'RITA vs Baselines on ETT Dataset', and save it as 'rita_performance.png'.
    """

    # --- Run the Master Agent ---
    final_response = master_agent.run(ultimate_query)

    print("\n\n" + "="*60)
    print("ULTIMATE TEST: MASTER AGENT'S FINAL SYNTHESIZED REPORT")
    print("="*60)
    print(final_response)

    if os.path.exists("rita_performance.png"):
        print("\nSUCCESS: 'rita_performance.png' was created successfully!")
    else:
        print("\nFAILURE: The plot file 'rita_performance.png' was not created.")


if __name__ == "__main__":
    # Ensure the foundational 'Attention' paper is loaded for comparison
    if not os.path.exists("./paper_db"):
        print("Knowledge base not found. Please run 'add_local_paper.py' first.")
        sys.exit(1)
    
    # We can add a check to make sure 'Attention' is there
    # For now, we assume it is.
    
    run_ultimate_test()