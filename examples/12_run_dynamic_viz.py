from paper_agent.agent import PaperAgent as WorkerAgent
from paper_agent.master_agent import MasterAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import os

def run_viz_test():
    # --- Setup ---
    worker_agent = WorkerAgent(db_path="./paper_db", llm_provider="google")
    master_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)
    master_agent = MasterAgent(worker_agent=worker_agent, llm=master_llm)

    # --- TEST QUERY ---
    # This query implies two different kinds of visualizations are needed.
    viz_query = """
    I need to understand the performance of models in the 'Attention is All You Need' paper.
    Please do the following:
    1. Extract the main results table that includes BLEU scores and Training Cost.
    2. From that data, create a visualization to **compare the BLEU scores** of the different models. Save it as 'bleu_comparison.png'.
    3. From the same data, create another visualization to show the **relationship between Training Cost and BLEU score**. Save it as 'cost_vs_performance.png'.
    4. Provide a final answer summarizing what the two charts show.
    """

    final_response = master_agent.run(viz_query)

    print("\n\n" + "="*60)
    print("DYNAMIC VIZ TEST: MASTER AGENT'S FINAL REPORT")
    print("="*60)
    print(final_response)

    if os.path.exists("bleu_comparison.png") and os.path.exists("cost_vs_performance.png"):
        print("\nSUCCESS: Both visualization files were created successfully!")
    else:
        print("\nFAILURE: One or more plot files were not created.")

if __name__ == "__main__":
    if not os.path.exists("./paper_db"):
        print("Knowledge base not found. Please run 'add_local_paper.py' first.")
        sys.exit(1)
    
    run_viz_test()