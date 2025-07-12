from paper_agent.agent import PaperAgent as WorkerAgent
from paper_agent.master_agent import MasterAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import os

def run_full_workflow_test():
    # --- Setup ---
    worker_agent = WorkerAgent(db_path="./paper_db", llm_provider="google")
    master_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.0) # Using Pro for best planning
    master_agent = MasterAgent(worker_agent=worker_agent, llm=master_llm)

    # --- The Final, Full Workflow Query ---
    final_query = """
    I need a complete analysis and data export for the 'Attention is All You Need' paper.

    Please perform the following steps:
    1.  Confirm the paper is in the library and get its ID.
    2.  Extract the main results table showing BLEU scores and Training Cost.
    3.  From that data, create an insightful visualization showing the relationship between cost and performance. Save it as 'final_chart.png'.
    4.  Also, save the raw extracted table data to a file named 'attention_results.csv'.
    5.  Finally, write a comprehensive report summarizing the paper's performance and mentioning the chart and CSV file you created.
    """

    # --- Run the Master Agent ---
    final_response = master_agent.run(final_query)

    print("\n\n" + "="*60)
    print("FULL WORKFLOW: MASTER AGENT'S FINAL REPORT")
    print("="*60)
    print(final_response)

    if os.path.exists("final_chart.png"):
        print("\nSUCCESS: 'final_chart.png' was created successfully!")
    else:
        print("\nFAILURE: The plot file 'final_chart.png' was not created.")

    if os.path.exists("attention_results.csv"):
        print("SUCCESS: 'attention_results.csv' was created successfully!")
    else:
        print("FAILURE: The CSV file 'attention_results.csv' was not created.")


if __name__ == "__main__":
    if not os.path.exists("./paper_db"):
        print("Knowledge base not found. Please run 'add_local_paper.py' first.")
        sys.exit(1)
    
    run_full_workflow_test()