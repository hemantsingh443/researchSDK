from paper_agent.agent import PaperAgent as WorkerAgent
from paper_agent.master_agent import MasterAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import os

def run_master_agent_test():
    # 1. Initialize the Worker Agent (our original ReAct agent)
    worker_agent = WorkerAgent(db_path="./paper_db", llm_provider="google")

    # 2. Initialize the LLM for the Master Agent's brain
    master_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)

    # 3. Create the Master Agent
    master_agent = MasterAgent(worker_agent=worker_agent, llm=master_llm)

    # 4. Define the complex query that failed before
    query = """
    First, find the paper 'Attention is All You Need'.
    Then, use the table_extraction_tool to extract the table showing translation quality (BLEU scores) from that paper.
    Finally, use the plot_generation_tool to create a bar chart of the BLEU scores, title it 'Translation Quality (BLEU)', and save it as 'bleu_scores.png'.
    """

    # 5. Run the Master Agent
    final_response = master_agent.run(query)

    print("\n\n" + "="*50)
    print("MASTER AGENT'S FINAL ANSWER")
    print("="*50)
    print(final_response)
    if os.path.exists("bleu_scores.png"):
        print("\nSUCCESS: 'bleu_scores.png' was created successfully!")

if __name__ == "__main__":
    if not os.path.exists("./paper_db"):
        print("Knowledge base not found. Please run 'add_local_paper.py' first.")
        sys.exit(1)
    
    run_master_agent_test()