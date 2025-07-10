from paper_agent.agent import PaperAgent
import sys
import os

def run_summarization_example():
    agent = PaperAgent(db_path="./paper_db")

    # This query requires the agent to first find the paper, then summarize it.
    query = "Please provide a summary of the 'Attention is All You Need' paper."
    
    response = agent.run(query)
    
    print("\n\n" + "="*50)
    print("AGENT'S FINAL ANSWER")
    print("="*50)
    print(response['output'])

if __name__ == "__main__":
    if not os.path.exists("./paper_db"):
        print("Knowledge base not found. Please run a setup script first.")
        sys.exit(1)
    
    run_summarization_example()