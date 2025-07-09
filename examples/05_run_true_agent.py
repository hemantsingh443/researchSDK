import sys
import os
from paper_agent.agent import PaperAgent
from paper_agent.ingestor import Ingestor
from paper_agent.knowledge_base import KnowledgeBase

def run_agentic_example():
    # Initialize our powerful new agent
    agent = PaperAgent(db_path="./paper_db")

    # --- Use Case 1: The agent decides to use the KB ---
    print("\n\n" + "="*50)
    print("USE CASE 1: Answering a question from the existing knowledge base.")
    print("="*50)
    query1 = "What are the limitations of LLM agents based on the papers?"
    response1 = agent.run(query1)
    print("\n--- Final Answer ---")
    print(response1.get('output', 'No output received.'))

    # --- Use Case 2: The agent decides it needs to load new papers ---
    print("\n\n" + "="*50)
    print("USE CASE 2: Loading new papers into the knowledge base.")
    print("="*50)
    query2 = "Find me some recent papers on 'Mixture of Experts' and add them to my library."
    response2 = agent.run(query2)
    print("\n--- Final Answer ---")
    print(response2.get('output', 'No output received.'))
    
    # --- Use Case 3: A follow-up question that uses the new papers ---
    print("\n\n" + "="*50)
    print("USE CASE 3: A follow-up question using the newly added papers.")
    print("="*50)
    query3 = "What is 'Mixture of Experts' based on the new papers I just added?"
    response3 = agent.run(query3)
    print("\n--- Final Answer ---")
    print(response3.get('output', 'No output received.'))

    # --- Use Case 4: The agent decides to use the web search tool ---
    print("\n\n" + "="*50)
    print("USE CASE 4: Answering a general knowledge question.")
    print("="*50)
    query4 = "Who is the CEO of NVIDIA?"
    response4 = agent.run(query4)
    print("\n--- Final Answer ---")
    print(response4.get('output', 'No output received.'))

def run_setup():
    """
    This function should be run once to populate the knowledge base.
    It downloads some initial papers to make the agent useful.
    """
    print("--- Running First-Time Setup: Populating Knowledge Base ---")
    ingestor = Ingestor()
    kb = KnowledgeBase(db_path="./paper_db")
    
    # Let's add the papers we know the agent will be tested on
    print("\nDownloading initial set of papers on 'LLM Agents'...")
    papers1 = ingestor.load_from_arxiv(query="Large Language Model Agents", max_results=3)
    kb.add_papers(papers1)
    
    # We can add more topics to make the KB more diverse from the start
    print("\nDownloading initial set of papers on 'Mixture of Experts'...")
    papers2 = ingestor.load_from_arxiv(query="Mixture of Experts", max_results=3)
    kb.add_papers(papers2)

    print("\n--- First-Time Setup Complete ---")
    print("The 'paper_db' directory has been created and populated. You can now run the agent normally.")


if __name__ == "__main__":
    # Check for a 'setup' argument in the command line
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        run_setup()
    else:
        # If not running setup, check if the database exists before starting the agent
        if not os.path.exists("./paper_db"):
            print("Knowledge base not found. Please run the first-time setup:")
            # Updated instruction
            print("python examples/05_run_true_agent.py setup")
            sys.exit(1)
        
        run_agentic_example()