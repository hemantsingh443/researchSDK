from paper_agent.ingestor import Ingestor
from paper_agent.knowledge_base import KnowledgeBase
from paper_agent.extractor import Extractor
from paper_agent.agent import PaperAgent

def run_agent_example():
    """
    Demonstrates the full end-to-end RAG pipeline.
    Initializes all components and uses the agent to answer a question.
    """
    # --- Setup Phase ---
    # We only need the Extractor for its LLM client, so we initialize it here.
    # The Ingestor will no longer use it directly.
    print("--- Initializing Components ---")
    extractor = Extractor(api_type="local")
    kb = KnowledgeBase(db_path="./paper_db") # Use our existing database

    # --- Agent Initialization ---
    agent = PaperAgent(knowledge_base=kb, llm_extractor=extractor)
    
    # --- Query Phase ---
    user_question = "What are the challenges and limitations of LLM agents mentioned in the papers?"
    
    print(f"\n--- Running Agent Query ---")
    print(f"User Question: {user_question}")
    
    answer = agent.run_query(user_question)
    
    print("\n\n==================== AGENT'S FINAL ANSWER ====================")
    print(answer)
    print("==============================================================")


def first_time_setup():
    """
    This function should be run once to populate the knowledge base.
    """
    print("--- Running First-Time Setup: Populating Knowledge Base ---")
    ingestor = Ingestor()
    papers = ingestor.load_from_arxiv(query="Large Language Model Agents", max_results=3)
    
    kb = KnowledgeBase(db_path="./paper_db")
    kb.add_papers(papers)
    print("--- First-Time Setup Complete ---")


if __name__ == "__main__":
    # If you haven't populated the DB yet, run this script with an argument:
    # python examples/04_run_rag_agent.py setup
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        first_time_setup()
    else:
        # Check if the DB exists. A simple check for the directory is enough.
        import os
        if not os.path.exists("./paper_db"):
            print("Knowledge base not found. Please run the first-time setup:")
            print("python examples/04_run_rag_agent.py setup")
        else:
            run_agent_example()