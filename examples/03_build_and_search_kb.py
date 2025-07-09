from paper_agent.ingestor import Ingestor
from paper_agent.knowledge_base import KnowledgeBase
from pprint import pprint

def run_kb_example():
    """
    Demonstrates loading papers, adding them to a knowledge base,
    and performing a semantic search.
    """
    # === Step 1: Ingest Papers ===
    ingestor = Ingestor()
    query = "Large Language Model Agents"
    print(f"Loading papers from arXiv with query: '{query}'")
    papers = ingestor.load_from_arxiv(query=query, max_results=3)

    if not papers:
        print("Could not load any papers. Exiting.")
        return

    # === Step 2: Build Knowledge Base ===
    # Using a different path to keep it separate from other potential DBs
    kb = KnowledgeBase(db_path="./paper_db")
    kb.add_papers(papers)
    
    # === Step 3: Search the Knowledge Base ===
    search_query = "What are the challenges of LLM agents?"
    
    search_results = kb.search(query=search_query, n_results=3)

    print(f"\n\n{'='*20} SEARCH RESULTS {'='*20}")
    print(f"Query: {search_query}\n")

    # The result object from ChromaDB is a bit nested, let's unpack it.
    ids = search_results['ids'][0]
    distances = search_results['distances'][0]
    metadatas = search_results['metadatas'][0]
    documents = search_results['documents'][0]

    for i in range(len(ids)):
        print(f"--- Result {i+1} (Distance: {distances[i]:.4f}) ---")
        print(f"Source Paper: {metadatas[i]['title']}")
        print(f"Authors: {metadatas[i]['authors']}")
        print("\nRelevant Chunk:")
        print(documents[i])
        print("-" * 30)

if __name__ == "__main__":
    run_kb_example()