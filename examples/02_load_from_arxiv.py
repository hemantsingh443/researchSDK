from paper_agent.ingestor import Ingestor

def run_arxiv_example():
    """
    Demonstrates searching and loading papers from arXiv.
    """
    ingestor = Ingestor()


    query = "Graph Neural Networks for drug discovery"
    
    papers = ingestor.load_from_arxiv(query=query, max_results=2)

    print(f"\n\n{'='*20} PROCESSING COMPLETE {'='*20}")
    print(f"Successfully processed {len(papers)} papers from arXiv.")

    for i, paper in enumerate(papers):
        print(f"\n--- Paper {i+1} ---")
        print(f"ID: {paper.paper_id}")
        print(f"Title: {paper.title}")
        print(f"Authors: {[author.name for author in paper.authors]}")
        print(f"Text Length: {len(paper.full_text)} characters")

if __name__ == "__main__":
    run_arxiv_example()