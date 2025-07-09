import os
from paper_agent.ingestor import Ingestor

def run_example():
    ingestor = Ingestor()
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'attention.pdf')

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print(f"Loading paper from: {pdf_path}")
    paper = ingestor.load_from_pdf(pdf_path)

    print("\n--- Enriched Paper Object ---")
    print(paper)

    print("\n--- Extracted Details ---")
    print(f"Title: {paper.title if paper.title else 'N/A'}")
    print(f"Authors: {[author.name for author in paper.authors] if paper.authors else 'N/A'}")
    if paper.abstract:
        print(f"Abstract: {paper.abstract[:200]}...")
    else:
        print("Abstract: N/A")

if __name__ == "__main__":
    run_example()