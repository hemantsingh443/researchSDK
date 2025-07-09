import fitz
import arxiv
from .structures import Paper, Author
from .extractor import Extractor

class Ingestor:
    def __init__(self):
        self.extractor = Extractor(api_type="local")
        print("Ingestor initialized with a LOCAL AI Extractor.")

    def load_from_pdf(self, file_path: str, source_metadata: dict = None) -> Paper:
        """
        Loads a paper from a local PDF file and extracts metadata.
        Prioritizes provided source_metadata over LLM extraction.
        """
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            print(f"Error opening or reading PDF: {e}")
            raise

        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text() + "\n"
        doc.close()

        paper_obj = Paper(paper_id=file_path, full_text=full_text)

   
        if source_metadata:
            print("Using pre-fetched metadata from API.")
            paper_obj.title = source_metadata.get("title")
            paper_obj.abstract = source_metadata.get("abstract")
            paper_obj.authors = source_metadata.get("authors", [])
            paper_obj.paper_id = source_metadata.get("paper_id", file_path)
        else:
            print("No source metadata provided. Falling back to LLM extraction.")
         
            paper_obj = self.extractor.extract_metadata(paper_obj)
        
        return paper_obj

    def load_from_arxiv(self, query: str, max_results: int = 3) -> list[Paper]:
        """
        Searches arXiv, downloads PDFs, and processes them using API metadata.
        """
        print(f"Searching arXiv for '{query}'...")
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
        
        if not results:
            print("No papers found on arXiv for this query.")
            return []

        papers = []
        for result in results:
            print(f"\n--- Processing arXiv paper: {result.title} ---")
            try:
                pdf_path = result.download_pdf()
                print(f"Downloaded to: {pdf_path}")
                
           
                arxiv_metadata = {
                    "paper_id": result.entry_id,
                    "title": result.title,
                    "abstract": result.summary.replace('\n', ' '),
                    "authors": [Author(name=str(a)) for a in result.authors]
                }
                
                paper = self.load_from_pdf(pdf_path, source_metadata=arxiv_metadata)
                papers.append(paper)

            except Exception as e:
                print(f"!! Failed to process paper {result.entry_id}. Error: {e}")
        
        return papers