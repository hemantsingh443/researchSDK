import fitz
import arxiv
from .structures import Paper, Author
from typing import Optional, Dict, Any
import shutil
import os

from .extractor import Extractor 

from .grobid_client import GrobidClient
from .paper_parser import parse_grobid_tei, to_markdown

class Ingestor:
    def __init__(self):
        """Initializes the Ingestor with the Grobid client and a fallback LLM Extractor."""
        self.extractor = Extractor(api_type="local")
        self.grobid_client = GrobidClient()
        print("Ingestor initialized with a Grobid-powered parsing pipeline.")

    def _parse_pdf_to_structured_text(self, file_path: str) -> str:
        """
        The core parsing pipeline: PDF -> Grobid XML -> Markdown.
        Returns the final Markdown text. Falls back to basic extraction on failure.
        """
        xml_content = self.grobid_client.process_pdf(file_path)
        if not xml_content:
            print(f"Grobid processing failed for '{file_path}'. Falling back to basic text extraction.")
            try:
                doc = fitz.open(file_path)
                full_text = "\n".join(page.get_text() for page in doc)
                doc.close()
                return full_text
            except Exception as e:
                print(f"Fallback basic extraction failed: {e}")
                return "" 
        temp_xml_path = file_path + ".tei.xml"
        try:
            with open(temp_xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            parsed_data = parse_grobid_tei(temp_xml_path)

            full_markdown_text = to_markdown(parsed_data) 
            return full_markdown_text
        
        except Exception as e:
            print(f"Failed to parse Grobid XML or convert to Markdown: {e}")
            return "" 
            
        finally:
            if os.path.exists(temp_xml_path):
                os.remove(temp_xml_path)

    def load_from_pdf(self, file_path: str, source_metadata: Optional[Dict[str, Any]] = None) -> Paper:
        """
        Loads a paper from a local PDF file, using Grobid for primary extraction.
        """
        structured_full_text = self._parse_pdf_to_structured_text(file_path)

        paper_obj = Paper(paper_id=file_path, full_text=structured_full_text)

        if source_metadata:
            print("Using pre-fetched metadata from API.")
            paper_obj.title = source_metadata.get("title")
            paper_obj.abstract = source_metadata.get("abstract")
            paper_obj.authors = source_metadata.get("authors", [])
            paper_obj.paper_id = source_metadata.get("paper_id", file_path)
        else:
            print("No source metadata. Using LLM Extractor on Grobid-parsed Markdown for validation.")
            paper_obj = self.extractor.extract_metadata(paper_obj)

        paper_obj.citations = self.extractor.extract_citations(structured_full_text)
        return paper_obj

    def load_from_arxiv(self, query: str, max_results: int = 3) -> list[Paper]:
        """
        Searches arXiv, downloads PDFs, and processes them using the Grobid pipeline.
        This method does not need to change.
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
                
                artifacts_dir = "artifacts"
                if not os.path.exists(artifacts_dir):
                    os.makedirs(artifacts_dir)
                if pdf_path and os.path.exists(pdf_path):
                    dest_path = os.path.join(artifacts_dir, os.path.basename(pdf_path))
                    try:
                        shutil.copy(pdf_path, dest_path)
                        print(f"Copied PDF to artifacts: {dest_path}")
                    except Exception as copy_exc:
                        print(f"!! Failed to copy PDF to artifacts: {copy_exc}")
                
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