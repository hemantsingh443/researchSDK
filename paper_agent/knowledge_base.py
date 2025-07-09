import chromadb
from chromadb.utils import embedding_functions
from .structures import Paper
from typing import List, Any
from tqdm import tqdm
import re

class KnowledgeBase:
    """
    Manages a collection of papers and provides semantic search capabilities
    using a vector database (ChromaDB).
    """
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initializes the KnowledgeBase.

        Args:
            db_path: Path to the directory where the ChromaDB database will be stored.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="papers",
            embedding_function=self.embedding_function  # type: ignore
        )
        print(f"KnowledgeBase initialized. Using ChromaDB at: {db_path}")

    def _split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Splits a long text into smaller, overlapping chunks.
        """
        # Simple splitting by sentences or paragraphs is more effective than by character count
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def add_papers(self, papers: List[Paper]):
        """
        Adds a list of papers to the knowledge base, skipping any that
        already exist in the collection.
        """
        print(f"Attempting to add {len(papers)} paper(s) to the knowledge base...")
        
        # --- NEW CACHING LOGIC ---
        existing_ids = set(self.collection.get(include=[])['ids'])
        
        papers_to_add = []
        for paper in papers:
            # Check if a chunk from this paper already exists. A simple but effective check.
            test_chunk_id = f"{paper.paper_id}_chunk_0"
            if test_chunk_id in existing_ids:
                print(f"-> Skipping paper '{paper.title}' as it already exists in the KB.")
            else:
                papers_to_add.append(paper)
        
        if not papers_to_add:
            print("All papers already in the knowledge base. Nothing to add.")
            return
        # --- END NEW CACHING LOGIC ---

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for paper in tqdm(papers_to_add, desc="Processing new papers"):
            # use both abstract and full_text for a comprehensive search
            text_to_index = f"Title: {paper.title}\n\nAbstract: {paper.abstract}\n\n{paper.full_text}"
            chunks = self._split_text_into_chunks(text_to_index)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{paper.paper_id}_chunk_{i}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": ", ".join([a.name for a in paper.authors])
                })
                all_ids.append(chunk_id)

        if all_ids:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"Successfully added {len(all_ids)} new text chunks to the vector store.")
        else:
            print("No new text chunks to add.")

    def search(self, query: str, n_results: int = 5) -> Any:
        """
        Performs a semantic search on the knowledge base.

        Args:
            query: The natural language query.
            n_results: The number of results to return.

        Returns:
            A dictionary containing the search results.
        """
        print(f"Searching for: '{query}'...")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results