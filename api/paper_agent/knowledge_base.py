import chromadb
from chromadb.utils import embedding_functions
from .structures import Paper
from typing import List, Any
from tqdm import tqdm
from neo4j import GraphDatabase

class KnowledgeBase:
    """
    Manages both a vector store (ChromaDB) for semantic search and a
    graph database (Neo4j) for structured relationships.
    """
    def __init__(self, db_path: str = "./paper_db", neo4j_uri = "neo4j://172.20.128.55:7687", neo4j_user="neo4j", neo4j_password="password"):
        """
        Initializes both ChromaDB and the Neo4j driver.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(name="papers", embedding_function=self.embedding_function) 
        
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_driver.verify_connectivity()
            print(f"KnowledgeBase initialized. Vector DB at: {db_path}. Graph DB at: {neo4j_uri}.")
        except Exception as e:
            print(f"CRITICAL: Could not connect to Neo4j database. Please ensure it is running. Error: {e}")
            self.neo4j_driver = None

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def _add_paper_to_graph(self, paper: Paper):
        """A private helper method to add a single paper and its authors to Neo4j."""
        if not self.neo4j_driver:
            return

        with self.neo4j_driver.session() as session:
            session.run("""
                MERGE (p:Paper {id: $paper_id})
                ON CREATE SET p.title = $title
                """, paper_id=paper.paper_id, title=paper.title)

            for author in paper.authors:
                session.run("""
                    MERGE (a:Author {name: $author_name})
                    WITH a
                    MATCH (p:Paper {id: $paper_id})
                    MERGE (a)-[:AUTHORED]->(p)
                    """, author_name=author.name, paper_id=paper.paper_id)

            if hasattr(paper, 'citations') and paper.citations:
                for cited_paper_title in paper.citations:
                    session.run("MERGE (cp:Paper {title: $title})", title=cited_paper_title)
                    session.run("""
                        MATCH (p1:Paper {id: $paper_id})
                        MATCH (p2:Paper {title: $cited_title})
                        MERGE (p1)-[:CITES]->(p2)
                        """, paper_id=paper.paper_id, cited_title=cited_paper_title)

    def _split_text_into_chunks(self, markdown_text: str, chunk_size: int = 2000) -> List[str]:
        """
        Splits Markdown text by headers, creating semantically relevant chunks.
        If a chunk is too large, further splits by sentences.
        """
        import re
        chunks = re.split(r'(?=\n##\s)', markdown_text)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                sub_chunks = re.split(r'(?<=\.)\s', chunk)
                final_chunks.extend([s.strip() for s in sub_chunks if s.strip()])
            elif chunk.strip():
                final_chunks.append(chunk.strip())
        return final_chunks

    def add_papers(self, papers: List[Paper]):
        """
        Adds papers to both the vector store (with caching) and the
        graph database (which handles its own duplicates with MERGE).
        """
        print(f"Attempting to process {len(papers)} paper(s) for the knowledge base...")
        
        existing_vector_ids = set(self.collection.get(include=[])['ids'])
        papers_to_add_to_vector_store = []
        for paper in papers:
            test_chunk_id = f"{paper.paper_id}_chunk_0"
            if test_chunk_id in existing_vector_ids:
                print(f"-> Paper '{paper.title}' already exists in the vector KB. Skipping vector add.")
            else:
                papers_to_add_to_vector_store.append(paper)
        
        if papers_to_add_to_vector_store:
            all_chunks, all_metadatas, all_ids = [], [], []
            for paper in tqdm(papers_to_add_to_vector_store, desc="Processing for vector store"):
                text_to_index = f"Title: {paper.title}\n\nAbstract: {paper.abstract}\n\n{paper.full_text}"
                chunks = self._split_text_into_chunks(text_to_index)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{paper.paper_id}_chunk_{i}"
                    metadata = {
                        "paper_id": paper.paper_id or "unknown_id",
                        "title": paper.title or "Title Not Available",
                        "authors": ", ".join([a.name for a in paper.authors]) if paper.authors else "Authors Not Available"
                    }
                    all_chunks.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)

            if all_ids:
                self.collection.add(documents=all_chunks, metadatas=all_metadatas, ids=all_ids)
                print(f"Successfully added {len(all_ids)} new text chunks to the vector store.")
        else:
            print("Vector knowledge base is already up-to-date.")

        if self.neo4j_driver:
            print("Updating graph database...")
            for paper in tqdm(papers, desc="Processing for graph store"):
                self._add_paper_to_graph(paper)
            print("Graph database update complete.")

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