import os
import logging
from typing import List, Dict, Any, Optional, Union

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from tqdm import tqdm

from .structures import Paper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Manages both a vector store (ChromaDB) for semantic search and a
    graph database (Neo4j) for structured relationships.
    
    Supports both local and Docker-based ChromaDB instances, with configurable
    embedding models and persistence options.
    """
    
    def __init__(
        self,
        db_path: str = "./paper_db",
        chroma_host: Optional[str] = None,
        chroma_port: int = 8000,
        collection_name: str = "papers",
        embedding_model: str = "all-MiniLM-L6-v2",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        persist: bool = True
    ):
        """Initialize the KnowledgeBase with ChromaDB and Neo4j connections.
        
        Args:
            db_path: Path to store the ChromaDB data (for local mode)
            chroma_host: Hostname or IP of ChromaDB server (if using Docker)
            chroma_port: Port of ChromaDB server
            collection_name: Name of the Chroma collection to use
            embedding_model: Name of the SentenceTransformer model to use
            neo4j_uri: URI of the Neo4j database
            neo4j_user: Username for Neo4j authentication
            neo4j_password: Password for Neo4j authentication
            persist: Whether to persist the ChromaDB to disk (local mode only)
        """
        self.collection_name = collection_name
        self._init_chroma(chroma_host, chroma_port, db_path, embedding_model, persist)
        self._init_neo4j(neo4j_uri, neo4j_user, neo4j_password)
        
        logger.info(
            f"KnowledgeBase initialized. Vector DB: {'Docker' if chroma_host else 'Local'}. "
            f"Graph DB: {neo4j_uri}."
        )
    
    def _init_chroma(
        self,
        host: Optional[str],
        port: int,
        db_path: str,
        embedding_model: str,
        persist: bool
    ) -> None:
        """Initialize the ChromaDB client and collection."""
        try:
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            
            # Configure Chroma client based on whether we're using Docker or local
            if host:
                # Docker/remote mode
                self.client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    ssl=False,
                    headers={"Authorization": f"Bearer {os.getenv('CHROMA_SERVER_AUTH_CREDENTIALS', '')}"}
                    if os.getenv('CHROMA_SERVER_AUTH_CREDENTIALS') else None
                )
            else:
                # Local mode - using the new persistent client
                self.client = chromadb.PersistentClient(
                    path=db_path if persist else None,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False
                    )
                )
            
            # Get or create the collection with the new API
            try:
                # Try to get the collection first
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            except Exception as e:
                # If collection doesn't exist, create it
                if "does not exist" in str(e).lower():
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"hnsw:space": "cosine"}  # Optional: specify the distance metric
                    )
                else:
                    raise
            
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _init_neo4j(self, uri: str, user: str, password: str) -> None:
        """Initialize the Neo4j driver with error handling."""
        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            self.neo4j_driver.verify_connectivity()
        except neo4j_exceptions.ServiceUnavailable as e:
            logger.warning(f"Neo4j service unavailable at {uri}. Graph features will be disabled. Error: {e}")
            self.neo4j_driver = None
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            self.neo4j_driver = None

    def close(self) -> None:
        """Close all database connections."""
        # Close Neo4j driver if it exists
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
                logger.info("Neo4j connection closed successfully.")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")
        
        # Close ChromaDB client if it exists and is persistent
        if hasattr(self, 'client') and not isinstance(self.client, chromadb.HttpClient):
            try:
                self.client.persist()
                logger.info("ChromaDB changes persisted to disk.")
            except Exception as e:
                logger.error(f"Error persisting ChromaDB: {e}")

    def _add_paper_to_graph(self, paper: Paper) -> bool:
        """Add a single paper and its authors to Neo4j.
        
        Args:
            paper: The paper to add to the graph
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not self.neo4j_driver:
            logger.warning("Cannot add paper to graph: Neo4j driver not initialized")
            return False
            
        try:
            with self.neo4j_driver.session() as session:
                # Add paper node
                session.execute_write(
                    lambda tx: tx.run(
                        """
                        MERGE (p:Paper {id: $paper_id})
                        ON CREATE SET p.title = $title,
                                      p.abstract = $abstract,
                                      p.published = $published
                        ON MATCH SET p.title = $title,
                                     p.abstract = $abstract,
                                     p.published = $published
                        """,
                        paper_id=paper.paper_id,
                        title=paper.title,
                        abstract=getattr(paper, 'abstract', ''),
                        published=getattr(paper, 'published', '')
                    )
                )

                # Add authors and relationships
                for author in paper.authors:
                    session.execute_write(
                        lambda tx: tx.run(
                            """
                            MERGE (a:Author {name: $author_name})
                            WITH a
                            MATCH (p:Paper {id: $paper_id})
                            MERGE (a)-[:AUTHORED]->(p)
                            """,
                            author_name=author.name,
                            paper_id=paper.paper_id
                        )
                    )

                # Add citations if available
                if hasattr(paper, 'citations') and paper.citations:
                    for cited_paper_title in paper.citations:
                        session.execute_write(
                            lambda tx: tx.run(
                                """
                                MERGE (p1:Paper {id: $paper_id})
                                MERGE (p2:Paper {title: $cited_title})
                                MERGE (p1)-[r:CITES]->(p2)
                                """,
                                paper_id=paper.paper_id,
                                cited_title=cited_paper_title
                            )
                        )
            
            logger.debug(f"Successfully added paper to graph: {paper.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper to graph: {e}")
            return False

    def _split_text_into_chunks(
        self,
        markdown_text: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Split Markdown text into chunks with metadata.
        
        Args:
            markdown_text: The Markdown text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing 'text' and 'metadata' for each chunk
        """
        import re
        from typing import List, Dict, Any
        
        # First split by major sections (h1, h2, h3 headers)
        # Using positive lookbehind with fixed width (\n followed by 1-3 # and a space)
        sections = re.split(r'(?<=\n)(?=#{1,3}\s)', markdown_text)
        
        chunks = []
        current_chunk = ""
        current_headers = []
        
        for section in sections:
            # Extract header if this section starts with one
            header_match = re.match(r'^(#+)\s*(.*?)\n', section, re.DOTALL)
            if header_match:
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                section_content = section[header_match.end():].strip()
                
                # Update current headers stack
                current_headers = current_headers[:header_level-1] + [header_text]
            else:
                section_content = section.strip()
            
            # If section is empty, skip it
            if not section_content:
                continue
                
            # If section is too big, split by paragraphs
            if len(section_content) > chunk_size:
                paragraphs = re.split(r'\n\s*\n', section_content)
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                        
                    # If paragraph is still too big, split by sentences
                    if len(para) > chunk_size:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        current_sentence_group = []
                        current_group_size = 0
                        
                        for sent in sentences:
                            sent = sent.strip()
                            if not sent:
                                continue
                                
                            sent_size = len(sent)
                            
                            # If adding this sentence would exceed chunk size (with some room for overlap)
                            if current_group_size + sent_size > chunk_size - chunk_overlap and current_sentence_group:
                                # Save current group as a chunk
                                chunk_text = ' '.join(current_sentence_group)
                                chunks.append({
                                    'text': chunk_text,
                                    'metadata': {
                                        'headers': current_headers,
                                        'chunk_type': 'sentence_group',
                                        'chunk_size': len(chunk_text)
                                    }
                                })
                                
                                # Start new group with overlap
                                overlap_start = max(0, len(current_sentence_group) // 2)
                                current_sentence_group = current_sentence_group[overlap_start:]
                                current_group_size = sum(len(s) + 1 for s in current_sentence_group)
                            
                            # Add current sentence to group
                            current_sentence_group.append(sent)
                            current_group_size += sent_size + 1
                        
                        # Add any remaining sentences
                        if current_sentence_group:
                            chunk_text = ' '.join(current_sentence_group)
                            chunks.append({
                                'text': chunk_text,
                                'metadata': {
                                    'headers': current_headers,
                                    'chunk_type': 'sentence_group',
                                    'chunk_size': len(chunk_text)
                                }
                            })
                    else:
                        # Paragraph is a good size, add as is
                        chunks.append({
                            'text': para,
                            'metadata': {
                                'headers': current_headers,
                                'chunk_type': 'paragraph',
                                'chunk_size': len(para)
                            }
                        })
            else:
                # Section is a good size, add as is
                chunks.append({
                    'text': section_content,
                    'metadata': {
                        'headers': current_headers,
                        'chunk_type': 'section',
                        'chunk_size': len(section_content)
                    }
                })
        
        return chunks

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
                    # Extract the text content from the chunk
                    chunk_text = chunk['text'] if isinstance(chunk, dict) else str(chunk)
                    metadata = {
                        "paper_id": paper.paper_id or "unknown_id",
                        "title": paper.title or "Title Not Available",
                        "authors": ", ".join([a.name for a in paper.authors]) if paper.authors else "Authors Not Available",
                        "chunk_type": chunk.get('metadata', {}).get('chunk_type', 'unknown') if isinstance(chunk, dict) else 'unknown',
                        "headers": " > ".join(chunk.get('metadata', {}).get('headers', [])) if isinstance(chunk, dict) and chunk.get('metadata', {}).get('headers') else ""
                    }
                    all_chunks.append(chunk_text)
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

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paper by its ID from the knowledge base.

        Args:
            paper_id: The ID of the paper to retrieve.

        Returns:
            A dictionary containing the paper data, or None if not found.
        """
        try:
            # Search for chunks belonging to this paper
            results = self.collection.get(
                where={"paper_id": paper_id}
            )
            
            if not results or not results.get('ids'):
                return None
            
            # Combine chunks into a single paper representation
            paper_data = {
                "paper_id": paper_id,
                "title": "",
                "content": "",
                "metadata": {}
            }
            
            # Extract information from the first chunk
            if results.get('metadatas') and len(results['metadatas']) > 0:
                first_metadata = results['metadatas'][0]
                paper_data["title"] = first_metadata.get("title", "")
                paper_data["metadata"] = first_metadata
            
            # Combine content from all chunks
            if results.get('documents'):
                paper_data["content"] = "\n\n".join(results['documents'])
            
            return paper_data
        except Exception as e:
            logger.error(f"Error retrieving paper {paper_id}: {e}")
            return None