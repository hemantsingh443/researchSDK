from paper_agent.ingestor import Ingestor
from paper_agent.knowledge_base import KnowledgeBase
import os

pdf_path = 'attention.pdf' 
if not os.path.exists(pdf_path):
    print("Please download attention.pdf into the project root.")
else:
    print("--- Populating all databases ---")
    # Use the full constructor to connect to both databases
    ingestor = Ingestor()
    kb = KnowledgeBase(db_path="./paper_db", neo4j_password="password")
    
    paper = ingestor.load_from_pdf(pdf_path)
    
    # This will now add to both ChromaDB and Neo4j
    kb.add_papers([paper])
    print(f"Successfully added '{paper.title}' to all knowledge bases.")
    
    # Close the Neo4j connection
    kb.close()