from paper_agent.ingestor import Ingestor
from paper_agent.knowledge_base import KnowledgeBase
from paper_agent.extractor import Extractor

# --- This is a mock citation extraction ---
# In a real system, the extractor would find this in the text.
# Here, we manually add it for the test case.
class MockExtractor(Extractor):
    def extract_citations(self, text):
        # If we see 'Transformer', we assume it cites 'Attention Is All You Need'
        if "Transformer" in text:
            return ["Attention Is All You Need"]
        return []

ingestor = Ingestor()
# Temporarily override the extractor with our mock one
ingestor.extractor = MockExtractor(api_type="google") 

kb = KnowledgeBase(db_path="./paper_db", neo4j_password="password")

# Fetch the BERT paper (ID 1810.04805)
papers = ingestor.load_from_arxiv(query="1810.04805", max_results=1)

# Manually add the citation for the test
if papers:
    papers[0].citations = ingestor.extractor.extract_citations(papers[0].full_text)
    kb.add_papers(papers)
    print("Added BERT paper with mock citation link to the knowledge base.")
    kb.close()