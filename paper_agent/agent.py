from .knowledge_base import KnowledgeBase
from .extractor import Extractor # We will use the Extractor's LLM for generation

class PaperAgent:
    """
    An agent that can answer questions based on a knowledge base of papers.
    It uses the RAG (Retrieval-Augmented Generation) pattern.
    """
    def __init__(self, knowledge_base: KnowledgeBase, llm_extractor: Extractor):
        """
        Initializes the agent.

        Args:
            knowledge_base: An instance of KnowledgeBase, pre-populated with papers.
            llm_extractor: An instance of Extractor to access the LLM for generation.
        """
        self.kb = knowledge_base
        self.llm = llm_extractor.client # We can reuse the OpenAI client
        self.model = llm_extractor.model
        print("PaperAgent initialized.")
        print(f"-> Using KnowledgeBase with {self.kb.collection.count()} text chunks.")
        print(f"-> Using LLM model: {self.model}")

    def run_query(self, user_query: str, n_results: int = 5) -> str:
        """
        Executes the full RAG pipeline for a user query.

        1. Searches the KB for relevant text chunks.
        2. Constructs a detailed prompt with the retrieved context.
        3. Asks the LLM to synthesize an answer based on the context.
        
        Returns:
            A string containing the synthesized answer.
        """
        # --- 1. Retrieval ---
        print(f"\n[Step 1] Retrieving relevant context for: '{user_query}'")
        search_results = self.kb.search(query=user_query, n_results=n_results)
        
        # We need to format the search results into a single string for the prompt
        context_str = ""
        for i, doc in enumerate(search_results['documents'][0]):
            source = search_results['metadatas'][0][i]['title']
            context_str += f"--- Context Snippet {i+1} (from paper: {source}) ---\n"
            context_str += doc
            context_str += "\n\n"

        # --- 2. Augmentation ---
        prompt = f"""
        You are a helpful AI research assistant. Your task is to answer the user's question based *only* on the provided context.
        Do not use any of your own prior knowledge. If the context does not contain the answer, state that you cannot answer based on the provided documents.

        Here is the context retrieved from the research papers:
        <context>
        {context_str}
        </context>

        Here is the user's question:
        <question>
        {user_query}
        </question>

        Based on the context, what is the answer?
        """

        # --- 3. Generation ---
        print("[Step 2] Generating synthesized answer with LLM...")
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 # Lower temperature for more factual, less creative answers
        )

        answer = response.choices[0].message.content
        print("[Step 3] Done.")
        
        return answer