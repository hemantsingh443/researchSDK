from paper_agent.agent import PaperAgent as ReActAgent
from paper_agent.planner import PlanningAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import os

def run_advanced_tests():
    # --- Setup ---
    react_agent = ReActAgent(db_path="./paper_db", llm_provider="google")
    planner_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)
    planning_agent = PlanningAgent(react_agent=react_agent, planner_llm=planner_llm)

    # --- TEST CASE 1: Table Extraction and Plotting ---
    print("\n\n" + "="*50)
    print("TEST CASE 1: Table Extraction and Plot Generation")
    print("="*50)
    
    # We're asking for something very specific that should be in the 'Attention' paper
    # Note: The original 'Attention is All You Need' paper has tables that are hard
    # to parse from raw text. This query tests the LLM's ability to find and format it.
    query1 = """
    First, find the paper 'Attention is All You Need'.
    Then, use the table_extraction_tool to extract the table showing translation quality (BLEU scores) from that paper.
    Finally, use the plot_generation_tool to create a bar chart of the BLEU scores, title it 'Translation Quality (BLEU)', and save it as 'bleu_scores.png'.
    """
    
    response1 = planning_agent.run(query1)
    
    print("\n\n" + "="*50)
    print("PLANNING AGENT'S FINAL ANSWER FOR TEST 1")
    print("="*50)
    print(response1)
    if os.path.exists("bleu_scores.png"):
        print("\nSUCCESS: 'bleu_scores.png' was created successfully!")
    else:
        print("\nFAILURE: The plot file was not created.")

    # --- TEST CASE 2: Relationship Analysis ---
    print("\n\n" + "="*50)
    print("TEST CASE 2: Relationship Analysis (Graph-based)")
    print("="*50)
    
    query2 = "What is the relationship between the 'BERT' paper and the 'Attention is All You Need' paper?"
    
    response2 = planning_agent.run(query2)
    
    print("\n\n" + "="*50)
    print("PLANNING AGENT'S FINAL ANSWER FOR TEST 2")
    print("="*50)
    print(response2)

# We'll add more tests to this file later.
if __name__ == "__main__":
    if not os.path.exists("./paper_db"):
        print("Knowledge base not found. Run 'add_local_paper.py' first.")
        sys.exit(1)
    
    run_advanced_tests()