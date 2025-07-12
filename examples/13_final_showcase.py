from paper_agent.agent import PaperAgent as WorkerAgent
from paper_agent.master_agent import MasterAgent
from langchain_google_genai import ChatGoogleGenerativeAI

def run_final_showcase():
    worker_agent = WorkerAgent(db_path="./paper_db", llm_provider="google")
    master_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.0)
    master_agent = MasterAgent(worker_agent=worker_agent, llm=master_llm)

    final_query = """
    I need a comprehensive analysis of the current state of my research library.
    1. First, find the most cited papers in my entire library.
    2. Take the top cited paper and extract its main keywords.
    3. Also, take that same top paper and extract its main results table.
    4. For that table, create the following visualizations and save each as a separate file:
        a. A violin plot showing the distribution of BLEU scores across all models.
        b. A box plot comparing the training costs of all models.
        c. A histogram of BLEU scores.
        d. A bar chart comparing BLEU scores for each model.
        e. A scatter plot of BLEU score vs. training cost, highlighting Transformer models in red.
    5. Write a final report summarizing all your findings, including insights from each plot and referencing the saved plot files.
    """

    final_response = master_agent.run(final_query)

    print("\n\n" + "="*60)
    print("FINAL SHOWCASE: MASTER AGENT'S REPORT")
    print("="*60)
    print(final_response)

if __name__ == "__main__":
    run_final_showcase()