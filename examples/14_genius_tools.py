from paper_agent.agent import PaperAgent as WorkerAgent
from paper_agent.master_agent import MasterAgent
from langchain_google_genai import ChatGoogleGenerativeAI


def run_genius_test():
    worker_agent = WorkerAgent(db_path="./paper_db", llm_provider="google")
    master_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.0)
    master_agent = MasterAgent(worker_agent=worker_agent, llm=master_llm)
    query1 = "Check if the BERT paper and another paper on 'ELMo' report conflicting results on the GLUE benchmark. and show the visuals"
    query2 = "Analyze the literature on 'Mixture of Experts' from the papers in my library and suggest a novel research direction."
    
    print("\n--- conflicting results ---")
    response2 = master_agent.run(query1)
   
    
    print("\n\n" + "="*60)
    print("GAP ANALYSIS REPORT")
    print("="*60)
    print(response2)

if __name__ == "__main__":
    run_genius_test()