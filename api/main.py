import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from paper_agent.master_agent import MasterAgent
from paper_agent.agent import PaperAgent as WorkerAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(
    title="Agentic AI Research SDK",
    description="An API for running complex research analysis tasks.",
    version="1.0.0"
)

print("--- Initializing Master Agent for API ---")
worker_agent = WorkerAgent(llm_provider="google")
master_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.0) 
master_agent = MasterAgent(worker_agent=worker_agent, llm=master_llm, max_loops=15)
print("--- Master Agent is Online and Ready ---")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    report: str
    artifacts: list[str] = []


@app.post("/execute-query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Takes a complex user query, runs the Master Agent, and returns the final report.
    """
    try:
        print(f"Received query: {request.query}")
        final_report = master_agent.run(request.query)
        
    
        generated_artifacts = [f for f in os.listdir('./artifacts') if os.path.isfile(os.path.join('./artifacts', f))]

        return {"report": final_report, "artifacts": generated_artifacts}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """
    Serves a generated file (plot, csv, etc.) from the artifacts directory.
    """
    file_path = os.path.join("./artifacts", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Artifact not found.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Paper Agent API. Use the /docs endpoint to see the API documentation."}