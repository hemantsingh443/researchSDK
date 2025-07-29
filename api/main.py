import os
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
import json
from fastapi.websockets import WebSocketState
from typing import Dict, List
import json
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from paper_agent.master_agent import MasterAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Agentic AI Research SDK",
    description="An API for running complex research analysis tasks.",
    version="1.0.0"
)

# Configure CORS for HTTP and WebSocket
allowed_origins = [
    "http://localhost:3001",
    "http://localhost:3000",
    "http://localhost:5173"  
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # 10 minutes
)

# Ensure artifacts directory exists
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Client {client_id} disconnected")

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                print(f"Sending message to client {client_id}: {message}")
                await websocket.send_json(message)

manager = ConnectionManager()

# Initialize the MasterAgent lazily
master_agent = None

def get_master_agent():
    global master_agent
    if master_agent is None:
        print("--- Initializing Master Agent for API (lazy loading) ---")
        master_agent = MasterAgent()
    return master_agent

print("--- API Starting (Master Agent will be loaded on first request) ---")
try:
    # Initialize with Google's Gemini model by default
    master_agent = MasterAgent(llm_provider="google", max_loops=20)
    print("--- Master Agent is Online and Ready ---")
except Exception as e:
    print(f"Failed to initialize MasterAgent: {e}")
    print("Falling back to local LLM...")
    master_agent = MasterAgent(llm_provider="local", max_loops=20)
    print("--- Master Agent is Online (Local LLM) and Ready ---")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    report: str
    artifacts: list[str] = []
    thought_process: list[dict] = []

@app.post("/api/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Takes a complex user query, runs the Master Agent, and returns the final report.
    """
    try:
        print(f"--- Received query: {request.query}")
        
        # Get the lazy-loaded MasterAgent instance
        agent = get_master_agent()
        
        # Process the query using the Master Agent
        print("--- Starting Master Agent execution...")
        final_report, thought_process = await agent.run(request.query)
        
        # Format the result to match expected structure
        result = {
            "final_answer": final_report,
            "thought_process": thought_process,
            "artifacts": []
        }
        
        if not result or not result.get("final_answer"):
            raise HTTPException(status_code=500, detail="Agent execution failed or returned no result")
        
        # Extract artifacts from the result
        artifacts = result.get("artifacts", [])
        thought_process = result.get("thought_process", [])
        
        print(f"--- Query completed. Artifacts generated: {len(artifacts)}")
        
        return QueryResponse(
            report=result["final_answer"],
            artifacts=artifacts,
            thought_process=thought_process
        )
    except Exception as e:
        error_msg = f"An error occurred while processing the query: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """
    Serves a generated file (plot, csv, etc.) from the artifacts directory.
    """
    file_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Artifact not found.")

@app.get("/artifacts")
async def list_artifacts():
    """List all artifacts in JSON format."""
    try:
        artifacts = []
        if os.path.exists(ARTIFACTS_DIR):
            for filename in os.listdir(ARTIFACTS_DIR):
                filepath = os.path.join(ARTIFACTS_DIR, filename)
                if os.path.isfile(filepath):
                    created = datetime.fromtimestamp(os.path.getctime(filepath))
                    size = os.path.getsize(filepath)
                    artifacts.append({
                        "name": filename,
                        "created": created.isoformat(),
                        "size": size,
                        "url": f"/artifacts/{filename}",
                    })
        
        # Sort by creation time, newest first
        artifacts.sort(key=lambda x: x["created"], reverse=True)
        return {"artifacts": artifacts}
    except Exception as e:
        print(f"Error listing artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/artifacts/delete/{filename}")
async def delete_artifact(filename: str):
    try:
        file_path = Path(f"artifacts/{filename}")
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return {"status": "success", "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifacts")
async def get_artifacts():
    artifacts_dir = Path("artifacts")
    artifacts = []
    
    if artifacts_dir.exists() and artifacts_dir.is_dir():
        for file_path in artifacts_dir.glob("*"):
            if file_path.is_file():
                artifacts.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "created": file_path.stat().st_ctime,
                    "url": f"/artifacts/{file_path.name}"
                })
    
    # Sort by creation time, newest first
    artifacts.sort(key=lambda x: x["created"], reverse=True)
    return {"artifacts": artifacts}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    # If no client_id is provided, generate one
    if client_id is None:
        import uuid
        client_id = str(uuid.uuid4())
    
    # Check origin
    origin = websocket.headers.get('origin')
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173"   # Vite dev server with IP
    ]
    
    if origin and origin not in allowed_origins:
        print(f"WebSocket connection rejected: Origin not allowed - {origin}")
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    
    try:
        print(f"WebSocket connection accepted for client {client_id}")
        
        # Add client to the connection manager
        await manager.connect(websocket, client_id)
        
        while True:
            try:
                # Wait for a message from the client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    print(f"Received message from {client_id}: {message}")
                    
                    if message.get('type') == 'new_query' and 'query' in message:
                        query = message['query']
                        
                        # Send initial response
                        await manager.send_message({
                            'type': 'status',
                            'status': 'processing',
                            'message': f'Processing query: {query}'
                        }, client_id)
                        
                        # Process the query with the master agent
                        try:
                            # Send initial processing message
                            await manager.send_message({
                                'type': 'task_update',
                                'status': 'in_progress',
                                'step': {
                                    'type': 'processing',
                                    'content': 'Processing your query...\n'
                                }
                            }, client_id)
                            
                            # Create a callback function to send messages via WebSocket
                            async def websocket_callback(message):
                                try:
                                    print(f"Processing callback message: {message}")
                                    # Forward all real-time messages directly to the frontend
                                    await manager.send_message(message, client_id)
                                except Exception as e:
                                    print(f"Error sending WebSocket message: {e}")
                            
                            # Execute the query with the callback
                            final_report, thought_process = await master_agent.run(query, websocket_callback=websocket_callback)
                            
                            # Send completion message with results
                            await manager.send_message({
                                'type': 'task_update',
                                'status': 'completed',
                                'result': final_report,
                                'artifacts': [
                                    f for f in os.listdir(ARTIFACTS_DIR) 
                                    if os.path.isfile(os.path.join(ARTIFACTS_DIR, f))
                                ]
                            }, client_id)
                            
                        except Exception as e:
                            error_msg = str(e)
                            print(f"Error processing query: {error_msg}")
                            await manager.send_message({
                                'type': 'task_update',
                                'status': 'error',
                                'error': f'Error processing query: {error_msg}'
                            }, client_id)
                
                except json.JSONDecodeError:
                    await manager.send_message({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }, client_id)
            
            except WebSocketDisconnect:
                print(f"Client {client_id} disconnected")
                manager.disconnect(client_id)
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                manager.disconnect(client_id)
                break
            
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(client_id)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint_with_id(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get('type') == 'new_query':
                    query_content = message.get('query', '')
                    print(f"Processing query from {client_id}: {query_content}")
                    
                    try:
                        # Get the MasterAgent instance (lazy loaded)
                        agent = get_master_agent()
                        
                        # Send initial processing message
                        await manager.send_message({
                            'type': 'task_update',
                            'status': 'in_progress',
                            'step': {
                                'type': 'processing',
                                'content': 'Processing your query...'
                            }
                        }, client_id)
                        
                        # Create a callback function to send messages via WebSocket
                        async def websocket_callback(message):
                            try:
                                await manager.send_message(message, client_id)
                            except Exception as e:
                                print(f"Error sending WebSocket message: {e}")
                        
                        # Run the agent with the callback
                        final_report, thought_process = await agent.run(
                            query_content,
                            websocket_callback=websocket_callback
                        )
                        
                        # Format the result to match expected structure
                        result = {
                            "final_answer": final_report,
                            "thought_process": thought_process
                        }
                        
                        if result and result.get("final_answer"):
                            response = {
                                'type': 'response',
                                'content': result["final_answer"]
                            }
                        else:
                            response = {
                                'type': 'error',
                                'content': 'Sorry, I could not process your query. Please try again.'
                            }
                        
                        await manager.send_message(json.dumps(response), client_id)
                        
                    except Exception as e:
                        print(f"Error processing query: {e}")
                        error_response = {
                            'type': 'error',
                            'content': f'Error processing query: {str(e)}'
                        }
                        await manager.send_message(json.dumps(error_response), client_id)
                        
            except json.JSONDecodeError:
                await manager.send_message(json.dumps({
                    'type': 'error',
                    'content': 'Invalid JSON format'
                }), client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client #{client_id} disconnected")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Paper Agent API. Use the /docs endpoint to see the API documentation."}

# Serve static files for the frontend
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")