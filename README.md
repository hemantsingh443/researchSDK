trying to build an Agentic SDK for research papers 

## for Testing

### 1. Set up the environment  

```
python -m venv myenv 
```

For Linux system or WSL:
```
source myenv/bin/activate
```
For Windows:
```
myenv\Scripts\activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

You may also need to install system packages for PDF processing:
```
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
```

### 3. Set up Ollama (for local LLM)
- Download and install [Ollama](https://ollama.com/)
- Pull a supported model (recommended: Llama 3):
```
ollama pull llama3:8b-instruct-q4_K_M
```
- Make sure Ollama is running on `http://localhost:11434`

### 4. (Optional) Set up environment variables
If you want to use OpenAI or other LLMs, set your API keys in a `.env` file.

### 5. Populate the Knowledge Base 
Run the setup script to download and index some papers:
```
python examples/05_run_true_agent.py setup
```

### 6. Run the Agentic System
To launch the agent and ask questions (you would need to modify code script for different questions for now)
```
python examples/05_run_true_agent.py
```


## Example Usage

- Ask a research question:
  - "What are the main challenges of LLM agents?"
- Add new papers to the knowledge base:
  - "Find and load new papers about graph neural networks."

## How Tool Chaining Works

- The agent can now reliably chain tools for multi-step tasks, such as summarizing a paper:
  1. It uses `paper_finder_tool` to get the paper_id (returns `{ "paper_id": "attention.pdf" }`).
  2. It immediately uses `paper_summarization_tool` with that paper_id.
  3. The final answer is the summary.

**Example summarization query:**

User: "Please provide a summary of the 'Attention is All You Need' paper."

Agent reasoning:
```
Thought: The user wants a summary. I should use paper_finder_tool to get the paper_id.
Action:
{
  "action": "paper_finder_tool",
  "action_input": {"query": "Attention is All You Need"}
}
Observation: {"paper_id": "attention.pdf"}
Thought: I have the paper_id. I will now use paper_summarization_tool.
Action:
{
  "action": "paper_summarization_tool",
  "action_input": {"paper_id": "attention.pdf"}
}
Observation: "[summary text]"
Thought: I have the summary. I will now provide the final answer.
Action:
{
  "action": "Final Answer",
  "action_input": "[summary text]"
}
```

- For author/collaboration queries, the agent uses the graph_query_tool directly.

## Dual Database Architecture: Vector Search and Graph Reasoning

This SDK uses **two databases** for maximum research power:

### 1. ChromaDB (Vector Store)
- Used for semantic search and retrieval-augmented generation (RAG).
- Stores paper content in vectorized chunks for fast similarity search.
- Powers tools like `paper_finder_tool` and `question_answering_tool`.
- Populated automatically when you ingest or add papers.

### 2. Neo4j (Graph Database)
- Used for structured queries about relationships: authors, collaborations, citations, etc.
- Powers the `graph_query_tool` for Cypher queries (e.g., "Who wrote X?", "Who collaborated with Y?").
- Populated automatically alongside ChromaDB when you ingest papers.

### How the Agent Uses Both
- **Semantic search:** Finds relevant papers using ChromaDB.
- **Graph queries:** Answers author/collaboration/relationship questions using Neo4j.
- **Chaining:** Finds a paper in ChromaDB, then uses its ID for graph or summarization queries.

### Neo4j Setup
- Download and install [Neo4j Community Edition](https://neo4j.com/download-center/#community)
- Start Neo4j and ensure it is accessible (default: `bolt://localhost:7687`)
- Set your Neo4j credentials in the agent config or environment variables if needed.
- The agent will automatically create nodes for papers and authors, and relationships like `:AUTHORED`.

### Populating Both Databases
- When you ingest a paper (via the setup script or agent tools), it is added to both ChromaDB and Neo4j.
- This ensures both semantic and structured queries are always up-to-date.

## Project Structure
- `paper_agent/` - Core SDK modules (agent, tools, ingestor, knowledge base, etc.)
- `examples/` - Example scripts for setup and running the agent
- `requirements.txt` - Python dependencies

## Troubleshooting
- If the agent loops or fails to chain tools, ensure `paper_finder_tool` returns only `{ "paper_id": ... }` as a JSON string.
- For LLM errors, check your Ollama/OpenAI/Groq setup and environment variables.
- For local LLMs, Ollama is recommended and tested.
- For graph queries, ensure Neo4j is running and accessible, and that papers/authors are ingested.

