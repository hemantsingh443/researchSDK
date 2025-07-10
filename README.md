trying to build an Agentic SDK for research papers 

## Setup and Configuration

- By default, the SDK uses a local Ollama LLM (e.g., llama3) for all agentic and extraction tasks. No cloud API keys are required for local mode.
- **Optional: Google Gemini Support**
    - If you want to use Google Gemini (Gemini 1.5 Flash) as your LLM, set the `GOOGLE_API_KEY` environment variable (e.g., in your `.env` file) and pass `llm_provider='google'` when initializing your agent.
    - Example `.env` entry:
      ```
      GOOGLE_API_KEY=your-google-api-key
      ```
    - Example agent initialization:
      ```python
      agent = PaperAgent(llm_provider="google")
      ```
    - If `llm_provider` is not set to `google`, the system will use your local Ollama model by default.
- **Note:** OpenAI API keys are not required or supported. The SDK is designed for privacy and local-first workflows, with optional Gemini cloud support.

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


### 4. Populate the Knowledge Base 
Run the setup script to download and index some papers:
```
python examples/05_run_true_agent.py setup
```

### 5. Run the Agentic System
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

## PlanningAgent: Multi-Step Reasoning and State Management

The `PlanningAgent` is an advanced orchestrator that enables the SDK to handle complex, multi-step research tasks. It works by:

- **Decomposing complex queries:** Uses an LLM to break down a user's high-level request into a clear, numbered plan of simple steps, each mapped to a specific tool or action.
- **Stateful execution:** Maintains a scratchpad (internal state) that tracks the results of each step, allowing information (like paper IDs or summaries) to be passed between steps.
- **Step-by-step orchestration:** For each step, the agent injects the current scratchpad into the prompt, so the worker agent has memory of all prior results and can use them as needed.
- **Final synthesis:** After all steps are complete, the agent uses the full scratchpad to generate a comprehensive, synthesized answer for the user.

This approach enables robust tool chaining, variable passing, and reliable completion of complex research workflows (e.g., "Find two papers on X, summarize both, and compare their approaches").

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

