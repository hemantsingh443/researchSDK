trying to build an Agentic SDK for research paper :)

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
python examples/05_run_rag_agent.py setup
```

### 6. Run the Agentic System
To launch the agent and ask questions(you would need to modify code script for different questions for now)
```
python examples/05_run_true_agent.py
```


## Example Usage

- Ask a research question:
  - "What are the main challenges of LLM agents?"
- Add new papers to the knowledge base:
  - "Find and load new papers about graph neural networks."

## Project Structure
- `paper_agent/` - Core SDK modules (agent, tools, ingestor, knowledge base, etc.)
- `examples/` - Example scripts for setup and running the agent
- `requirements.txt` - Python dependencies

