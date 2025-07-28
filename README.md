# Paper Agent SDK

A next-generation scientific research assistant that combines large language models (LLMs), knowledge graphs, and automated analysis tools to revolutionize literature review and research synthesis.

## 🚀 Quick Start

1. **Start the application** (requires [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)):

   ```bash
   docker-compose up -d
   ```

2. **Access the web interface** at [http://localhost:3000](http://localhost:3000) 
note: I am currently building the UI, so its not ideal for use

3. **Or use the API** at [http://localhost:8000/docs](http://localhost:8000/docs)

## 🛑 Stopping the Application

```bash
docker-compose down
```

## 🔧 Configuration 

Create a `.env` file in the project root to customize settings:

```env
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional (defaults shown) 
NEO4J_URI=bolt://db:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

```

---

## Example: Automated Visualization from Research Paper

This system can extract quantitative results from scientific papers and generate insightful visualizations automatically. Below is an example workflow you can run via the API:

**Example Prompt:**
```
I need to understand the performance of models in the 'Attention is All You Need' paper.
Please do the following:
1. Extract the main results table that includes BLEU scores and Training Cost.
2. From that data, create a visualization to **compare the BLEU scores** of the different models. Save it as 'bleu_comparison.png'.
3. From the same data, create another visualization to show the **relationship between Training Cost and BLEU score**. Save it as 'cost_vs_performance.png'.
4. Provide a final answer summarizing what the two charts show.
```

**How to Use:**
- Submit this prompt to the `/query` endpoint via the API docs or your own client.
- The resulting visualizations and CSVs will be available in the `artifacts/` directory and can be downloaded via the `/artifacts/{filename}` endpoint.

**Resulting Visualizations:**

<p align="center">
  <img src="GeneratedGraph/bleu_comparison.png" alt="BLEU Score Comparison" width="400" style="display:inline-block; margin-right: 20px;"/>
  <img src="GeneratedGraph/cost_vs_performance.png" alt="Training Cost vs BLEU Score" width="400" style="display:inline-block;"/>
</p>

<p align="center">
  <b>Left:</b> BLEU Scores of Different Machine Translation Models &nbsp;&nbsp;&nbsp; <b>Right:</b> Relationship between Training Cost and BLEU Score
</p>

---

## Architecture & Tooling
- See `agent_and_tools_overview.md` for a detailed breakdown of agents and tools.
- Modular, extensible, and robust for real-world research workflows.

---


