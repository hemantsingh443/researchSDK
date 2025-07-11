# Paper Agent SDK

A scientific research assistant system that leverages LLMs, LangChain, Neo4j, and dynamic visualization tools to automate literature search, data extraction, and analysis from research papers.

---

## Features
- **Automated paper search and ingestion** (arXiv, local, graph-based)
- **Structured table extraction** from scientific papers
- **Dynamic, LLM-driven data visualization** (bar, line, scatter, etc.)
- **Relationship and citation analysis** via knowledge graph
- **Robust error handling and fallback strategies**

---

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up Neo4j and vector DB as described in the docs.**
3. **Ingest papers** (see `add_local_paper.py` or arXiv tools).
4. **Run an example agentic workflow:**
   ```bash
   python examples/12_run_dynamic_viz.py
   ```

---

## Example: Automated Visualization from Research Paper

This system can extract quantitative results from scientific papers and generate insightful visualizations automatically. Below is an example using the prompt from `examples/12_run_dynamic_viz.py`:

**Prompt:**
```
I need to understand the performance of models in the 'Attention is All You Need' paper.
Please do the following:
1. Extract the main results table that includes BLEU scores and Training Cost.
2. From that data, create a visualization to **compare the BLEU scores** of the different models. Save it as 'bleu_comparison.png'.
3. From the same data, create another visualization to show the **relationship between Training Cost and BLEU score**. Save it as 'cost_vs_performance.png'.
4. Provide a final answer summarizing what the two charts show.
```

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


