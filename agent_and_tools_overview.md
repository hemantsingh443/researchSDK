# Agent and Tools Overview

This document provides an overview of the core agents and tools in the scientific research assistant system, describing their roles, capabilities, and how they interact to accomplish complex research tasks.

---

## Agents

### MasterAgent
- **Role:** The central coordinator and planner.
- **Responsibilities:**
  - Interprets user queries and decomposes them into subtasks.
  - Selects and sequences the appropriate tools and agents to fulfill each subtask.
  - Synthesizes results into a final, user-friendly answer.
  - Handles error recovery and fallback strategies.

### WorkerAgent (PaperAgent)
- **Role:** The domain expert and executor.
- **Responsibilities:**
  - Executes specific research tasks (e.g., table extraction, summarization, visualization) as directed by the MasterAgent.
  - Interfaces with the knowledge base, LLMs, and external APIs.
  - Returns structured results for aggregation.

---

## Main Tools

### 1. **graph_paper_finder_tool**
- **Purpose:** Finds papers in the knowledge graph by title and returns metadata (including paper ID).
- **Usage:** First step for locating a specific paper for downstream tasks.

### 2. **table_extraction_tool**
- **Purpose:** Extracts structured tabular data (as JSON) from a paper based on a topic of interest.
- **Usage:** Used to obtain quantitative results for analysis and visualization.

### 3. **dynamic_visualization_tool**
- **Purpose:** Uses an LLM to generate and execute Python code for flexible, context-aware data visualizations (bar, line, scatter, etc.).
- **Usage:** Converts extracted data into insightful charts and plots.

### 4. **question_answering_tool**
- **Purpose:** Answers user questions using the knowledge base and retrieval-augmented generation (RAG).
- **Usage:** Provides direct answers or extracts information when table extraction is insufficient.

### 5. **arxiv_paper_search_and_load_tool**
- **Purpose:** Searches for and loads new papers from arXiv into the knowledge base.
- **Usage:** Expands the system's knowledge with the latest research.

### 6. **arxiv_fetch_by_id_tool**
- **Purpose:** Fetches a specific paper from arXiv using its unique ID.
- **Usage:** Ensures precise retrieval of known papers.

### 7. **paper_summarization_tool**
- **Purpose:** Generates concise summaries of specific papers in the knowledge base.
- **Usage:** Provides quick overviews for users.

### 8. **graph_query_tool**
- **Purpose:** Runs Cypher queries on the Neo4j knowledge graph to answer questions about relationships (e.g., authorship, citations).
- **Usage:** Explores connections between papers, authors, and topics.

### 9. **relationship_analysis_tool**
- **Purpose:** Explains the relationship between two papers using graph queries and LLM synthesis.
- **Usage:** Clarifies citation paths, collaborations, and research influence.

---

## How Agents and Tools Interact
- The **MasterAgent** receives a user query and plans a sequence of tool invocations.
- The **WorkerAgent** executes these tool calls, handling data extraction, analysis, and visualization.
- Tools are modular and can be combined in various ways to solve complex research questions.
- Robust error handling and fallback logic ensure the system can recover from missing data or tool failures.

---

This architecture enables flexible, robust, and intelligent automation of scientific research workflows, from literature search to data extraction, analysis, and visualization. 