# Graph-Based Retrieval-Augmented Generation

This project implements and compares two Retrieval-Augmented Generation (RAG) pipelines using **“Attention Is All You Need”** by Vaswani et al. (2017) as the sole knowledge source:

- **Graph RAG**: retrieves structured knowledge graph triples generated with KGGen.
- **Vector RAG**: retrieves semantically similar text chunks from a local ChromaDB vector database.

Both systems use the same source paper, embedding model, and generation model where applicable, allowing their retrieval behavior to be compared under similar conditions.

---

## 1. Project Structure

```text
Graph_RAG/
│
├── data/
│   ├── attention_is_all_you_need.pdf
│   ├── attention_is_all_you_need.txt
│   └── kg_triples.json
│
├── build_kg.py
├── graph_rag.py
├── vector_rag.py
│
├── .env.template
├── .gitignore
├── requirements.txt
└── README.md
```

Additional files and directories are generated locally when the pipelines are run:

```text
data/entity_index.json   # Persistent entity embeddings used by Graph RAG
chroma_db/               # Local persistent ChromaDB database used by Vector RAG
```

These local indexes are excluded from Git through `.gitignore` and can be regenerated from the source data.

---

## 2. Requirements

The project requires:

- Python 3.10 or later
- An OpenAI API key
- Internet access for OpenAI API calls

All Python dependencies are listed in `requirements.txt`.

---

## 3. Clone the Repository

```bash
git clone https://github.com/Xinyue-Zhang01/Graph_RAG.git
cd Graph_RAG
```

---

## 4. Create a Virtual Environment

Using a Python virtual environment is recommended.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 5. Configure Environment Variables

The repository contains an `.env.template` file. Create a local `.env` file from this template.

### Windows

```bash
copy .env.template .env
```

### macOS / Linux

```bash
cp .env.template .env
```

Open `.env` and replace:

```env
OPENAI_API_KEY=your_api_key_here
```

with a valid OpenAI API key.

The default configuration is:

```env
# OpenAI
OPENAI_API_KEY=your_api_key_here

# Models
KG_MODEL=openai/gpt-4o
GENERATION_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Generation parameters
TEMPERATURE=0.1
TOP_P=0.3

# KG Construction
KG_CHUNK_SIZE=5000
KG_CLUSTER=true

# Graph RAG
GRAPH_SEED_K=5
GRAPH_MAX_HOPS=2
GRAPH_MAX_TRIPLES=20

GRAPH_TRIPLE_SIM_WEIGHT=0.7
GRAPH_SEED_SCORE_WEIGHT=0.3
GRAPH_HOP_PENALTY=0.05

# Vector RAG
CHROMA_COLLECTION=attention_is_all_you_need
CHROMA_PERSIST_DIR=./chroma_db
VECTOR_CHUNK_SIZE=1000
VECTOR_CHUNK_OVERLAP=150
VECTOR_TOP_K=5
```

The local `.env` file is ignored by Git and should never be committed.

---

## 6. Source Data

The sole knowledge source used by both RAG systems is:

**Vaswani, A. et al. (2017). “Attention Is All You Need.”**

Original paper: https://arxiv.org/pdf/1706.03762.pdf

The source PDF is stored at:

```text
data/attention_is_all_you_need.pdf
```

The same paper is used for both the Graph RAG and Vector RAG pipelines.

---

# Graph RAG

## 7. Build the Knowledge Graph

Run:

```bash
python build_kg.py
```

The script performs the following pipeline:

```text
PDF
 ↓
Extract text with PyMuPDF
 ↓
KGGen
 ↓
Semantic triples
 ↓
kg_triples.json
```

### Step 1 — Text Extraction

Text is extracted from:

```text
data/attention_is_all_you_need.pdf
```

and stored in:

```text
data/attention_is_all_you_need.txt
```

If the text file already exists, `build_kg.py` reuses it instead of extracting the PDF again.

### Step 2 — Knowledge Graph Construction

KGGen processes the extracted paper and produces relations represented exclusively as triples:

```text
(Subject, Predicate, Object)
```

The triples are stored as JSON in:

```text
data/kg_triples.json
```

Example:

```json
{
  "subject": "Multi-Head Attention",
  "predicate": "attends to",
  "object": "representation subspaces"
}
```

The main KG construction settings are:

```env
KG_MODEL=openai/gpt-4o
KG_CHUNK_SIZE=5000
KG_CLUSTER=true
```

---

## 8. Run Graph RAG

After `kg_triples.json` has been created, run:

```bash
python graph_rag.py
```

On startup, the program:

```text
Loads kg_triples.json
 ↓
Builds a NetworkX MultiDiGraph
 ↓
Loads or creates entity embeddings
 ↓
Creates data/entity_index.json if needed
 ↓
Starts the command-line question-answering interface
```

After initialization, the program displays:

```text
Graph RAG
Hybrid retrieval: lexical + semantic seeds, graph expansion, triple reranking
Type 'exit' to quit.

Question:
```

Enter a question about the paper and press Enter.

To stop the program:

```text
exit
```

---

## 9. Graph RAG Retrieval Method

The current Graph RAG pipeline uses a hybrid, query-aware retrieval strategy:

```text
User question
      ↓
Hybrid entity linking
      ├── lexical entity matching
      └── semantic entity matching
      ↓
Seed entities
      ↓
Graph expansion over incoming and outgoing edges
      ↓
Candidate triples from up to 2 hops
      ↓
Query-aware semantic triple reranking
      ↓
Top retrieved triples
      ↓
LLM answer with exact triple citations
```

### 9.1 Hybrid Entity Linking

The query is first linked to knowledge graph entities in two ways.

**Lexical matching** identifies KG entities whose normalized names occur directly in the question. These direct matches are assigned a seed score of `1.0`.

**Semantic matching** embeds the complete query and compares it with the stored KG entity embeddings using cosine similarity. The top semantic entities are added as additional seeds.

Duplicate lexical and semantic matches are merged, with lexical matches taking priority.

### 9.2 Graph Expansion

The seed entities are used as entry points into a NetworkX `MultiDiGraph`.

The retriever follows both incoming and outgoing graph edges up to:

```env
GRAPH_MAX_HOPS=2
```

All triples reached during this stage form a **candidate triple set**. The candidate set is not immediately truncated to the final context size.

### 9.3 Query-Aware Triple Reranking

Each candidate triple is serialized as:

```text
subject predicate object
```

and embedded with the same embedding model used for entity linking.

Candidate triples are then scored using three factors:

```text
final score
=
(query–triple semantic similarity × GRAPH_TRIPLE_SIM_WEIGHT)
+
(seed relevance × GRAPH_SEED_SCORE_WEIGHT)
-
(hop penalty)
```

With the default configuration:

```env
GRAPH_TRIPLE_SIM_WEIGHT=0.7
GRAPH_SEED_SCORE_WEIGHT=0.3
GRAPH_HOP_PENALTY=0.05
```

This gives greater importance to triples that are semantically relevant to the query while still considering the relevance of the seed entity and graph distance.

After reranking, only the highest-scoring triples are supplied to the generation model:

```env
GRAPH_MAX_TRIPLES=20
```

### 9.4 Answer Generation and Citations

The generation model is instructed to answer using **only the retrieved triples**.

Every factual claim must cite its supporting triple directly in the exact format:

```text
[Subject] -> [Predicate] -> [Object]
```

For example:

```text
[Transformer] -> [relies on] -> [self-attention]
```

If the retrieved triples do not contain enough information, the system is instructed to state that the retrieved knowledge graph is insufficient rather than use outside knowledge.

---

## 10. Graph RAG Configuration

The main Graph RAG retrieval parameters are:

```env
GRAPH_SEED_K=5
GRAPH_MAX_HOPS=2
GRAPH_MAX_TRIPLES=20
GRAPH_TRIPLE_SIM_WEIGHT=0.7
GRAPH_SEED_SCORE_WEIGHT=0.3
GRAPH_HOP_PENALTY=0.05
```

- `GRAPH_SEED_K`: number of semantic entity matches added to the seed set. Lexical matches are added separately.
- `GRAPH_MAX_HOPS`: maximum graph expansion depth.
- `GRAPH_MAX_TRIPLES`: maximum number of reranked triples passed to the LLM.
- `GRAPH_TRIPLE_SIM_WEIGHT`: weight of query–triple semantic similarity in reranking.
- `GRAPH_SEED_SCORE_WEIGHT`: weight of the seed entity relevance score.
- `GRAPH_HOP_PENALTY`: penalty applied to triples reached through additional hops.

The command-line output also displays:

- seed entities and whether they were found lexically or semantically;
- the number of candidate triples found before reranking;
- the final retrieved triples;
- semantic, final, and hop scores for each retrieved triple.

These outputs are useful for inspecting the behavior of the graph retrieval pipeline.

---

## 11. Rebuild the Entity Index

The entity embedding index is stored locally at:

```text
data/entity_index.json
```

It is automatically reused on later runs.

If `kg_triples.json` is regenerated or the embedding model is changed, delete the existing index before running Graph RAG again.

### Windows

```bash
del data\entity_index.json
```

### macOS / Linux

```bash
rm data/entity_index.json
```

Then run:

```bash
python graph_rag.py
```

The entity embeddings will be rebuilt automatically.

---

# Vector RAG

## 12. Build and Run the Vector RAG Pipeline

Run:

```bash
python vector_rag.py
```

If no existing ChromaDB collection is found, the program automatically performs:

```text
PDF
 ↓
Load pages
 ↓
Split into overlapping text chunks
 ↓
Generate embeddings
 ↓
Store chunks in ChromaDB
 ↓
Start the command-line question-answering interface
```

After initialization, the program displays:

```text
Vector RAG
Source: Attention Is All You Need
Type 'exit' to quit.

Question:
```

For each query, Vector RAG performs:

```text
Question
 ↓
Query embedding
 ↓
Vector similarity search
 ↓
Top-k text chunks
 ↓
LLM
 ↓
Answer with source citations
```

The generation model is instructed to answer only from the retrieved text chunks.

---

## 13. Vector RAG Configuration

The main Vector RAG settings are:

```env
VECTOR_CHUNK_SIZE=1000
VECTOR_CHUNK_OVERLAP=150
VECTOR_TOP_K=5
```

- `VECTOR_CHUNK_SIZE`: text chunk size used to construct the vector index.
- `VECTOR_CHUNK_OVERLAP`: overlap between neighboring chunks.
- `VECTOR_TOP_K`: number of chunks retrieved for each query.

The local ChromaDB collection is stored at:

```text
chroma_db/
```

---

## 14. Rebuild the Vector Database

The existing ChromaDB index is reused on later runs.

If the embedding model, chunk size, or chunk overlap is changed, rebuild the vector database with:

```bash
python vector_rag.py --rebuild
```

This deletes the existing local ChromaDB index and recreates it from the source PDF using the current configuration.

---

## 15. Generation Parameters

Graph RAG and Vector RAG use the same default generation settings:

```env
TEMPERATURE=0.1
TOP_P=0.3
```

This allows the retrieval systems to be compared while keeping generation settings consistent.

The values can be changed in `.env` for parameter experiments, for example:

```env
TEMPERATURE=0.7
TOP_P=0.25
```

or:

```env
TEMPERATURE=1.4
TOP_P=0.98
```

Changing only `TEMPERATURE` or `TOP_P` does not require rebuilding the knowledge graph, entity index, or ChromaDB index.

---

## 16. Complete Setup and Execution Workflow

A complete run from a fresh clone is:

### Step 1 — Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Create `.env`

```bash
copy .env.template .env
```

Add the OpenAI API key to `.env`.

### Step 4 — Build or regenerate the knowledge graph

```bash
python build_kg.py
```

### Step 5 — Run Graph RAG

```bash
python graph_rag.py
```

### Step 6 — Run Vector RAG

```bash
python vector_rag.py
```

To explicitly rebuild the vector database:

```bash
python vector_rag.py --rebuild
```

---

## 17. Methodological Difference Between the Two Systems

The central difference is the retrieval unit and retrieval process.

### Graph RAG

```text
Query
→ hybrid entity linking
→ graph traversal
→ graph-derived candidate triples
→ semantic triple reranking
→ top triples
→ LLM
```

Embeddings are used to identify relevant graph entry points and to rerank triples that were obtained through graph expansion. The evidence supplied to the LLM remains structured knowledge graph triples.

### Vector RAG

```text
Query
→ vector similarity search over document chunks
→ top text chunks
→ LLM
```

The vector retriever directly searches representations of the original document chunks.

This setup allows structured graph-based retrieval to be compared with conventional vector-based semantic retrieval while keeping the source document and generation model consistent.

---

## 18. Notes on Graph Retrieval

The quality of Graph RAG depends on two separate stages:

1. **Knowledge graph construction** — the information must first be represented in `kg_triples.json` by KGGen.
2. **Graph retrieval** — relevant entities and triples must then be reachable from the selected seed entities and ranked highly enough to enter the final context.

Because the generation model is restricted to retrieved triples, information that was not extracted into the knowledge graph cannot be recovered during answer generation. Similarly, semantically relevant information may still be difficult to retrieve when related concepts are weakly connected or disconnected in the generated graph.

These behaviors are important when interpreting the experimental comparison between Graph RAG and Vector RAG.
