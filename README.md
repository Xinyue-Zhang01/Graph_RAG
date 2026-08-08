# Graph-Based Retrieval-Augmented Generation

This project implements and compares two Retrieval-Augmented Generation (RAG) pipelines using the paper **“Attention Is All You Need”** by Vaswani et al. (2017) as the sole knowledge source:

* **Graph RAG**: retrieves information from a knowledge graph constructed with KGGen.
* **Vector RAG**: retrieves semantically similar text chunks from a ChromaDB vector database.

The two pipelines use the same source document and generation model so that their retrieval behavior can be compared under similar conditions.

---

## 1. Project Structure

```text
Graph-Based-Retrieval-Augmented-Generation/
│
├── data/
│   └── attention_is_all_you_need.pdf
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

Additional files are generated locally when the pipelines are run:

```text
data/attention_is_all_you_need.txt   # Source text extracted from PDF
data/kg_triples.json                 # Knowledge Graph with semantic triples
data/entity_index.json               # Entity embeddings for Graph RAG
chroma_db/                           # Local ChromaDB database for Vector RAG
```

These generated files are excluded from Git through `.gitignore`.

---

## 2. Requirements

The project requires:

* Python 3.10 or later
* An OpenAI API key
* Internet access for OpenAI API calls

The main Python dependencies are listed in `requirements.txt`.

---

## 3. Create a Virtual Environment

It is recommended to run the project in a Python virtual environment.

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

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

The repository contains an `.env.template` file.

Create a local `.env` file from the template.

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

with your actual OpenAI API key.

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
GRAPH_MAX_HOPS=1
GRAPH_MAX_TRIPLES=20

# Vector RAG
CHROMA_COLLECTION=attention_is_all_you_need
CHROMA_PERSIST_DIR=./chroma_db
VECTOR_CHUNK_SIZE=1000
VECTOR_CHUNK_OVERLAP=150
VECTOR_TOP_K=5
```

The `.env` file is ignored by Git and should not be committed.

---

## 5. Source Data

The sole knowledge source used in this project is:

**Vaswani et al. (2017), “Attention Is All You Need.”**

Direct PDF Link: https://arxiv.org/pdf/1706.03762.pdf

The PDF should be located at:

```text
data/attention_is_all_you_need.pdf
```

Both the Graph RAG and Vector RAG pipelines use this same document.

---

## 6. Graph RAG - Build the Knowledge Graph

Run:

```bash
python build_kg.py
```

The script performs the following steps:

```text
PDF
 ↓
Extract text
 ↓
KGGen
 ↓
Semantic triples
 ↓
kg_triples.json
```

First, text is extracted from:

```text
data/attention_is_all_you_need.pdf
```

and stored in:

```text
data/attention_is_all_you_need.txt
```

KGGen then processes the extracted text and creates semantic triples in the form:

```text
(Subject, Predicate, Object)
```

The resulting knowledge graph data is stored in:

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

If the extracted text file already exists, `build_kg.py` reuses it instead of extracting the PDF again.

---

## 7. Graph RAG - Run Graph RAG

After the knowledge graph has been created, run:

```bash
python graph_rag.py
```

When Graph RAG is started for the first time, the system:

```text
Loads kg_triples.json
 ↓
Builds a NetworkX graph
 ↓
Generates embeddings for graph entities
 ↓
Creates data/entity_index.json
 ↓
Starts the question-answering interface
```

After initialization, the command line displays:

```text
Graph RAG
Type 'exit' to quit.

Question:
```

Enter a question about the paper.

For each query, Graph RAG:

```text
Question
 ↓
Query embedding
 ↓
Relevant seed entities
 ↓
Knowledge graph traversal
 ↓
Retrieved semantic triples
 ↓
LLM
 ↓
Answer with exact triple citations
```

The answer is generated only from retrieved knowledge graph triples.

Example citation format:

```text
[Transformer] -> [utilizes] -> [multi-head attention]
```

To exit:

```text
exit
```

### Rebuilding the Entity Index

The entity index is automatically reused after it has been created.

If `kg_triples.json` is regenerated, delete the existing index before running Graph RAG again:

### Windows

```bash
del data\entity_index.json
```

### macOS / Linux

```bash
rm data/entity_index.json
```

The next execution of:

```bash
python graph_rag.py
```

will automatically regenerate the entity embeddings.

---

## 8. Vector RAG - Build and Run the Vector RAG Pipeline

Run:

```bash
python vector_rag.py
```

If no existing ChromaDB vector database is found, the script automatically performs:

```text
PDF
 ↓
Load pages
 ↓
Split into text chunks
 ↓
Generate embeddings
 ↓
Store chunks in ChromaDB
 ↓
Start the question-answering interface
```

After initialization, the command line displays:

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

The answer is generated only from the retrieved text chunks.

To exit:

```text
exit
```

---

## 9. Vector RAG - Rebuild the Vector Database

After the first execution, the existing ChromaDB database is reused.

If any of the following parameters are changed:

```env
EMBEDDING_MODEL
VECTOR_CHUNK_SIZE
VECTOR_CHUNK_OVERLAP
```

the vector database should be rebuilt.

Run:

```bash
python vector_rag.py --rebuild
```

This deletes the existing local ChromaDB index and reconstructs it using the current configuration.

---

## 10. Retrieval Configuration

### Graph RAG

The main Graph RAG retrieval parameters are:

```env
GRAPH_SEED_K=5
GRAPH_MAX_HOPS=1
GRAPH_MAX_TRIPLES=20
```

* `GRAPH_SEED_K`: number of semantically relevant entities used as graph entry points.
* `GRAPH_MAX_HOPS`: maximum graph traversal depth.
* `GRAPH_MAX_TRIPLES`: maximum number of retrieved triples supplied to the generation model.

### Vector RAG

The main Vector RAG retrieval parameters are:

```env
VECTOR_CHUNK_SIZE=1000
VECTOR_CHUNK_OVERLAP=150
VECTOR_TOP_K=5
```

* `VECTOR_CHUNK_SIZE`: size of text chunks used for vector indexing.
* `VECTOR_CHUNK_OVERLAP`: overlap between neighboring chunks.
* `VECTOR_TOP_K`: number of chunks retrieved for each query.

---

## 11. Generation Parameters

Both Graph RAG and Vector RAG use the same generation parameters by default:

```env
TEMPERATURE=0.1
TOP_P=0.3
```

This makes it possible to compare the two retrieval methods while keeping the generation configuration consistent.

These values can be changed in `.env` for parameter experiments.

For example:

```env
TEMPERATURE=0.7
TOP_P=0.25
```

or:

```env
TEMPERATURE=1.4
TOP_P=0.98
```

When only generation parameters are changed, the knowledge graph, entity index, and vector database do not need to be rebuilt.

---

## 12. Running the Two Systems

A typical complete workflow is:

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Configure the API key

```bash
copy .env.template .env
```

Edit `.env` and add the OpenAI API key.

### Step 3 — Build the knowledge graph

```bash
python build_kg.py
```

### Step 4 — Run Graph RAG

```bash
python graph_rag.py
```

### Step 5 — Run Vector RAG

```bash
python vector_rag.py
```

---

## 13. Methodological Difference

The main difference between the two systems is the retrieval unit.

### Graph RAG

```text
Query
→ semantic entity matching
→ graph traversal
→ semantic triples
→ LLM
```

The embedding model is used to identify relevant entry points in the graph, while the actual evidence is retrieved through graph traversal.

### Vector RAG

```text
Query
→ semantic similarity search
→ text chunks
→ LLM
```

The embedding model directly retrieves relevant portions of the original document.

This allows the project to compare structured graph-based retrieval with conventional vector-based semantic retrieval while using the same source document and generation model.