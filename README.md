# Graph-Based Retrieval-Augmented Generation

This repository implements and evaluates two Retrieval-Augmented Generation (RAG) pipelines using **“Attention Is All You Need”** by Vaswani et al. (2017) as the sole knowledge source:

- **Graph RAG** — retrieves structured `(Subject, Predicate, Object)` triples from a knowledge graph generated with KGGen.
- **Vector RAG** — retrieves semantically similar text chunks from a local ChromaDB vector database.

The project compares the two retrieval paradigms under a shared source document and generation model, and includes both a five-query validation experiment and a parameter analysis of `temperature` and `top_p`.

---

## 1. Repository Structure

```text
Graph_RAG/
│
├── data/
│   ├── attention_is_all_you_need.pdf
│   ├── attention_is_all_you_need.txt
│   └── kg_triples.json
│
├── results/
│   ├── Experimental_Validation_A.txt
│   ├── Experimental_Validation_B.txt
│   └── report.pdf
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

Two additional indexes are generated locally when the systems are run:

```text
data/entity_index.json   # Graph entity embeddings
chroma_db/               # Persistent ChromaDB vector index
```

These local indexes are excluded from Git and can be regenerated from the repository contents.

---

## 2. Source Document

Both systems use the same source:

**Vaswani, A. et al. (2017). “Attention Is All You Need.”**

Original PDF: https://arxiv.org/pdf/1706.03762.pdf

The local source file is:

```text
data/attention_is_all_you_need.pdf
```

No external document is used as retrieval evidence during the experiments.

---

## 3. System Overview

### Graph RAG

```text
Paper PDF
   ↓
Text extraction
   ↓
KGGen
   ↓
(Subject, Predicate, Object) triples
   ↓
Hybrid entity linking
   ↓
Graph expansion
   ↓
Query-aware triple reranking
   ↓
Top triples
   ↓
LLM answer with exact triple citations
```

### Vector RAG

```text
Paper PDF
   ↓
Page loading
   ↓
Overlapping text chunks
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Vector similarity search
   ↓
Top text chunks
   ↓
LLM answer with source citations
```

The important methodological difference is that **Graph RAG retrieves structured graph evidence**, whereas **Vector RAG retrieves sections of the original text directly**.

---

## 4. Requirements

- Python 3.10 or later
- OpenAI API key
- Internet access for OpenAI API calls

Install the dependencies from:

```text
requirements.txt
```

---

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/Xinyue-Zhang01/Graph_RAG.git
cd Graph_RAG
```

Create a virtual environment.

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

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Environment Configuration

Create `.env` from the supplied template.

### Windows

```bash
copy .env.template .env
```

### macOS / Linux

```bash
cp .env.template .env
```

Add a valid OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

The default experimental configuration is:

```env
# Models
KG_MODEL=openai/gpt-4o
GENERATION_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# Generation
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

The local `.env` file is ignored by Git and should not be committed.

---

# Graph RAG

## 7. Build the Knowledge Graph

Run:

```bash
python build_kg.py
```

The script:

1. extracts text from the PDF with PyMuPDF;
2. stores the extracted text in `data/attention_is_all_you_need.txt`;
3. sends the source text to KGGen;
4. converts KGGen relations into JSON triples;
5. stores the graph data in `data/kg_triples.json`.

Triples use the format:

```json
{
  "subject": "Multi-Head Attention",
  "predicate": "attends to",
  "object": "representation subspaces"
}
```

KG construction uses the source paper directly without a task-specific extraction context. This avoids explicitly tuning graph construction toward the later evaluation questions.

If the extracted `.txt` file already exists, `build_kg.py` reuses it instead of extracting the PDF again.

---

## 8. Run Graph RAG

```bash
python graph_rag.py
```

On first use, the system creates entity embeddings and stores them locally in:

```text
data/entity_index.json
```

The command-line interface then accepts questions about the paper.

```text
Graph RAG
Hybrid retrieval: lexical + semantic seeds, graph expansion, triple reranking
Type 'exit' to quit.

Question:
```

The generation model is restricted to the retrieved graph triples. Every factual claim must cite an exact supporting triple in the form:

```text
[Subject] -> [Predicate] -> [Object]
```

Example:

```text
[Transformer] -> [relies on] -> [self-attention]
```

If the retrieved graph evidence is insufficient, the model is instructed to say so instead of supplementing the answer with outside knowledge.

---

## 9. Graph Retrieval Method

The current retriever uses three stages.

### 9.1 Hybrid Entity Linking

A question is linked to graph entities through:

- **lexical matching** — entity names explicitly appearing in the normalized question receive a seed score of `1.0`;
- **semantic matching** — the query embedding is compared with stored entity embeddings using cosine similarity.

The two result sets are merged, with lexical matches taking priority when the same entity appears in both.

### 9.2 Graph Expansion

Seed entities are used as entry points into a NetworkX `MultiDiGraph`.

The retriever follows both incoming and outgoing edges up to:

```env
GRAPH_MAX_HOPS=2
```

All reached relations form a candidate triple set. The candidate set is intentionally collected before applying the final context-size limit.

### 9.3 Query-Aware Triple Reranking

Each candidate triple is serialized as:

```text
subject predicate object
```

and embedded for comparison with the query.

The final score combines semantic relevance, seed relevance, and graph distance:

```text
final score
=
(query–triple similarity × GRAPH_TRIPLE_SIM_WEIGHT)
+
(seed relevance × GRAPH_SEED_SCORE_WEIGHT)
-
(hop penalty)
```

Default weights:

```env
GRAPH_TRIPLE_SIM_WEIGHT=0.7
GRAPH_SEED_SCORE_WEIGHT=0.3
GRAPH_HOP_PENALTY=0.05
```

Only the top:

```env
GRAPH_MAX_TRIPLES=20
```

triples are supplied to the generation model.

The CLI prints the seed entities, candidate count, retrieved triples, semantic scores, final scores, and hop distances to make the retrieval process inspectable.

---

## 10. Rebuild the Entity Index

If `kg_triples.json` is regenerated or the embedding model is changed, delete the old entity index before running Graph RAG again.

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

will regenerate the index automatically.

---

# Vector RAG

## 11. Run Vector RAG

```bash
python vector_rag.py
```

If no existing ChromaDB collection is found, the pipeline automatically:

1. loads the PDF pages;
2. splits them into overlapping text chunks;
3. embeds the chunks;
4. stores them in ChromaDB;
5. starts the command-line interface.

For each question:

```text
Question
   ↓
Query embedding
   ↓
Vector similarity search
   ↓
Top-5 chunks
   ↓
LLM
   ↓
Answer with [Source n] citations
```

The generation model is instructed to use only the retrieved chunks.

---

## 12. Vector RAG Configuration

```env
VECTOR_CHUNK_SIZE=1000
VECTOR_CHUNK_OVERLAP=150
VECTOR_TOP_K=5
```

- `VECTOR_CHUNK_SIZE`: size of each indexed text chunk;
- `VECTOR_CHUNK_OVERLAP`: overlap between adjacent chunks;
- `VECTOR_TOP_K`: number of chunks returned for each query.

The persistent local database is stored in:

```text
chroma_db/
```

If the embedding model, chunk size, or chunk overlap changes, rebuild it with:

```bash
python vector_rag.py --rebuild
```

---

# Experimental Evaluation

## 13. Experimental Validation A — Graph RAG vs. Vector RAG

The full raw experiment record is available at:

- [`results/Experimental_Validation_A.txt`](results/Experimental_Validation_A.txt)

Five questions were tested using the default generation profile:

```text
Temperature = 0.1
Top-p = 0.3
```

The questions cover author affiliations, dropout, training data, regularization, and the motivation for self-attention.

---

## 14. Experimental Validation B — Parameter Analysis

The full raw experiment record is available at:

- [`results/Experimental_Validation_B.txt`](results/Experimental_Validation_B.txt)

The same factual query was evaluated with three generation profiles:

| Profile | Temperature | Top-p |
|---|---:|---:|
| 1 | 0.1 | 0.3 |
| 2 | 0.7 | 0.25 |
| 3 | 1.4 | 0.98 |

For Graph RAG, the same retrieved triple set was used across the three generation profiles. For Vector RAG, the same chunks were retrieved; only negligible floating-point differences appeared in the displayed vector distances.

---

## 15. Full Report

The final written report is included in the repository:

- [`results/report.pdf`](results/report.pdf)

It contains the comparison and parameter-analysis discussion based on the recorded experimental outputs.

---

## 16. Models Used

The repository configuration records the following API model identifiers:

```text
KG generation:     openai/gpt-4o (through KGGen)
Answer generation: gpt-4o
Embeddings:        text-embedding-3-small
```

The experiment records use these configured identifiers; a separate dated backend snapshot was not recorded.
