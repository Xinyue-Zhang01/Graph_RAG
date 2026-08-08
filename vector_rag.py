import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "gpt-4o",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

PDF_PATH = Path(
    "data/attention_is_all_you_need.pdf"
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "attention_is_all_you_need",
)

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "./chroma_db",
)

CHUNK_SIZE = int(
    os.getenv("VECTOR_CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("VECTOR_CHUNK_OVERLAP", "150")
)

DEFAULT_TOP_K = int(
    os.getenv("VECTOR_TOP_K", "5")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.1")
)

TOP_P = float(
    os.getenv("TOP_P", "0.3")
)


# --------------------------------------------------
# OpenAI Clients
# --------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
)


# --------------------------------------------------
# 1. Load the Paper
# --------------------------------------------------

def load_paper():
    """Load the required source paper: 'Attention Is All You Need'."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Paper not found: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from {PDF_PATH}.")
    return documents


# --------------------------------------------------
# 2. Chunk the Paper
# --------------------------------------------------

def chunk_documents(documents):
    """Split the paper into overlapping text chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    print(
        f"Created {len(chunks)} chunks from {len(documents)} pages."
    )

    return chunks


# --------------------------------------------------
# 3. ChromaDB Vector Store
# --------------------------------------------------

def get_vector_store() -> Chroma:
    """Create or load the local persistent Chroma vector store."""

    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def build_vector_database(reset: bool = True) -> Chroma:
    """
    Full ingestion pipeline:
        1. Load PDF
        2. Chunk pages
        3. Embed chunks
        4. Store chunks in ChromaDB
    """

    if reset and os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"Deleted old Chroma database: {CHROMA_PERSIST_DIR}")

    documents = load_paper()
    chunks = chunk_documents(documents)

    vector_store = get_vector_store()

    ids = [
        f"attention-chunk-{i}"
        for i in range(len(chunks))
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(f"Inserted {len(chunks)} chunks into ChromaDB.")
    print(f"Collection: {CHROMA_COLLECTION}")
    print(f"Persist directory: {CHROMA_PERSIST_DIR}")

    return vector_store


# --------------------------------------------------
# 4. Initialize Vector RAG
# --------------------------------------------------

def initialize_vector_rag(rebuild: bool = False) -> Chroma:
    """
    Prepare the vector database before showing the question prompt.

    Reuse an existing populated Chroma collection unless --rebuild is used.
    """

    print("\nInitializing Vector RAG...")

    if rebuild:
        vector_store = build_vector_database(reset=True)
    else:
        vector_store = get_vector_store()
        stored_data = vector_store.get()
        stored_ids = stored_data.get("ids", [])

        if not stored_ids:
            print("No existing vector index found. Building ChromaDB...")
            vector_store = build_vector_database(reset=False)
        else:
            print(
                f"Loading existing ChromaDB with {len(stored_ids)} chunks."
            )

    print("Vector RAG ready.\n")
    return vector_store


# --------------------------------------------------
# 5. Retrieve Relevant Chunks
# --------------------------------------------------

def retrieve_context(
    question: str,
    vector_store: Chroma,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[str, List[Tuple[Any, float]]]:
    """Retrieve the most relevant chunks for a question."""

    results = vector_store.similarity_search_with_score(
        query=question,
        k=top_k,
    )

    context_parts = []

    for i, (doc, distance) in enumerate(results, start=1):
        page = doc.metadata.get("page", "unknown")

        if isinstance(page, int):
            display_page = page + 1
        else:
            display_page = page

        chunk_id = doc.metadata.get("chunk_id", "unknown")

        context_parts.append(
            f"[Source {i}]\n"
            f"Page: {display_page}\n"
            f"Chunk: {chunk_id}\n"
            f"Vector distance: {distance:.6f}\n"
            f"Text:\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)
    return context, results


# --------------------------------------------------
# 6. Generate Response with LLM
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a question-answering system for the academic paper
"Attention Is All You Need".

Answer using ONLY the retrieved text chunks supplied in the context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts that are not supported by the retrieved text.
3. If the retrieved text is insufficient, explicitly say that the
   retrieved context does not contain enough information.
4. Cite factual claims using [Source 1], [Source 2], etc.
5. You may cite multiple sources when necessary.
6. Do not invent source IDs.
7. Give a concise and accurate natural-language answer.
"""


def generate_response(
    question: str,
    context: str,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """Generate an answer based only on retrieved Vector RAG context."""

    user_prompt = f"""
Retrieved Context:

{context}

Question:

{question}

Answer the question using only the retrieved context.
"""

    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ],
        temperature=temperature,
        top_p=top_p,
    )

    return response.choices[0].message.content


# --------------------------------------------------
# 7. Complete Vector RAG Pipeline
# --------------------------------------------------

def vector_rag(
    query: str,
    vector_store: Chroma,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Complete RAG chatbot function.
    
        Pipeline:
            1. Retrieve relevant context
            2. Generate answer
            3. Return answer and sources
    """

    context, sources = retrieve_context(
        question=query,
        vector_store=vector_store,
        top_k=top_k,
    )

    answer = generate_response(
        question=query,
        context=context,
        temperature=temperature,
        top_p=top_p,
    )

    retrieved_chunks = []

    for doc, distance in sources:
        page = doc.metadata.get("page", "unknown")

        if isinstance(page, int):
            display_page = page + 1
        else:
            display_page = page

        retrieved_chunks.append(
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", str(PDF_PATH)),
                "page": display_page,
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "distance": float(distance),
            }
        )

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
    }


# --------------------------------------------------
# 8. Command-Line Chat
# --------------------------------------------------

def run_cli(vector_store: Chroma):
    """Interactive command-line Vector RAG."""

    print("Vector RAG")
    print("Source: Attention Is All You Need")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        result = vector_rag(
            query=query,
            vector_store=vector_store,
        )

        print("\nRetrieved chunks:")

        for i, source in enumerate(
            result["retrieved_chunks"],
            start=1,
        ):
            print(
                f"- [Source {i}] "
                f"page={source['page']}, "
                f"chunk={source['chunk_id']}, "
                f"distance={source['distance']:.4f}"
            )

        print("\nAnswer:")
        print(result["answer"])

        print("\n" + "=" * 70 + "\n")


# --------------------------------------------------
# 9. Main
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Vector RAG baseline for 'Attention Is All You Need'"
        )
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Delete the existing ChromaDB and rebuild the vector index "
            "before starting chat."
        ),
    )

    args = parser.parse_args()

    vector_store = initialize_vector_rag(
        rebuild=args.rebuild
    )

    run_cli(vector_store)
