import json
import os
import re
from pathlib import Path

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

KG_PATH = Path("data/kg_triples.json")
ENTITY_INDEX_PATH = Path("data/entity_index.json")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "gpt-4o",
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

SEED_K = int(
    os.getenv("GRAPH_SEED_K", "5")
)

MAX_HOPS = int(
    os.getenv("GRAPH_MAX_HOPS", "2")
)

MAX_TRIPLES = int(
    os.getenv("GRAPH_MAX_TRIPLES", "20")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.1")
)

TOP_P = float(
    os.getenv("TOP_P", "0.3")
)

TRIPLE_SIM_WEIGHT = float(
    os.getenv(
        "GRAPH_TRIPLE_SIM_WEIGHT",
        "0.7",
    )
)

SEED_SCORE_WEIGHT = float(
    os.getenv(
        "GRAPH_SEED_SCORE_WEIGHT",
        "0.3",
    )
)

HOP_PENALTY = float(
    os.getenv(
        "GRAPH_HOP_PENALTY",
        "0.05",
    )
)

TRIPLE_EMBEDDING_CACHE = {}


# --------------------------------------------------
# 1. Load Knowledge Graph
# --------------------------------------------------

def load_triples():
    with KG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_graph(triples):
    graph = nx.MultiDiGraph()

    for triple_id, triple in enumerate(triples):
        subject = triple["subject"]
        predicate = triple["predicate"]
        obj = triple["object"]

        graph.add_edge(
            subject,
            obj,
            key=triple_id,
            triple_id=triple_id,
            predicate=predicate,
        )

    return graph


# --------------------------------------------------
# 2. Embeddings
# --------------------------------------------------

def get_embeddings(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]


def build_entity_index(graph):
    entities = list(graph.nodes)

    print(
        f"Creating embeddings for "
        f"{len(entities)} KG entities..."
    )

    embeddings = get_embeddings(entities)

    entity_index = {
        "entities": entities,
        "embeddings": embeddings,
    }

    ENTITY_INDEX_PATH.write_text(
        json.dumps(
            entity_index,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved entity index to "
        f"{ENTITY_INDEX_PATH}"
    )

    return entity_index


def load_or_create_entity_index(graph):
    if ENTITY_INDEX_PATH.exists():
        print(
            f"Loading existing entity index "
            f"from {ENTITY_INDEX_PATH}..."
        )

        return json.loads(
            ENTITY_INDEX_PATH.read_text(
                encoding="utf-8"
            )
        )

    return build_entity_index(graph)


# --------------------------------------------------
# 3. Similarity Helpers
# --------------------------------------------------

def cosine_similarity(
    query_vector,
    matrix,
):
    query_vector = np.array(
        query_vector,
        dtype=np.float32,
    )

    matrix = np.array(
        matrix,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(
        query_vector
    )

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    scores = (
        matrix @ query_vector
    ) / (
        matrix_norms * query_norm
        + 1e-12
    )

    return scores


def normalize_text(text):
    text = text.lower()

    text = re.sub(
        r"[‐‑‒–—−]",
        "-",
        text,
    )

    text = text.replace(
        "’",
        "'",
    )

    text = re.sub(
        r"[^a-z0-9\s\-']",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# --------------------------------------------------
# 4. Hybrid Entity Linking
# --------------------------------------------------

def find_lexical_entities(
    query,
    entities,
):
    normalized_query = normalize_text(
        query
    )

    matches = []

    for entity in entities:
        normalized_entity = (
            normalize_text(entity)
        )

        if len(normalized_entity) < 4:
            continue

        if normalized_entity in normalized_query:
            matches.append(
                {
                    "entity": entity,
                    "score": 1.0,
                    "method": "lexical",
                }
            )

    matches.sort(
        key=lambda item: (
            -len(
                normalize_text(
                    item["entity"]
                )
            ),
            item["entity"].lower(),
        )
    )

    return matches


def find_semantic_entities(
    query_embedding,
    entity_index,
    top_k=SEED_K,
):
    entities = entity_index[
        "entities"
    ]

    embeddings = entity_index[
        "embeddings"
    ]

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "entity": entities[index],
                "score": float(
                    scores[index]
                ),
                "method": "semantic",
            }
        )

    return results


def find_seed_entities(
    query,
    entity_index,
    top_k=SEED_K,
):
    query_embedding = get_embeddings(
        [query]
    )[0]

    entities = entity_index[
        "entities"
    ]

    lexical_results = (
        find_lexical_entities(
            query=query,
            entities=entities,
        )
    )

    semantic_results = (
        find_semantic_entities(
            query_embedding=query_embedding,
            entity_index=entity_index,
            top_k=top_k,
        )
    )

    merged = {}

    for item in semantic_results:
        merged[item["entity"]] = item

    for item in lexical_results:
        merged[item["entity"]] = item

    results = list(
        merged.values()
    )

    results.sort(
        key=lambda item: (
            0
            if item["method"] == "lexical"
            else 1,
            -item["score"],
        )
    )

    return (
        results,
        query_embedding,
    )


# --------------------------------------------------
# 5. Graph Expansion
# --------------------------------------------------

def add_candidate(
    retrieved,
    triple_id,
    subject,
    predicate,
    obj,
    hop,
    seed_score,
):
    candidate = {
        "triple_id": triple_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "hop": hop,
        "seed_score": seed_score,
    }

    if triple_id not in retrieved:
        retrieved[triple_id] = candidate
        return

    existing = retrieved[triple_id]

    existing["hop"] = min(
        existing["hop"],
        hop,
    )

    existing["seed_score"] = max(
        existing["seed_score"],
        seed_score,
    )


def traverse_graph(
    graph,
    seed_entities,
    max_hops=MAX_HOPS,
):
    retrieved = {}
    visited_nodes = set()

    frontier = {
        item["entity"]: item["score"]
        for item in seed_entities
    }

    for hop in range(
        1,
        max_hops + 1,
    ):
        next_frontier = {}

        for node, seed_score in frontier.items():
            if node in visited_nodes:
                continue

            visited_nodes.add(node)

            for (
                source,
                target,
                key,
                data,
            ) in graph.in_edges(
                node,
                keys=True,
                data=True,
            ):
                triple_id = data["triple_id"]

                add_candidate(
                    retrieved=retrieved,
                    triple_id=triple_id,
                    subject=source,
                    predicate=data["predicate"],
                    obj=target,
                    hop=hop,
                    seed_score=seed_score,
                )

                if source not in visited_nodes:
                    next_frontier[
                        source
                    ] = max(
                        next_frontier.get(
                            source,
                            0.0,
                        ),
                        seed_score,
                    )

            for (
                source,
                target,
                key,
                data,
            ) in graph.out_edges(
                node,
                keys=True,
                data=True,
            ):
                triple_id = data["triple_id"]

                add_candidate(
                    retrieved=retrieved,
                    triple_id=triple_id,
                    subject=source,
                    predicate=data["predicate"],
                    obj=target,
                    hop=hop,
                    seed_score=seed_score,
                )

                if target not in visited_nodes:
                    next_frontier[
                        target
                    ] = max(
                        next_frontier.get(
                            target,
                            0.0,
                        ),
                        seed_score,
                    )

        frontier = next_frontier

    return list(
        retrieved.values()
    )


# --------------------------------------------------
# 6. Query-Aware Triple Reranking
# --------------------------------------------------

def triple_to_text(triple):
    return (
        f"{triple['subject']} "
        f"{triple['predicate']} "
        f"{triple['object']}"
    )


def get_cached_triple_embeddings(
    triple_texts
):
    missing = [
        text
        for text in triple_texts
        if text not in TRIPLE_EMBEDDING_CACHE
    ]

    if missing:
        new_embeddings = get_embeddings(
            missing
        )

        for text, embedding in zip(
            missing,
            new_embeddings,
        ):
            TRIPLE_EMBEDDING_CACHE[
                text
            ] = embedding

    return [
        TRIPLE_EMBEDDING_CACHE[
            text
        ]
        for text in triple_texts
    ]


def rerank_triples(
    query_embedding,
    candidate_triples,
    max_triples=MAX_TRIPLES,
):
    if not candidate_triples:
        return []

    triple_texts = [
        triple_to_text(triple)
        for triple in candidate_triples
    ]

    triple_embeddings = (
        get_cached_triple_embeddings(
            triple_texts
        )
    )

    semantic_scores = (
        cosine_similarity(
            query_embedding,
            triple_embeddings,
        )
    )

    reranked = []

    for triple, semantic_score in zip(
        candidate_triples,
        semantic_scores,
    ):
        hop_penalty = (
            HOP_PENALTY
            * max(
                triple["hop"] - 1,
                0,
            )
        )

        final_score = (
            TRIPLE_SIM_WEIGHT
            * float(semantic_score)
            + SEED_SCORE_WEIGHT
            * triple["seed_score"]
            - hop_penalty
        )

        ranked_triple = dict(
            triple
        )

        ranked_triple[
            "semantic_score"
        ] = float(
            semantic_score
        )

        ranked_triple[
            "final_score"
        ] = float(
            final_score
        )

        reranked.append(
            ranked_triple
        )

    reranked.sort(
        key=lambda item: (
            -item["final_score"],
            item["hop"],
        )
    )

    return reranked[
        :max_triples
    ]


# --------------------------------------------------
# 7. Complete Retrieval Pipeline
# --------------------------------------------------

def retrieve_triples(
    query,
    graph,
    entity_index,
    seed_k=SEED_K,
    max_hops=MAX_HOPS,
    max_triples=MAX_TRIPLES,
):
    (
        seed_entities,
        query_embedding,
    ) = find_seed_entities(
        query=query,
        entity_index=entity_index,
        top_k=seed_k,
    )

    candidate_triples = traverse_graph(
        graph=graph,
        seed_entities=seed_entities,
        max_hops=max_hops,
    )

    retrieved_triples = rerank_triples(
        query_embedding=query_embedding,
        candidate_triples=candidate_triples,
        max_triples=max_triples,
    )

    return (
        seed_entities,
        candidate_triples,
        retrieved_triples,
    )


# --------------------------------------------------
# 8. Format Triples for Prompt
# --------------------------------------------------

def format_triples_for_prompt(
    triples
):
    lines = []

    for triple in triples:
        triple_text = (
            f"[{triple['subject']}] "
            f"-> "
            f"[{triple['predicate']}] "
            f"-> "
            f"[{triple['object']}]"
        )

        lines.append(
            triple_text
        )

    return "\n".join(
        lines
    )


# --------------------------------------------------
# 9. LLM Response Generation
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a question-answering system for the academic
paper "Attention Is All You Need".

Answer using ONLY the supplied knowledge graph triples.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts that are not supported by the
   retrieved triples.
3. If the retrieved triples are insufficient, say that
   the retrieved knowledge graph does not contain enough
   information.
4. Cite every factual claim using the exact supporting
   knowledge graph triple.
5. Copy cited triples exactly as they appear in the
   retrieved triples. Do not paraphrase or modify them.
6. Use this exact citation format:
   [Subject] -> [Predicate] -> [Object]
7. Place citations directly after the factual claim they
   support.
8. You may cite multiple triples when necessary.
9. Never create a triple that is not present in the
   retrieved triples.
10. Give a concise natural-language answer.
"""


def generate_answer(
    query,
    retrieved_triples,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    context = (
        format_triples_for_prompt(
            retrieved_triples
        )
    )

    user_prompt = f"""
Question:

{query}

Retrieved Knowledge Graph Triples:

{context}

Answer the question using only these triples.

For every factual claim, cite the supporting triple
exactly as it appears above.
"""

    response = (
        client
        .chat.completions
        .create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            top_p=top_p,
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# --------------------------------------------------
# 10. Initialize Graph RAG
# --------------------------------------------------

def initialize_graph_rag():
    print(
        "\nInitializing Graph RAG..."
    )

    triples = load_triples()

    print(
        f"Loaded "
        f"{len(triples)} triples."
    )

    graph = build_graph(
        triples
    )

    print(
        f"Built graph with "
        f"{graph.number_of_nodes()} "
        f"nodes and "
        f"{graph.number_of_edges()} "
        f"edges."
    )

    entity_index = (
        load_or_create_entity_index(
            graph
        )
    )

    print(
        f"Entity index ready with "
        f"{len(entity_index['entities'])} "
        f"entities."
    )

    print(
        "Graph RAG ready.\n"
    )

    return (
        graph,
        entity_index,
    )


# --------------------------------------------------
# 11. Complete Graph RAG Pipeline
# --------------------------------------------------

def graph_rag(
    query,
    graph,
    entity_index,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    seed_k=SEED_K,
    max_hops=MAX_HOPS,
    max_triples=MAX_TRIPLES,
):
    (
        seed_entities,
        candidate_triples,
        retrieved_triples,
    ) = retrieve_triples(
        query=query,
        graph=graph,
        entity_index=entity_index,
        seed_k=seed_k,
        max_hops=max_hops,
        max_triples=max_triples,
    )

    answer = generate_answer(
        query=query,
        retrieved_triples=(
            retrieved_triples
        ),
        temperature=temperature,
        top_p=top_p,
    )

    return {
        "query": query,
        "seed_entities": seed_entities,
        "candidate_count": len(
            candidate_triples
        ),
        "retrieved_triples": (
            retrieved_triples
        ),
        "answer": answer,
    }


# --------------------------------------------------
# 12. Command-Line Chat
# --------------------------------------------------

def run_cli(
    graph,
    entity_index,
):
    print(
        "Graph RAG"
    )

    print(
        "Hybrid retrieval: "
        "lexical + semantic seeds, "
        "graph expansion, "
        "triple reranking"
    )

    print(
        "Type 'exit' to quit.\n"
    )

    while True:
        query = input(
            "Question: "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        result = graph_rag(
            query=query,
            graph=graph,
            entity_index=entity_index,
        )

        print(
            "\nSeed entities:"
        )

        for item in result[
            "seed_entities"
        ]:
            print(
                f"- "
                f"{item['entity']} "
                f"("
                f"{item['score']:.3f}, "
                f"{item['method']}"
                f")"
            )

        print(
            "\nGraph candidates:"
        )

        print(
            f"- "
            f"{result['candidate_count']} "
            f"candidate triples "
            f"before reranking"
        )

        print(
            "\nRetrieved triples "
            "after reranking:"
        )

        for triple in result[
            "retrieved_triples"
        ]:
            print(
                f"- "
                f"[{triple['subject']}] "
                f"-> "
                f"[{triple['predicate']}] "
                f"-> "
                f"[{triple['object']}] "
                f"(score="
                f"{triple['final_score']:.3f}, "
                f"semantic="
                f"{triple['semantic_score']:.3f}, "
                f"hop="
                f"{triple['hop']}"
                f")"
            )

        print(
            "\nAnswer:"
        )

        print(
            result["answer"]
        )

        print(
            "\n"
            + "=" * 70
            + "\n"
        )


if __name__ == "__main__":
    graph, entity_index = (
        initialize_graph_rag()
    )

    run_cli(
        graph,
        entity_index,
    )
