import json
import os
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
    os.getenv("GRAPH_MAX_HOPS", "1")
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


# --------------------------------------------------
# 1. Load Knowledge Graph
# --------------------------------------------------

def load_triples():
    with KG_PATH.open(
        "r",
        encoding="utf-8"
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
# 2. Entity Embeddings
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
# 3. Find Seed Entities
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


def find_seed_entities(
    query,
    entity_index,
    top_k=5,
):

    query_embedding = get_embeddings(
        [query]
    )[0]

    entities = entity_index["entities"]
    embeddings = entity_index["embeddings"]

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
            }
        )

    return results


# --------------------------------------------------
# 4. Traverse Knowledge Graph
# --------------------------------------------------

def traverse_graph(
    graph,
    seed_entities,
    max_hops=1,
    max_triples=20,
):

    retrieved = {}
    visited_nodes = set()

    frontier = {
        item["entity"]: item["score"]
        for item in seed_entities
    }

    for hop in range(max_hops):

        next_frontier = {}

        for node, seed_score in frontier.items():

            if node in visited_nodes:
                continue

            visited_nodes.add(node)

            # Incoming edges
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

                triple_id = data[
                    "triple_id"
                ]

                retrieved[triple_id] = {
                    "triple_id": triple_id,
                    "subject": source,
                    "predicate": data[
                        "predicate"
                    ],
                    "object": target,
                    "hop": hop + 1,
                    "seed_score": seed_score,
                }

                if (
                    source
                    not in visited_nodes
                ):
                    next_frontier[
                        source
                    ] = seed_score

            # Outgoing edges
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

                triple_id = data[
                    "triple_id"
                ]

                retrieved[triple_id] = {
                    "triple_id": triple_id,
                    "subject": source,
                    "predicate": data[
                        "predicate"
                    ],
                    "object": target,
                    "hop": hop + 1,
                    "seed_score": seed_score,
                }

                if (
                    target
                    not in visited_nodes
                ):
                    next_frontier[
                        target
                    ] = seed_score

        frontier = next_frontier

    results = list(
        retrieved.values()
    )

    results.sort(
        key=lambda item: (
            item["hop"],
            -item["seed_score"],
        )
    )

    return results[:max_triples]


# --------------------------------------------------
# 5. Retrieve Relevant Triples
# --------------------------------------------------

def retrieve_triples(
    query,
    graph,
    entity_index,
    seed_k=SEED_K,
    max_hops=MAX_HOPS,
    max_triples=MAX_TRIPLES,
):

    seed_entities = find_seed_entities(
        query=query,
        entity_index=entity_index,
        top_k=seed_k,
    )

    retrieved_triples = traverse_graph(
        graph=graph,
        seed_entities=seed_entities,
        max_hops=max_hops,
        max_triples=max_triples,
    )

    return (
        seed_entities,
        retrieved_triples,
    )


# --------------------------------------------------
# 6. Prepare Triple Citations
# --------------------------------------------------

def format_triples_for_prompt(
    triples
):
    lines = []
    citation_map = {}

    for index, triple in enumerate(
        triples,
        start=1,
    ):

        citation_id = (
            f"T{index:03d}"
        )

        triple_text = (
            f"[{triple['subject']}] "
            f"-> "
            f"[{triple['predicate']}] "
            f"-> "
            f"[{triple['object']}]"
        )

        citation_map[
            citation_id
        ] = triple_text

        lines.append(
            f"[{citation_id}] "
            f"{triple_text}"
        )

    return (
        "\n".join(lines),
        citation_map,
    )


# --------------------------------------------------
# 7. LLM Response Generation
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
4. Cite every factual claim using the supporting triple ID,
   for example [T001].
5. You may cite multiple triples when necessary.
6. Do not invent or modify citation IDs.
7. Give a concise natural-language answer.
"""


def generate_answer(
    query,
    retrieved_triples,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):

    context, citation_map = (
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
"""

    response = (
        client.chat.completions.create(
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

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return answer, citation_map


# --------------------------------------------------
# 8. Expand Citation IDs into Exact Triples
# --------------------------------------------------

def expand_citations(
    answer,
    citation_map,
):

    for (
        citation_id,
        triple_text,
    ) in citation_map.items():

        answer = answer.replace(
            f"[{citation_id}]",
            triple_text,
        )

    return answer


# --------------------------------------------------
# 9. Complete Graph RAG Pipeline
# --------------------------------------------------

def initialize_graph_rag():

    print("\nInitializing Graph RAG...")

    # Load triples
    triples = load_triples()

    print(
        f"Loaded {len(triples)} triples."
    )

    # Build NetworkX graph
    graph = build_graph(triples)

    print(
        f"Built graph with "
        f"{graph.number_of_nodes()} nodes "
        f"and {graph.number_of_edges()} edges."
    )

    # Load or create entity embeddings
    entity_index = (
        load_or_create_entity_index(
            graph
        )
    )

    print(
        f"Entity index ready with "
        f"{len(entity_index['entities'])} entities."
    )

    print("Graph RAG ready.\n")

    return graph, entity_index


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
        retrieved_triples,
    ) = retrieve_triples(
        query=query,
        graph=graph,
        entity_index=entity_index,
        seed_k=seed_k,
        max_hops=max_hops,
        max_triples=max_triples,
    )

    (
        raw_answer,
        citation_map,
    ) = generate_answer(
        query=query,
        retrieved_triples=retrieved_triples,
        temperature=temperature,
        top_p=top_p,
    )

    final_answer = expand_citations(
        raw_answer,
        citation_map,
    )

    return {
        "query": query,
        "seed_entities": seed_entities,
        "retrieved_triples": retrieved_triples,
        "answer": final_answer,
    }


# --------------------------------------------------
# 10. Command-Line Chat
# --------------------------------------------------

def run_cli(
    graph,
    entity_index,
):

    print("Graph RAG")
    print("Type 'exit' to quit.\n")

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

        print("\nSeed entities:")

        for item in result["seed_entities"]:
            print(
                f"- {item['entity']} "
                f"({item['score']:.3f})"
            )

        print("\nRetrieved triples:")

        for triple in result[
            "retrieved_triples"
        ]:
            print(
                f"- [{triple['subject']}] "
                f"-> [{triple['predicate']}] "
                f"-> [{triple['object']}]"
            )

        print("\nAnswer:")
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