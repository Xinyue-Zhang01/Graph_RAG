import json
import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from kg_gen import KGGen


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

PDF_PATH = Path("data/attention_is_all_you_need.pdf")
TEXT_PATH = Path("data/attention_is_all_you_need.txt")
OUTPUT_PATH = Path("data/kg_triples.json")

KG_MODEL = os.getenv(
    "KG_MODEL",
    "openai/gpt-4o",
)

CHUNK_SIZE = int(
    os.getenv("KG_CHUNK_SIZE", "5000")
)

CLUSTER = (
    os.getenv("KG_CLUSTER", "true").lower()
    == "true"
)


# --------------------------------------------------
# 1. Extract Text from PDF
# --------------------------------------------------

def extract_text_from_pdf():
    document = pymupdf.open(PDF_PATH)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        text = page.get_text("text")

        pages.append(
            f"\n\n===== PAGE {page_number} =====\n\n"
            f"{text}"
        )

    full_text = "".join(pages)

    TEXT_PATH.write_text(
        full_text,
        encoding="utf-8",
    )

    print(
        f"Extracted {len(document)} pages "
        f"from {PDF_PATH}."
    )

    print(
        f"Saved extracted text to "
        f"{TEXT_PATH}."
    )

    return full_text


# --------------------------------------------------
# 2. Load or Extract Source Text
# --------------------------------------------------

def load_source_text():

    if TEXT_PATH.exists():

        print(
            f"Loading existing text from "
            f"{TEXT_PATH}..."
        )

        return TEXT_PATH.read_text(
            encoding="utf-8"
        )

    print(
        "Extracted text file not found. "
        "Extracting text from PDF..."
    )

    return extract_text_from_pdf()


# --------------------------------------------------
# 3. Build Knowledge Graph with KGGen
# --------------------------------------------------

def build_knowledge_graph():

    text = load_source_text()

    print(
        f"Loaded source text: "
        f"{len(text)} characters."
    )

    kg = KGGen(
        model=KG_MODEL,
        temperature=0.0,
        api_key=os.getenv(
            "OPENAI_API_KEY"
        ),
    )

    print(
        "Generating knowledge graph..."
    )

    graph = kg.generate(
        input_data=text,
        chunk_size=CHUNK_SIZE,
        cluster=CLUSTER,
    )

    print(
        f"Entities: "
        f"{len(graph.entities)}"
    )

    print(
        f"Relations: "
        f"{len(graph.relations)}"
    )

    triples = []

    for (
        subject,
        predicate,
        obj,
    ) in graph.relations:

        triples.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            }
        )

    triples.sort(
        key=lambda x: (
            x["subject"],
            x["predicate"],
            x["object"],
        )
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            triples,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved "
        f"{len(triples)} triples "
        f"to {OUTPUT_PATH}."
    )


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

if __name__ == "__main__":
    build_knowledge_graph()