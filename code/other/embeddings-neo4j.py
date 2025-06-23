"""
Before running the code to populate embeddings in neo4j the below commands should be run in the neo4j instance with the DiaKG:

CREATE VECTOR INDEX entities
FOR (n:Entity) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
}

CREATE FULLTEXT INDEX entityEnFulltextIndex
FOR (n:Entity) ON EACH [n.entity_en]

"""

import os, json, re, time, sys
from pathlib import Path

from neo4j import GraphDatabase
from langchain_mistralai.embeddings import MistralAIEmbeddings
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType

# ── Configuration ────────────────────────────────────────────────────
NEO4J_URI=        'neo4j+s://a98dafc9.databases.neo4j.io'
NEO4J_USERNAME=   'neo4j'
NEO4J_PASSWORD=   'mdH7ytqZjlfj9k2j4UxqNmJR2qkm_J1mnBP98ekN2kI'

MISTRAL_API_KEY = 'BRgpTVSWtU2BaQPOT618j6IN0dVJeHbI'

SKIP_JSON = Path("exclusion-entities.json")
BATCH     = 256
DELAY_S   = 0.5

# ── 1.  Build the explicit skip‑set ───────────────────────────────────
try:
    SKIP = set(json.load(open(SKIP_JSON)))
    print(f"Loaded {len(SKIP):,} explicit dosage literals from {SKIP_JSON}")
except FileNotFoundError:
    sys.exit(f"❌  Skip‑list file not found: {SKIP_JSON}")

# Regex that recognises *any* “number‑unit‑per‑kg‑per‑time” pattern
NUMERIC_RX = re.compile(
    r"""
    ^\s*                     # leading spaces
    \d+(\.\d+)?              # first number
    (\s*[-~–]\s*\d+(\.\d+)?)? # optional range "‑" or "~"
    \s*
    (units?|u|U|mg|g|µ?mol)  # unit
    ([/·]\s*(kg|body|d|day|h|hr|hour))+
    """,
    re.IGNORECASE | re.VERBOSE,
)

def should_skip(text: str) -> bool:
    """
    True  → do NOT embed this string
    False → embed
    """
    return text in SKIP or NUMERIC_RX.match(text) is not None

# ── 2.  Helper to pull nodes that need embeddings ─────────────────────
def fetch_candidates(tx):
    """
    Return [(elementId, label), …] for Entity nodes whose
    - entity_en is NOT null
    - embedding  is null
    """
    query = """
    MATCH (e:Entity)
    WHERE e.entity_en IS NOT NULL AND e.embedding IS NULL
    RETURN elementId(e) AS id, e.entity_en AS text
    """
    return tx.run(query).data()

# ── 3.  Main pipeline ─────────────────────────────────────────────────
def populate_embeddings():
    driver   = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    embedder = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)

    with driver.session() as session:
        candidates = session.execute_read(fetch_candidates)

    print(f"Found {len(candidates):,} candidate nodes without embeddings.")

    # Filter out the ones we want to skip
    todo = [rec for rec in candidates if not should_skip(rec["text"])]
    skipped = len(candidates) - len(todo)
    print(f"→ {skipped:,} nodes matched dosage/number patterns and will be skipped.")
    print(f"→ {len(todo):,} nodes will be embedded.\n")

    # ── batching loop ─────────────────────────────────────────────────
    successes = failures = rate_limit_hits = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        texts = [rec["text"] for rec in batch]
        ids   = [rec["id"]   for rec in batch]

        # progress log
        pct = (i + len(batch)) / len(todo) * 100
        print(f"[{pct:5.1f}%]  embedding {len(batch):3d} nodes …", end="", flush=True)

        try:
            vectors = embedder.embed_documents(texts)
            upsert_vectors(
                driver,
                ids=ids,
                embeddings=vectors,
                embedding_property="embedding",
                entity_type=EntityType.NODE,
            )
            successes += len(batch)
            print(" ✓")
            time.sleep(DELAY_S)

        except Exception as e:
            failures  += len(batch)
            err = str(e)[:120].replace("\n", " ")
            print(f" ✗  ({err})")

            if "429" in err or "rate limit" in err.lower():
                rate_limit_hits += 1
                wait = min(30, 5 * rate_limit_hits)
                print(f"    ↳ rate‑limited, backing off {wait}s …")
                time.sleep(wait)

    # ── summary ───────────────────────────────────────────────────────
    print("\n========  SUMMARY  ========")
    print(f"Embedded : {successes:,}")
    print(f"Skipped  : {skipped:,}")
    print(f"Failed   : {failures:,}")
    print("===========================\n")

    # Optional: create the vector index *after* embeddings are written
    with driver.session() as session:
        session.run("""
        CREATE VECTOR INDEX entities
        IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: "cosine"}
        }""")
    driver.close()

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    populate_embeddings()
