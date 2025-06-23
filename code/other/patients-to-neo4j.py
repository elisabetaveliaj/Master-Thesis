import json, re, logging, sys
from neo4j import GraphDatabase, exceptions

# ─── EDIT THESE 3 LINES ────────────────────────────────────────────
JSONL_FILE   = "patient_facts.jsonl"
NEO4J_URI    = "neo4j+s://a98dafc9.databases.neo4j.io"
NEO4J_USER   = "neo4j"
NEO4J_PASS   = "mdH7ytqZjlfj9k2j4UxqNmJR2qkm_J1mnBP98ekN2kI"
# ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

BATCH_SIZE = 5_000

rel_clean = lambda r: re.sub(r"[^0-9A-Za-z]", "_", r).upper()

def parse_tail(t: str):
    if t.startswith("num::"):
        return True, t, float(t.split("::", 1)[1])
    return False, t, None

def load_jsonl(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        sys.exit(f"❌  File not found: {path}")
    except Exception as e:
        sys.exit(f"❌  Could not read {path}: {e}")

def upload(triples, driver):
    def tx_func(tx, chunk):
        """Write one batch of triples inside a single transaction.

        Preconditions
        -------------
        • Uniqueness constraints exist on (:Entity ent_id) and (:Literal ent_id).
        • `chunk` is a list of dicts with keys 'h', 'r', 't'.
        """

        for tr in chunk:
            hid = tr["h"]  # subject id
            rel = rel_clean(tr["r"])  # cleaned rel type
            tail = tr["t"]

            if tail.startswith("num::"):
                # object is a Literal node
                tid = tail
                num_v = float(tail.split("::", 1)[1])

                tx.run(f"""
                    // subject ---------------------------------------------------
                    MERGE (h:Entity  {{ent_id:$hid}})
                    // object ----------------------------------------------------
                    MERGE (o:Literal {{ent_id:$tid}})
                      ON CREATE SET o.value = $val
                    // relationship ---------------------------------------------
                    MERGE (h)-[:{rel}]->(o)
                """, hid=hid, tid=tid, val=num_v)

            else:
                # object is another Entity node
                tid = tail
                tx.run(f"""
                    MERGE (h:Entity  {{ent_id:$hid}})
                    MERGE (o:Entity  {{ent_id:$tid}})
                    MERGE (h)-[:{rel}]->(o)
                """, hid=hid, tid=tid)

    with driver.session(database="neo4j") as session:
        for i in range(0, len(triples), BATCH_SIZE):
            session.execute_write(tx_func, triples[i:i+BATCH_SIZE])
            logging.info(f"Committed {i + len(triples[i:i+BATCH_SIZE]):>7}/{len(triples)} triples")

if __name__ == "__main__":
    triples = load_jsonl(JSONL_FILE)
    logging.info(f"Loaded {len(triples)} triples from {JSONL_FILE}")

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        driver.verify_connectivity()
    except exceptions.AuthError:
        sys.exit("❌  Neo4j authentication failed.")
    except Exception as e:
        sys.exit(f"❌  Could not connect to Neo4j: {e}")

    upload(triples, driver)
    driver.close()
    logging.info("✅  Upload complete – patient facts are now in Neo4j")
