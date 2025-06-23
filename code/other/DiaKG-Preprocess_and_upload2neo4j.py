import json
import glob
import os
import logging
from neo4j import GraphDatabase, exceptions

# --- Configuration ---
NEO4J_URI = "*****************************"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "*************"
JSON_DIRECTORY = "/0521_new_format_translated"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Neo4j Driver Initialization ---
driver = None
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    logging.info(f"Successfully connected to Neo4j at {NEO4J_URI}")
except exceptions.AuthError as e:
    logging.error(f"Neo4j authentication failed: {e}. Please check your username and password.")
    exit()
except exceptions.ServiceUnavailable as e:
    logging.error(
        f"Neo4j service unavailable at {NEO4J_URI}: {e}. Check if the database is running or the URI is correct.")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred while connecting to Neo4j: {e}")
    exit()


def generate_conceptual_entity_id(entity_text, entity_type):
    """Generates a unique ID for a conceptual entity based on its text and type."""
    # Normalize text (lowercase, strip whitespace) for the key
    norm_text = entity_text.strip().lower()
    norm_type = entity_type.strip().lower()  # Use lowercase type for key consistency
    return f"{norm_text}_{norm_type}"


def _upsert_conceptual_entity_tx(tx, entity_data, doc_id):
    """
    MERGE a conceptual entity node.
    Set entity_en as a direct property on the node.
    Store original mention details (including entity_en for provenance)
    as a LIST OF JSON STRINGS in 'mentions' property.
    Returns the conceptual_id.
    """
    conceptual_id = generate_conceptual_entity_id(entity_data['entity'], entity_data['entity_type'])

    # Extract the entity_en value. The check in import_json_file_conceptual
    # ensures 'entity_en' key exists in entity_data.
    entity_en_value = entity_data.get('entity_en')

    # Mention details to store (entity_en is still included here for complete provenance of the mention)
    mention_info_dict = {
        "source_doc_id": doc_id,
        "source_entity_id": entity_data['entity_id'],
        "original_text": entity_data['entity'],
        "start_idx": entity_data['start_idx'],
        "end_idx": entity_data['end_idx'],
        "entity_en": entity_en_value
    }
    # Serialize the dictionary to a JSON string for the 'mentions' property
    mention_info_json_string = json.dumps(mention_info_dict)

    # Modified Cypher query
    query = (
        "MERGE (e:Entity {conceptual_id: $conceptual_id}) "
        "ON CREATE SET "
        "  e.canonical_text = $canonical_text, "
        "  e.entity_type = $entity_type_original_case, "
        "  e.entity_en = $entity_en_param, "  # Set entity_en as a direct property on creation
        "  e.mentions = [$mention_info_str] "
        "ON MATCH SET "
        "  e.mentions = CASE WHEN $mention_info_str IN e.mentions THEN e.mentions ELSE e.mentions + $mention_info_str END, "
        "  e.entity_en = $entity_en_param "  # Also set/update entity_en on match
    )

    tx.run(query,
           conceptual_id=conceptual_id,
           canonical_text=entity_data['entity'].strip(),
           entity_type_original_case=entity_data['entity_type'].strip(),
           entity_en_param=entity_en_value,  # Pass the extracted entity_en value
           mention_info_str=mention_info_json_string
           )
    return conceptual_id


def _create_conceptual_relation_tx(tx, relation_data, doc_id, head_conceptual_id, tail_conceptual_id):
    """
    MERGE a conceptual relationship between two entity nodes.
    If the relationship already exists, its source_mentions list is updated with
    details of the current relation instance.
    """
    rel_type_cleaned = relation_data["relation_type"].strip().upper().replace(" ", "_").replace("-", "_")

    # Create a dictionary with the details of this specific relation instance/mention
    relation_mention_info_dict = {
        "source_doc_id": doc_id,
        "source_relation_id": relation_data['relation_id'], # Original relation_id from the document
        "original_relation_type": relation_data['relation_type']
        # Add any other per-mention details you want to preserve here
    }
    # Serialize the dictionary to a JSON string to store it in the list property
    relation_mention_info_json_string = json.dumps(relation_mention_info_dict)

    # This is the Cypher query that uses the parameters $head_conceptual_id,
    # $tail_conceptual_id, and $relation_mention_info_str.
    # The relationship type is dynamically inserted using an f-string.
    query = f"""
    MATCH (h:Entity {{conceptual_id: $head_conceptual_id}})
    MATCH (t:Entity {{conceptual_id: $tail_conceptual_id}})
    MERGE (h)-[r:{rel_type_cleaned}]->(t)
    ON CREATE SET r.source_mentions = [$relation_mention_info_str]
    ON MATCH SET r.source_mentions = CASE
                                        WHEN $relation_mention_info_str IN r.source_mentions THEN r.source_mentions
                                        ELSE r.source_mentions + $relation_mention_info_str
                                      END
    """

    # CRITICAL: Ensure all parameters used in the Cypher query string (prefixed with $)
    # are provided as keyword arguments here, with matching names (without the $).
    tx.run(
        query,
        head_conceptual_id=head_conceptual_id,          # Corresponds to $head_conceptual_id
        tail_conceptual_id=tail_conceptual_id,          # Corresponds to $tail_conceptual_id
        relation_mention_info_str=relation_mention_info_json_string # Corresponds to $relation_mention_info_str
    )



def import_json_file_conceptual(file_path):
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            document = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        return

    doc_id = document.get("doc_id")
    if not doc_id:
        logging.warning(f"Skipping file {file_path} due to missing 'doc_id'.")
        return

    logging.info(f"Processing document for conceptual graph: {doc_id} from file {file_path}")

    # Map to store (original_doc_id, original_entity_id) -> conceptual_id for the current file
    entity_mention_to_conceptual_map = {}

    # --- Phase 1: Upsert Conceptual Entities and build the map ---
    entities_for_batch_upsert = []
    for para_idx, para in enumerate(document.get("paragraphs", [])):
        for sent_idx, sent in enumerate(para.get("sentences", [])):
            for ent_idx, ent_data in enumerate(sent.get("entities", [])):
                if not all(k in ent_data for k in ["entity_id", "entity", "entity_en", "entity_type", "start_idx", "end_idx"]):
                    logging.warning(
                        f"Skipping malformed entity in {doc_id} (P:{para_idx},S:{sent_idx},E:{ent_idx}): {ent_data}")
                    continue
                entities_for_batch_upsert.append(ent_data)

    if entities_for_batch_upsert:
        try:
            with driver.session(database="neo4j") as session:
                def batch_upsert_entities_tx_fn(tx, entities_list):
                    logging.info(f"  Upserting {len(entities_list)} conceptual entities for doc_id: {doc_id}...")
                    for entity_item in entities_list:
                        conceptual_id = _upsert_conceptual_entity_tx(tx, entity_item, doc_id)
                        # Store mapping for relation creation phase
                        entity_mention_to_conceptual_map[(doc_id, entity_item['entity_id'])] = conceptual_id

                session.execute_write(batch_upsert_entities_tx_fn, entities_for_batch_upsert)
                logging.info(f"  Successfully processed conceptual entities for doc_id: {doc_id}.")
        except Exception as e:
            logging.error(f"Error during conceptual entity upsert for doc_id {doc_id}: {e}")
            return  # Stop processing this file if entities fail

    # --- Phase 2: Create Conceptual Relations using the map ---
    relations_for_batch_create = []
    for para_idx, para in enumerate(document.get("paragraphs", [])):
        for sent_idx, sent in enumerate(para.get("sentences", [])):
            for rel_idx, rel_data in enumerate(sent.get("relations", [])):
                if not all(k in rel_data for k in ["relation_type", "relation_id", "head_entity_id", "tail_entity_id"]):
                    logging.warning(
                        f"Skipping malformed relation in {doc_id} (P:{para_idx},S:{sent_idx},R:{rel_idx}): {rel_data}")
                    continue

                head_original_key = (doc_id, rel_data['head_entity_id'])
                tail_original_key = (doc_id, rel_data['tail_entity_id'])

                if head_original_key not in entity_mention_to_conceptual_map or \
                        tail_original_key not in entity_mention_to_conceptual_map:
                    logging.warning(
                        f"Could not find conceptual IDs for head/tail of relation {rel_data['relation_id']} in {doc_id}. Skipping this relation.")
                    continue

                head_conceptual_id = entity_mention_to_conceptual_map[head_original_key]
                tail_conceptual_id = entity_mention_to_conceptual_map[tail_original_key]

                relations_for_batch_create.append({
                    "data": rel_data,
                    "head_conceptual_id": head_conceptual_id,
                    "tail_conceptual_id": tail_conceptual_id
                })

    if relations_for_batch_create:
        try:
            with driver.session(database="neo4j") as session:
                def batch_create_relations_tx_fn(tx, relations_list_with_conceptual_ids):
                    logging.info(
                        f"  Creating {len(relations_list_with_conceptual_ids)} conceptual relations for doc_id: {doc_id}...")
                    for item in relations_list_with_conceptual_ids:
                        _create_conceptual_relation_tx(tx, item["data"], doc_id, item["head_conceptual_id"],
                                                       item["tail_conceptual_id"])

                session.execute_write(batch_create_relations_tx_fn, relations_for_batch_create)
                logging.info(f"  Successfully processed conceptual relations for doc_id: {doc_id}.")
        except Exception as e:
            logging.error(f"Error during conceptual relation creation for doc_id {doc_id}: {e}")


def import_all_json_files_conceptual(directory_path):
    search_path = os.path.join(directory_path, "*.json")
    file_list = glob.glob(search_path)

    if not file_list:
        logging.warning(f"No JSON files found in directory: {directory_path} (searched for: {search_path})")
        return

    logging.info(f"Found {len(file_list)} JSON files in {directory_path} for conceptual import.")
    for file_path in file_list:
        import_json_file_conceptual(file_path)  # Call the new conceptual import function
    logging.info("Finished conceptual import of all JSON files.")


if __name__ == "__main__":
    logging.info("Please ensure you have a unique constraint on :Entity(conceptual_id) in your Neo4j database.")
    logging.info(
        "Example Cypher: CREATE CONSTRAINT entity_conceptual_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.conceptual_id IS UNIQUE;")

    if "YOUR_AURA_DB_PASSWORD" in NEO4J_PASSWORD or "YOUR_AURA_DB_URI" in NEO4J_URI:
        logging.error("Please update NEO4J_URI and NEO4J_PASSWORD in the script.")
    else:
        import_all_json_files_conceptual(JSON_DIRECTORY)  # Call the conceptual import main function

    if driver:
        driver.close()
        logging.info("Neo4j driver closed.")

