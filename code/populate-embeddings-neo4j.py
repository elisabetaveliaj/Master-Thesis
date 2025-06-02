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


import os
import time
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.mistral import MistralAIEmbeddings
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType

# Configuration
NEO4J_URI = '*************'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = '*************'
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')


def get_entities_needing_embeddings(driver):
    """Get ALL entities that need embeddings (no limit)"""
    with driver.session() as session:
        # Count first
        total_count = session.run("""
            MATCH (e:Entity) 
            WHERE e.entity_en IS NOT NULL 
            RETURN count(e) as total
        """).single()["total"]

        no_embedding_count = session.run("""
            MATCH (e:Entity) 
            WHERE e.entity_en IS NOT NULL AND e.embedding IS NULL
            RETURN count(e) as total
        """).single()["total"]

        print(f"Total entities: {total_count}")
        print(f"Entities needing embeddings: {no_embedding_count}")
        print(f"Entities already embedded: {total_count - no_embedding_count}")

        if no_embedding_count == 0:
            print("All entities already have embeddings!")
            return []

        # Get all entities without embeddings
        entities = session.run("""
            MATCH (e:Entity) 
            WHERE e.entity_en IS NOT NULL AND e.embedding IS NULL
            RETURN elementId(e) as id, e.entity_en as text
            ORDER BY e.entity_en
        """).data()

        return entities


def populate_all_embeddings():
    """Process all entities with resume capability and rate limiting"""

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    embedder = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)

    # Get all entities that need embeddings
    entities = get_entities_needing_embeddings(driver)

    if not entities:
        driver.close()
        return

    total_entities = len(entities)
    print(f"\nProcessing {total_entities} entities...")

    successful = 0
    failed = 0
    rate_limit_delays = 0

    # Process entities one by one
    for i, entity in enumerate(entities):
        try:
            # Progress indicator
            if i % 100 == 0:  # Show progress every 100 entities
                print(f"\n Progress: {i}/{total_entities} ({i / total_entities * 100:.1f}%)")
                print(f"  Successful: {successful} | Failed: {failed}")

            # Show current entity being processed
            entity_text = entity['text'][:50] + ("..." if len(entity['text']) > 50 else "")
            print(f" {i + 1}/{total_entities}: {entity_text}")

            # Generate embedding
            embedding = embedder.embed_query(entity['text'])

            # Store in Neo4j
            upsert_vectors(
                driver,
                ids=[entity['id']],
                embedding_property="embedding",
                embeddings=[embedding],
                entity_type=EntityType.NODE
            )

            successful += 1

            # Rate limiting: wait between requests
            time.sleep(0.5)  # 0.5 seconds between requests

        except Exception as e:
            failed += 1
            error_msg = str(e)
            print(f"Failed: {error_msg[:100]}...")

            # Handle rate limiting
            if "429" in error_msg or "rate limit" in error_msg.lower():
                rate_limit_delays += 1
                delay = min(30, 5 * rate_limit_delays)  # Exponential backoff, max 30s
                print(f"Rate limited! Waiting {delay} seconds... (attempt {rate_limit_delays})")
                time.sleep(delay)

                # Retry the same entity
                try:
                    print(f"🔄 Retrying: {entity_text}")
                    embedding = embedder.embed_query(entity['text'])
                    upsert_vectors(
                        driver,
                        ids=[entity['id']],
                        embedding_property="embedding",
                        embeddings=[embedding],
                        entity_type=EntityType.NODE
                    )
                    successful += 1
                    failed -= 1  # Remove from failed count
                    print(f" Retry successful!")
                except Exception as retry_error:
                    print(f"Retry also failed: {str(retry_error)[:50]}...")

            # If too many failures, pause longer
            if failed > 10 and (failed / (i + 1)) > 0.1:  # More than 10% failure rate
                print(f" High failure rate. Pausing 30 seconds...")
                time.sleep(30)

    # Final results
    print(f"\n FINAL RESULTS:")
    print(f"Successfully embedded: {successful}/{total_entities}")
    print(f"Failed: {failed}/{total_entities}")
    print(f"Rate limit delays: {rate_limit_delays}")

    # Verify final state
    print(f"\n Verifying results...")
    remaining = get_entities_needing_embeddings(driver)
    print(f" Entities still needing embeddings: {len(remaining)}")

    driver.close()


if __name__ == "__main__":
    populate_all_embeddings()
    print("\nComplete! All entities processed.")
