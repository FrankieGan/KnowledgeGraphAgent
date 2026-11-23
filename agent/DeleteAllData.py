"""Delete all nodes and relationships in the configured Neo4j database."""

from neo4j import GraphDatabase
from dependencies import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def main() -> None:
    with GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    ) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            summary = session.run("MATCH (n) DETACH DELETE n").consume()
        print(
            f"Deleted {summary.counters.nodes_deleted} nodes and "
            f"{summary.counters.relationships_deleted} relationships."
        )


if __name__ == "__main__":
    main()
