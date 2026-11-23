from __future__ import annotations
from dependencies import *



# 每轮最多物理合并的节点数量。
MERGE_BATCH_SIZE = 100


class Neo4jClient:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self.close()

    def open(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self.driver.verify_connectivity()
            print(f"✅ Connected to {self.uri}")
        except Exception as exc:
            self.close()
            raise ConnectionError(
                f"Neo4j connection failed: {exc}"
            ) from exc

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def get_driver(self):
        if self.driver is None:
            raise RuntimeError(
                "Driver not initialized. Call open() first."
            )

        return self.driver


@dataclass
class PhysicalMergeState:
    counter: int = 0

    # 合并前 SAME_AS 关系数量。
    same_as_before: int = 0

    # 实际被物理合并的源节点数量。
    nodes_merged: int = 0

    # 因一个源节点指向多个目标而跳过的节点数量。
    ambiguous_sources_skipped: int = 0

    # 删除的自指 SAME_AS 关系数量。
    self_same_as_removed: int = 0

    # 删除的重复关系数量。
    duplicate_relationships_removed: int = 0

    # 处理完成后仍然存在的 SAME_AS 关系数量。
    unresolved_same_as: int = 0


def escape_identifier(identifier: str) -> str:
    """安全转义 Cypher 标签名、关系类型和属性名。"""
    return str(identifier).replace("`", "``")


def run_node(
    state: PhysicalMergeState,
) -> PhysicalMergeState:
    """
    物理合并通过 SAME_AS 连接的节点。

    关系方向约定：

        (source)-[:SAME_AS]->(target)

    source:
        待删除的非结构化实体节点。

    target:
        最终保留的结构化业务节点。

    合并规则：

    1. 保留 target 节点。
    2. 删除 source 节点。
    3. source 的标签添加到 target。
    4. target 已经存在的属性优先。
    5. source 中不冲突的属性添加到 target。
    6. source 的其他关系迁移到 target。
    7. 不保留合并后产生的自指关系。
    8. source 指向多个 target 时跳过。
    9. SAME_AS 链条从末端开始逐层合并。
    """

    escaped_same_as = escape_identifier(
        SAME_AS_RELATIONSHIP
    )

    with Neo4jClient(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
    ) as client:
        driver = client.get_driver()

        def execute_query(
            query: str,
            parameters: Optional[
                Dict[str, Any]
            ] = None,
        ) -> List[Dict[str, Any]]:
            with driver.session() as session:
                result = session.run(
                    query,
                    parameters or {},
                )

                rows = [
                    record.data()
                    for record in result
                ]

                result.consume()
                return rows

        def verify_apoc() -> str:
            """
            检查 APOC 是否可用。

            节点物理合并依赖：
                apoc.refactor.mergeNodes
            """
            try:
                result = execute_query(
                    """
                    RETURN apoc.version() AS version
                    """
                )
            except Exception as exc:
                raise RuntimeError(
                    "APOC is unavailable. "
                    "Install and enable the Neo4j APOC plugin "
                    "before running this script."
                ) from exc

            if not result:
                raise RuntimeError(
                    "Unable to determine the APOC version."
                )

            return str(result[0]["version"])

        def count_same_as_relationships() -> int:
            """统计当前 SAME_AS 关系数量。"""
            query = f"""
                MATCH ()-[relationship:`{
                    escaped_same_as
                }`]->()

                RETURN count(relationship)
                    AS relationship_count
            """

            result = execute_query(query)

            if not result:
                return 0

            return int(
                result[0]["relationship_count"]
            )

        def count_ambiguous_sources() -> int:
            """
            统计存在多个目标的源节点。

            例如：

                (A)-[:SAME_AS]->(B)
                (A)-[:SAME_AS]->(C)

            无法确定应该保留 B 还是 C，因此跳过 A。
            """
            query = f"""
                MATCH (
                    source
                )-[:`{escaped_same_as}`]->(
                    target
                )

                WHERE source <> target

                WITH
                    source,
                    collect(DISTINCT target) AS targets

                WHERE size(targets) > 1

                RETURN count(source) AS source_count
            """

            result = execute_query(query)

            if not result:
                return 0

            return int(result[0]["source_count"])

        def merge_one_batch(
            batch_size: int,
        ) -> int:
            """
            物理合并一批没有歧义的节点。

            对于链式 SAME_AS：

                A -> B -> C

            首先合并：

                B -> C

            B 被合并到 C 后，A 原来指向 B 的关系会迁移为：

                A -> C

            下一轮再将 A 合并到 C。
            """
            query = f"""
                MATCH (
                    source
                )-[:`{escaped_same_as}`]->(
                    candidate
                )

                WHERE source <> candidate

                WITH
                    source,
                    collect(DISTINCT candidate) AS targets

                // 一个源节点只能有一个唯一目标。
                WHERE size(targets) = 1

                WITH
                    source,
                    head(targets) AS target

                // 只处理 SAME_AS 链条末端的目标节点。
                WHERE NOT EXISTS {{
                    MATCH (
                        target
                    )-[:`{escaped_same_as}`]->(
                        next
                    )

                    WHERE next <> target
                }}

                WITH source, target
                LIMIT $batch_size

                /*
                 * target 放在数组第一位，因此 target 是最终
                 * 保留节点，source 会被物理删除。
                 */
                CALL apoc.refactor.mergeNodes(
                    [target, source],
                    {{
                        properties: "discard",
                        mergeRels: true,
                        produceSelfRel: false,
                        preserveExistingSelfRels: false
                    }}
                )
                YIELD node

                RETURN count(node) AS merged_count
            """

            try:
                result = execute_query(
                    query,
                    {
                        "batch_size": batch_size,
                    },
                )
            except Exception as exc:
                raise RuntimeError(
                    "Node merge failed. Check that "
                    "apoc.refactor.mergeNodes is enabled and "
                    "that Neo4j constraints allow the merge."
                ) from exc

            if not result:
                return 0

            return int(result[0]["merged_count"])

        def remove_self_same_as_relationships() -> int:
            """
            删除自指 SAME_AS 关系。

            例如：

                (node)-[:SAME_AS]->(node)
            """
            query = f"""
                MATCH (
                    node
                )-[relationship:`{
                    escaped_same_as
                }`]->(
                    node
                )

                WITH collect(
                    relationship
                ) AS relationships

                FOREACH (
                    relationship IN relationships |
                    DELETE relationship
                )

                RETURN size(
                    relationships
                ) AS removed_count
            """

            result = execute_query(query)

            if not result:
                return 0

            return int(result[0]["removed_count"])

        def remove_duplicate_relationships() -> int:
            """
            删除节点物理合并后产生的重复关系。

            只有以下内容全部相同才认为是重复关系：

            1. 起点相同。
            2. 终点相同。
            3. 关系类型相同。

            不会删除方向相反或类型不同的关系。
            """
            query = """
                MATCH (start)-[relationship]->(end)

                WITH
                    start,
                    end,
                    type(relationship)
                        AS relationship_type,
                    collect(relationship)
                        AS relationships

                WHERE size(relationships) > 1

                WITH
                    head(relationships) AS retained,
                    tail(relationships) AS duplicates

                /*
                 * 将重复关系的属性添加到保留关系。
                 *
                 * 如果存在同名属性，后处理的重复关系属性
                 * 可能覆盖保留关系中的同名属性。
                 */
                FOREACH (
                    duplicate IN duplicates |
                    SET retained += properties(duplicate)
                )

                WITH
                    duplicates,
                    size(duplicates) AS removed_count

                FOREACH (
                    duplicate IN duplicates |
                    DELETE duplicate
                )

                RETURN coalesce(
                    sum(removed_count),
                    0
                ) AS removed_count
            """

            result = execute_query(query)

            if not result:
                return 0

            return int(result[0]["removed_count"])

        # ---------- 开始执行 ----------

        apoc_version = verify_apoc()
        print(f"APOC version: {apoc_version}")

        same_as_before = (
            count_same_as_relationships()
        )

        ambiguous_sources = (
            count_ambiguous_sources()
        )

        print(
            "SAME_AS relationships before merge: "
            f"{same_as_before}"
        )

        print(
            "Ambiguous source nodes skipped: "
            f"{ambiguous_sources}"
        )

        total_merged = 0
        batch_number = 0

        while True:
            merged_count = merge_one_batch(
                MERGE_BATCH_SIZE
            )

            if merged_count == 0:
                break

            batch_number += 1
            total_merged += merged_count

            print(
                f"Batch {batch_number}: "
                f"merged {merged_count} nodes; "
                f"total merged={total_merged}"
            )

        print(
            "Physical node merge completed: "
            f"{total_merged} nodes merged"
        )

        print(
            "\nRemoving self-referencing "
            "SAME_AS relationships..."
        )

        self_same_as_removed = (
            remove_self_same_as_relationships()
        )

        print(
            "Self-referencing SAME_AS relationships "
            f"removed: {self_same_as_removed}"
        )

        print(
            "\nRemoving duplicate relationships..."
        )

        duplicate_relationships_removed = (
            remove_duplicate_relationships()
        )

        print(
            "Duplicate relationships removed: "
            f"{duplicate_relationships_removed}"
        )

        unresolved_same_as = (
            count_same_as_relationships()
        )

        if unresolved_same_as > 0:
            print(
                "\n⚠️ Unresolved SAME_AS relationships: "
                f"{unresolved_same_as}"
            )
            print(
                "These relationships may represent "
                "ambiguous mappings or SAME_AS cycles."
            )
        else:
            print(
                "\n✅ All SAME_AS relationships "
                "were resolved."
            )

        state.counter += 1
        state.same_as_before = same_as_before
        state.nodes_merged = total_merged
        state.ambiguous_sources_skipped = (
            ambiguous_sources
        )
        state.self_same_as_removed = (
            self_same_as_removed
        )
        state.duplicate_relationships_removed = (
            duplicate_relationships_removed
        )
        state.unresolved_same_as = (
            unresolved_same_as
        )

    return state


# ---------- LangGraph 构建 ----------

graph_builder = StateGraph(
    PhysicalMergeState
)

graph_builder.add_node(
    "run",
    run_node,
)

graph_builder.set_entry_point(
    "run"
)

graph_builder.add_edge(
    "run",
    END,
)

PhysicalMergeGraph = (
    graph_builder.compile()
)


if __name__ == "__main__":
    initial_state = PhysicalMergeState()

    final_state = PhysicalMergeGraph.invoke(
        initial_state
    )

    print(
        "\nPhysicalMergeGraph completed: "
        f"{final_state}"
    )