from __future__ import annotations
from dependencies import *


TARGET_LABELS: Final[list[str]] = [
    "chunk",
    "document",
]

DEFAULT_BATCH_SIZE: Final[int] = 1000


class CleanerState(TypedDict):
    """
    LangGraph 清理流程状态。

    batch_size:
        每批删除的最大节点数。

    dry_run:
        是否只统计、不执行删除。

    target_count:
        删除前匹配的节点数量。

    total_deleted:
        实际删除的节点数量。

    remaining_count:
        删除完成后仍然匹配的节点数量。

    status:
        当前任务执行结果。
    """

    batch_size: int
    dry_run: bool
    target_count: int
    total_deleted: int
    remaining_count: int
    status: str


class Neo4jClient:
    """负责管理 Neo4j 驱动连接。"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Driver | None = None

    def __enter__(self) -> "Neo4jClient":
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        self.close()

    def open(self) -> None:
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

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def get_driver(self) -> Driver:
        if self.driver is None:
            raise RuntimeError(
                "Neo4j driver is not initialized."
            )

        return self.driver


class Neo4jNodeCleaner:
    """负责统计和删除指定标签的 Neo4j 节点。"""

    def __init__(
        self,
        driver: Driver,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        self.driver = driver
        self.batch_size = batch_size

    @staticmethod
    def _count_target_nodes_transaction(
        transaction: ManagedTransaction,
    ) -> int:
        query = """
            MATCH (node)

            WHERE any(
                label IN labels(node)
                WHERE toLower(label) IN $target_labels
            )

            RETURN count(node) AS node_count
        """

        record = transaction.run(
            query,
            target_labels=TARGET_LABELS,
        ).single()

        if record is None:
            return 0

        return int(record["node_count"])

    @staticmethod
    def _delete_one_batch_transaction(
        transaction: ManagedTransaction,
        batch_size: int,
    ) -> int:
        query = """
            MATCH (node)

            WHERE any(
                label IN labels(node)
                WHERE toLower(label) IN $target_labels
            )

            WITH node
            LIMIT $batch_size

            DETACH DELETE node

            RETURN count(*) AS deleted_count
        """

        record = transaction.run(
            query,
            target_labels=TARGET_LABELS,
            batch_size=batch_size,
        ).single()

        if record is None:
            return 0

        return int(record["deleted_count"])

    def count_target_nodes(self) -> int:
        """统计带有 Chunk 或 Document 标签的节点数量。"""
        with self.driver.session() as session:
            return session.execute_read(
                self._count_target_nodes_transaction
            )

    def delete_one_batch(self) -> int:
        """在一个写事务中删除一批目标节点。"""
        with self.driver.session() as session:
            return session.execute_write(
                self._delete_one_batch_transaction,
                self.batch_size,
            )

    def delete_all(self) -> int:
        """循环分批删除所有目标节点。"""
        total_deleted = 0
        batch_number = 0

        while True:
            deleted_count = self.delete_one_batch()

            if deleted_count == 0:
                break

            batch_number += 1
            total_deleted += deleted_count

            print(
                f"Batch {batch_number}: "
                f"deleted {deleted_count} nodes; "
                f"total deleted={total_deleted}"
            )

        return total_deleted


def create_cleaner(
    batch_size: int,
) -> tuple[Neo4jClient, Neo4jNodeCleaner]:
    """
    创建 Neo4j 客户端和节点清理器。

    返回的 Neo4jClient 需要由调用方关闭。
    """
    client = Neo4jClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    client.open()

    cleaner = Neo4jNodeCleaner(
        driver=client.get_driver(),
        batch_size=batch_size,
    )

    return client, cleaner


# ---------- LangGraph 节点 ----------


def initialize_cleaner_node(state: CleanerState) -> CleanerState:
    """Apply shared defaults for CLI, main workflow, and direct graph calls."""
    initialized: CleanerState = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "dry_run": False,
        "target_count": 0,
        "total_deleted": 0,
        "remaining_count": 0,
        "status": "initialized",
        **state,
    }
    batch_size = initialized["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return initialized


def count_target_nodes_node(
    state: CleanerState,
) -> dict:
    """统计需要删除的目标节点。"""
    client, cleaner = create_cleaner(
        state["batch_size"]
    )

    try:
        target_count = cleaner.count_target_nodes()
    finally:
        client.close()

    print(
        f"Found {target_count} nodes carrying "
        "Chunk or Document labels."
    )

    return {
        "target_count": target_count,
        "remaining_count": target_count,
        "status": "counted",
    }


def route_after_count(
    state: CleanerState,
) -> Literal[
    "dry_run",
    "nothing_to_delete",
    "delete",
]:
    """根据统计结果和 dry-run 参数选择后续流程。"""
    if state["dry_run"]:
        return "dry_run"

    if state["target_count"] == 0:
        return "nothing_to_delete"

    return "delete"


def dry_run_node(
    state: CleanerState,
) -> dict:
    """完成 dry-run，不修改数据库。"""
    print("Dry-run completed. No nodes were deleted.")

    return {
        "total_deleted": 0,
        "remaining_count": state["target_count"],
        "status": "dry_run_completed",
    }


def nothing_to_delete_node(
    state: CleanerState,
) -> dict:
    """处理没有目标节点的情况。"""
    print("No matching nodes need to be deleted.")

    return {
        "total_deleted": 0,
        "remaining_count": 0,
        "status": "nothing_to_delete",
    }


def delete_target_nodes_node(
    state: CleanerState,
) -> dict:
    """分批删除所有目标节点。"""
    client, cleaner = create_cleaner(
        state["batch_size"]
    )

    try:
        total_deleted = cleaner.delete_all()
    finally:
        client.close()

    print(
        f"Deletion phase completed: "
        f"{total_deleted} nodes deleted."
    )

    return {
        "total_deleted": total_deleted,
        "status": "deleted",
    }


def verify_remaining_nodes_node(
    state: CleanerState,
) -> dict:
    """删除完成后重新统计目标节点。"""
    client, cleaner = create_cleaner(
        state["batch_size"]
    )

    try:
        remaining_count = (
            cleaner.count_target_nodes()
        )
    finally:
        client.close()

    if remaining_count == 0:
        print(
            "✅ Verification completed. "
            "No matching nodes remain."
        )
        status = "completed"
    else:
        print(
            "⚠️ Verification completed, but "
            f"{remaining_count} matching nodes remain."
        )
        status = "completed_with_remaining_nodes"

    return {
        "remaining_count": remaining_count,
        "status": status,
    }


# ---------- LangGraph 构建 ----------


graph_builder = StateGraph(CleanerState)

graph_builder.add_node("initialize", initialize_cleaner_node)
graph_builder.add_edge("initialize", "count_target_nodes")

graph_builder.add_node(
    "count_target_nodes",
    count_target_nodes_node,
)

graph_builder.add_node(
    "dry_run",
    dry_run_node,
)

graph_builder.add_node(
    "nothing_to_delete",
    nothing_to_delete_node,
)

graph_builder.add_node(
    "delete_target_nodes",
    delete_target_nodes_node,
)

graph_builder.add_node(
    "verify_remaining_nodes",
    verify_remaining_nodes_node,
)

graph_builder.set_entry_point(
    "initialize"
)

graph_builder.add_conditional_edges(
    "count_target_nodes",
    route_after_count,
    {
        "dry_run": "dry_run",
        "nothing_to_delete": "nothing_to_delete",
        "delete": "delete_target_nodes",
    },
)

graph_builder.add_edge(
    "dry_run",
    END,
)

graph_builder.add_edge(
    "nothing_to_delete",
    END,
)

graph_builder.add_edge(
    "delete_target_nodes",
    "verify_remaining_nodes",
)

graph_builder.add_edge(
    "verify_remaining_nodes",
    END,
)

NodeCleanerGraph = graph_builder.compile()


# ---------- 命令行入口 ----------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete all Neo4j nodes carrying "
            "a Chunk or Document label."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of nodes deleted per batch; "
            f"default: {DEFAULT_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only count matching nodes "
            "without deleting them."
        ),
    )

    arguments = parser.parse_args()

    if arguments.batch_size <= 0:
        parser.error(
            "--batch-size must be greater than 0"
        )

    return arguments


def main() -> None:
    arguments = parse_arguments()

    initial_state = {
        "batch_size": arguments.batch_size,
        "dry_run": arguments.dry_run,
    }

    final_state = NodeCleanerGraph.invoke(
        initial_state
    )

    print("\nNeo4j node cleanup completed:")
    print(
        f"  Initial matching nodes: "
        f"{final_state['target_count']}"
    )
    print(
        f"  Deleted nodes: "
        f"{final_state['total_deleted']}"
    )
    print(
        f"  Remaining matching nodes: "
        f"{final_state['remaining_count']}"
    )
    print(
        f"  Status: {final_state['status']}"
    )


if __name__ == "__main__":
    main()
