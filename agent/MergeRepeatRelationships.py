from __future__ import annotations
from dependencies import *


# 关系语义相似度达到该值时允许合并。
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.7

# neo4j_graphrag 的 OpenAIEmbeddings 通过 embed_query()
# 逐条生成嵌入。该参数只控制进度分组。
DEFAULT_EMBEDDING_BATCH_SIZE: Final[int] = 64

# 每次写入 Neo4j 的关系合并计划数量。
DEFAULT_WRITE_BATCH_SIZE: Final[int] = 200


EmbeddingVector = List[float]
SimilarityMatrix = List[List[float]]


class RelationshipInfo(TypedDict):
    element_id: str
    relationship_type: str
    start_id: str
    end_id: str
    direction: str
    properties: Dict[str, Any]


class RelationshipGroup(TypedDict):
    left_id: str
    right_id: str
    relationships: List[RelationshipInfo]


class MergePlan(TypedDict):
    retained_id: str
    duplicate_ids: List[str]

    properties: Dict[str, Any]

    original_types: List[str]
    original_directions: List[str]
    original_relationship_ids: List[str]

    minimum_similarity: float
    maximum_similarity: float


class RelationshipMergeState(TypedDict):
    similarity_threshold: float
    embedding_batch_size: int
    write_batch_size: int
    dry_run: bool

    relationship_groups: List[RelationshipGroup]
    merge_plans: List[MergePlan]

    candidate_pair_count: int
    candidate_relationship_count: int

    merge_cluster_count: int
    relationships_to_remove: int

    merged_cluster_count: int
    relationships_removed: int

    remaining_candidate_pairs: int
    status: str


@dataclass
class MergeStatistics:
    merged_cluster_count: int = 0
    relationships_removed: int = 0


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
                auth=(
                    self.user,
                    self.password,
                ),
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


def chunks(
    items: List[MergePlan],
    size: int,
) -> Iterator[List[MergePlan]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def normalize_relationship_type(
    relationship_type: str,
) -> str:
    """
    将关系类型转换为适合嵌入模型处理的文本。

    示例：

        WORKS_FOR       -> works for
        employedBy      -> employed by
        located-in      -> located in
        供应商_属于_行业 -> 供应商 属于 行业
    """
    normalized = str(
        relationship_type
    ).strip()

    # 拆分驼峰命名。
    normalized = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        " ",
        normalized,
    )

    # 下划线、连字符和连续空格统一为空格。
    normalized = re.sub(
        r"[_\-\s]+",
        " ",
        normalized,
    )

    return normalized.strip().casefold()


def normalize_embedding(
    vector: List[float],
) -> EmbeddingVector:
    """将嵌入向量转换为单位向量。"""
    magnitude = sum(
        float(value) * float(value)
        for value in vector
    ) ** 0.5

    if magnitude == 0:
        raise ValueError(
            "Embedding service returned a zero vector."
        )

    return [
        float(value) / magnitude
        for value in vector
    ]


def calculate_cosine_similarity(
    first_vector: EmbeddingVector,
    second_vector: EmbeddingVector,
) -> float:
    """计算两个归一化向量的余弦相似度。"""
    if len(first_vector) != len(second_vector):
        raise ValueError(
            "Embedding dimensions do not match: "
            f"{len(first_vector)} != "
            f"{len(second_vector)}"
        )

    score = sum(
        first_value * second_value
        for first_value, second_value
        in zip(
            first_vector,
            second_vector,
        )
    )

    # 防止浮点误差让结果略微超出范围。
    return max(
        -1.0,
        min(1.0, float(score)),
    )


def create_embedding_cache(
    texts: List[str],
    batch_size: int,
) -> Dict[str, EmbeddingVector]:
    """
    使用 neo4j_graphrag 的 OpenAIEmbeddings 生成嵌入。

    neo4j_graphrag.embeddings.OpenAIEmbeddings 使用：

        EMBEDDING_LLM.embed_query(text)

    不支持 LangChain 的 embed_documents()，因此逐条处理。

    batch_size 只用于进度分组，不表示一次 API 请求
    会发送多个文本。
    """
    if batch_size <= 0:
        raise ValueError(
            "embedding batch size must be greater than 0"
        )

    embedding_cache: Dict[
        str,
        EmbeddingVector
    ] = {}

    if not texts:
        return embedding_cache

    embed_query = getattr(
        EMBEDDING_LLM,
        "embed_query",
        None,
    )

    if not callable(embed_query):
        raise TypeError(
            "EMBEDDING_LLM does not provide "
            "embed_query(). "
            f"Actual type: {type(EMBEDDING_LLM)!r}; "
            f"module: "
            f"{type(EMBEDDING_LLM).__module__!r}"
        )

    total_texts = len(texts)

    total_batches = (
        total_texts + batch_size - 1
    ) // batch_size

    print(
        "Embedding implementation: "
        f"{type(EMBEDDING_LLM).__module__}."
        f"{type(EMBEDDING_LLM).__name__}"
    )

    for batch_number, start_index in enumerate(
        range(0, total_texts, batch_size),
        start=1,
    ):
        text_batch = texts[
            start_index:start_index + batch_size
        ]

        print(
            "Embedding relationship types: "
            f"batch {batch_number}/{total_batches}, "
            f"items={len(text_batch)}"
        )

        for item_number, text in enumerate(
            text_batch,
            start=1,
        ):
            global_index = (
                start_index + item_number
            )

            print(
                f"  Embedding {global_index}/"
                f"{total_texts}: {text}"
            )

            try:
                vector = embed_query(text)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to generate embedding "
                    f"for relationship type: {text!r}. "
                    f"Embedder type: "
                    f"{type(EMBEDDING_LLM)!r}"
                ) from exc

            if vector is None:
                raise RuntimeError(
                    "Embedding service returned None "
                    f"for relationship type: {text!r}"
                )

            vector_list = [
                float(value)
                for value in vector
            ]

            if not vector_list:
                raise RuntimeError(
                    "Embedding service returned an "
                    "empty vector for relationship type: "
                    f"{text!r}"
                )

            embedding_cache[text] = (
                normalize_embedding(
                    vector_list
                )
            )

    return embedding_cache


def build_similarity_matrix(
    relationships: List[RelationshipInfo],
    embedding_cache: Dict[
        str,
        EmbeddingVector
    ],
) -> SimilarityMatrix:
    """
    构建一个节点对中所有关系的语义相似度矩阵。
    """
    relationship_count = len(relationships)

    matrix: SimilarityMatrix = [
        [0.0] * relationship_count
        for _ in range(relationship_count)
    ]

    for index in range(relationship_count):
        matrix[index][index] = 1.0

    for first_index in range(
        relationship_count
    ):
        first_type = normalize_relationship_type(
            relationships[first_index][
                "relationship_type"
            ]
        )

        first_vector = embedding_cache[
            first_type
        ]

        for second_index in range(
            first_index + 1,
            relationship_count,
        ):
            second_type = (
                normalize_relationship_type(
                    relationships[second_index][
                        "relationship_type"
                    ]
                )
            )

            # 完全相同的类型不需要再次计算相似度。
            if first_type == second_type:
                similarity = 1.0
            else:
                second_vector = embedding_cache[
                    second_type
                ]

                similarity = (
                    calculate_cosine_similarity(
                        first_vector,
                        second_vector,
                    )
                )

            matrix[first_index][
                second_index
            ] = similarity

            matrix[second_index][
                first_index
            ] = similarity

    return matrix


def find_semantic_clusters(
    similarity_matrix: SimilarityMatrix,
    threshold: float,
) -> List[List[int]]:
    """
    使用完全链接聚类寻找可合并的关系簇。

    只有两个簇之间的所有关系相似度都达到阈值，
    才允许合并，避免链式过度合并。

    例如：

        A 与 B 相似
        B 与 C 相似
        A 与 C 不相似

    此时不会把 A、B、C 全部合并。
    """
    clusters: List[List[int]] = [
        [index]
        for index in range(
            len(similarity_matrix)
        )
    ]

    while True:
        best_pair: tuple[int, int] | None = None
        best_minimum_similarity = -1.0

        for first_cluster_index in range(
            len(clusters)
        ):
            for second_cluster_index in range(
                first_cluster_index + 1,
                len(clusters),
            ):
                first_cluster = clusters[
                    first_cluster_index
                ]

                second_cluster = clusters[
                    second_cluster_index
                ]

                cross_similarities = [
                    similarity_matrix[
                        first_member
                    ][
                        second_member
                    ]
                    for first_member in first_cluster
                    for second_member in second_cluster
                ]

                minimum_similarity = min(
                    cross_similarities
                )

                if (
                    minimum_similarity >= threshold
                    and minimum_similarity
                    > best_minimum_similarity
                ):
                    best_pair = (
                        first_cluster_index,
                        second_cluster_index,
                    )

                    best_minimum_similarity = (
                        minimum_similarity
                    )

        if best_pair is None:
            break

        (
            first_cluster_index,
            second_cluster_index,
        ) = best_pair

        clusters[first_cluster_index] = (
            clusters[first_cluster_index]
            + clusters[second_cluster_index]
        )

        del clusters[second_cluster_index]

    return clusters


def select_representative_relationship(
    cluster: List[int],
    relationships: List[RelationshipInfo],
    similarity_matrix: SimilarityMatrix,
) -> int:
    """
    选择语义簇中需要保留的代表关系。

    优先级：

    1. 与簇内其他关系平均相似度最高。
    2. 关系属性数量更多。
    3. 关系类型和 elementId 用于稳定排序。
    """
    if len(cluster) == 1:
        return cluster[0]

    def representative_score(
        relationship_index: int,
    ) -> tuple[float, int, str, str]:
        other_indexes = [
            other_index
            for other_index in cluster
            if other_index != relationship_index
        ]

        average_similarity = sum(
            similarity_matrix[
                relationship_index
            ][
                other_index
            ]
            for other_index in other_indexes
        ) / len(other_indexes)

        relationship = relationships[
            relationship_index
        ]

        return (
            average_similarity,
            len(relationship["properties"]),
            relationship["relationship_type"],
            relationship["element_id"],
        )

    return max(
        cluster,
        key=representative_score,
    )


def merge_relationship_properties(
    representative: RelationshipInfo,
    cluster_relationships: List[
        RelationshipInfo
    ],
) -> Dict[str, Any]:
    """
    合并关系属性。

    代表关系已有属性优先。
    其他关系中的非冲突属性会添加到代表关系。
    """
    merged_properties = dict(
        representative["properties"]
    )

    for relationship in cluster_relationships:
        if (
            relationship["element_id"]
            == representative["element_id"]
        ):
            continue

        for property_key, property_value in (
            relationship["properties"].items()
        ):
            merged_properties.setdefault(
                property_key,
                property_value,
            )

    return merged_properties


def create_merge_plan(
    cluster: List[int],
    relationships: List[RelationshipInfo],
    similarity_matrix: SimilarityMatrix,
) -> MergePlan:
    """为一个高相似度关系簇创建合并计划。"""
    representative_index = (
        select_representative_relationship(
            cluster,
            relationships,
            similarity_matrix,
        )
    )

    representative = relationships[
        representative_index
    ]

    cluster_relationships = [
        relationships[index]
        for index in cluster
    ]

    duplicate_relationships = [
        relationship
        for relationship in cluster_relationships
        if relationship["element_id"]
        != representative["element_id"]
    ]

    pairwise_similarities = [
        similarity_matrix[
            cluster[first_position]
        ][
            cluster[second_position]
        ]
        for first_position in range(len(cluster))
        for second_position in range(
            first_position + 1,
            len(cluster),
        )
    ]

    return {
        "retained_id": representative["element_id"],
        "duplicate_ids": [
            relationship["element_id"]
            for relationship
            in duplicate_relationships
        ],
        "properties": merge_relationship_properties(
            representative,
            cluster_relationships,
        ),
        "original_types": [
            relationship["relationship_type"]
            for relationship
            in cluster_relationships
        ],
        "original_directions": [
            relationship["direction"]
            for relationship
            in cluster_relationships
        ],
        "original_relationship_ids": [
            relationship["element_id"]
            for relationship
            in cluster_relationships
        ],
        "minimum_similarity": round(
            min(pairwise_similarities),
            6,
        ),
        "maximum_similarity": round(
            max(pairwise_similarities),
            6,
        ),
    }


def load_relationship_groups(
    driver: Driver,
) -> List[RelationshipGroup]:
    """
    查找两个节点之间存在两条及以上关系的节点对。

    方向不影响分组：

        (A)-[:R1]->(B)
        (A)-[:R2]->(B)
        (B)-[:R3]->(A)

    以上三条关系会进入同一个节点对。
    """
    query = """
        MATCH (start)-[relationship]->(end)

        WITH
            start,
            end,
            relationship,
            CASE
                WHEN elementId(start) <= elementId(end)
                THEN elementId(start)
                ELSE elementId(end)
            END AS left_id,
            CASE
                WHEN elementId(start) <= elementId(end)
                THEN elementId(end)
                ELSE elementId(start)
            END AS right_id

        WITH
            left_id,
            right_id,
            collect({
                element_id: elementId(relationship),
                relationship_type: type(relationship),
                start_id: elementId(start),
                end_id: elementId(end),
                properties: properties(relationship)
            }) AS relationships

        WHERE size(relationships) >= 2

        RETURN
            left_id,
            right_id,
            relationships

        ORDER BY left_id, right_id
    """

    groups: List[RelationshipGroup] = []

    with driver.session() as session:
        result = session.run(query)

        for record in result:
            left_id = str(record["left_id"])
            right_id = str(record["right_id"])

            relationships: List[
                RelationshipInfo
            ] = []

            for raw_relationship in record[
                "relationships"
            ]:
                start_id = str(
                    raw_relationship["start_id"]
                )

                end_id = str(
                    raw_relationship["end_id"]
                )

                if start_id == end_id:
                    direction = "self"
                elif (
                    start_id == left_id
                    and end_id == right_id
                ):
                    direction = "left_to_right"
                else:
                    direction = "right_to_left"

                relationships.append({
                    "element_id": str(
                        raw_relationship["element_id"]
                    ),
                    "relationship_type": str(
                        raw_relationship[
                            "relationship_type"
                        ]
                    ),
                    "start_id": start_id,
                    "end_id": end_id,
                    "direction": direction,
                    "properties": dict(
                        raw_relationship[
                            "properties"
                        ]
                        or {}
                    ),
                })

            groups.append({
                "left_id": left_id,
                "right_id": right_id,
                "relationships": relationships,
            })

        result.consume()

    return groups


def apply_merge_plans(
    driver: Driver,
    plans: List[MergePlan],
    batch_size: int,
) -> MergeStatistics:
    """
    执行关系合并计划。

    保留代表关系，并删除同一语义簇中的其他关系。
    低相似度关系不会进入合并计划，因此原样保留。
    """
    query = """
        UNWIND $plans AS plan

        MATCH ()-[retained]->()
        WHERE elementId(retained) = plan.retained_id

        MATCH ()-[duplicate]->()
        WHERE elementId(duplicate)
            IN plan.duplicate_ids

        WITH
            plan,
            retained,
            collect(DISTINCT duplicate)
                AS duplicates

        SET retained += plan.properties

        SET
            retained.semantic_merged = true,
            retained.semantic_merged_at = datetime(),
            retained.semantic_merge_count =
                size(duplicates) + 1,
            retained.semantic_original_types =
                plan.original_types,
            retained.semantic_original_directions =
                plan.original_directions,
            retained.semantic_original_relationship_ids =
                plan.original_relationship_ids,
            retained.semantic_minimum_similarity =
                plan.minimum_similarity,
            retained.semantic_maximum_similarity =
                plan.maximum_similarity

        FOREACH (
            duplicate IN duplicates |
            DELETE duplicate
        )

        RETURN
            count(retained)
                AS merged_cluster_count,
            coalesce(
                sum(size(duplicates)),
                0
            ) AS relationships_removed
    """

    statistics = MergeStatistics()

    for batch_number, batch in enumerate(
        chunks(plans, batch_size),
        start=1,
    ):
        with driver.session() as session:
            record = session.run(
                query,
                plans=batch,
            ).single()

        if record is None:
            continue

        merged_cluster_count = int(
            record["merged_cluster_count"]
        )

        relationships_removed = int(
            record["relationships_removed"]
        )

        statistics.merged_cluster_count += (
            merged_cluster_count
        )

        statistics.relationships_removed += (
            relationships_removed
        )

        print(
            f"Write batch {batch_number}: "
            f"merged clusters="
            f"{merged_cluster_count}, "
            f"removed relationships="
            f"{relationships_removed}"
        )

    return statistics


# ---------- LangGraph 节点 ----------


def initialize_relationship_merge_node(
    state: RelationshipMergeState,
) -> RelationshipMergeState:
    """Apply shared defaults for CLI, main workflow, and direct graph calls."""
    initialized: RelationshipMergeState = {
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "embedding_batch_size": DEFAULT_EMBEDDING_BATCH_SIZE,
        "write_batch_size": DEFAULT_WRITE_BATCH_SIZE,
        "dry_run": False,
        "relationship_groups": [],
        "merge_plans": [],
        "candidate_pair_count": 0,
        "candidate_relationship_count": 0,
        "merge_cluster_count": 0,
        "relationships_to_remove": 0,
        "merged_cluster_count": 0,
        "relationships_removed": 0,
        "remaining_candidate_pairs": 0,
        "status": "initialized",
        **state,
    }
    for key in ("embedding_batch_size", "write_batch_size"):
        value = initialized[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    threshold = initialized["similarity_threshold"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not 0.0 <= threshold <= 1.0):
        raise ValueError("similarity_threshold must be between 0 and 1")
    if not isinstance(initialized["dry_run"], bool):
        raise ValueError("dry_run must be a boolean")
    return initialized


def scan_relationships_node(
    state: RelationshipMergeState,
) -> Dict[str, Any]:
    """读取所有拥有多条关系的无序节点对。"""
    with Neo4jClient(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
    ) as client:
        groups = load_relationship_groups(
            client.get_driver()
        )

    relationship_count = sum(
        len(group["relationships"])
        for group in groups
    )

    print(
        "Candidate node pairs: "
        f"{len(groups)}"
    )

    print(
        "Relationships in candidate pairs: "
        f"{relationship_count}"
    )

    return {
        "relationship_groups": groups,
        "candidate_pair_count": len(groups),
        "candidate_relationship_count": (
            relationship_count
        ),
        "status": "relationships_scanned",
    }


def analyze_relationships_node(
    state: RelationshipMergeState,
) -> Dict[str, Any]:
    """
    调用 EMBEDDING_LLM 分析关系语义，并生成合并计划。
    """
    groups = state["relationship_groups"]

    if not groups:
        print(
            "No node pairs with multiple "
            "relationships were found."
        )

        return {
            "merge_plans": [],
            "merge_cluster_count": 0,
            "relationships_to_remove": 0,
            "status": "no_candidate_pairs",
        }

    # 只对唯一的规范化关系类型生成一次嵌入。
    unique_relation_texts = sorted({
        normalize_relationship_type(
            relationship["relationship_type"]
        )
        for group in groups
        for relationship in group["relationships"]
    })

    print(
        "Unique relationship types to embed: "
        f"{len(unique_relation_texts)}"
    )

    embedding_cache = create_embedding_cache(
        texts=unique_relation_texts,
        batch_size=state[
            "embedding_batch_size"
        ],
    )

    merge_plans: List[MergePlan] = []

    for group_number, group in enumerate(
        groups,
        start=1,
    ):
        relationships = group["relationships"]

        similarity_matrix = (
            build_similarity_matrix(
                relationships,
                embedding_cache,
            )
        )

        semantic_clusters = (
            find_semantic_clusters(
                similarity_matrix,
                state["similarity_threshold"],
            )
        )

        group_merge_count = 0

        for cluster in semantic_clusters:
            # 单独一个关系不需要合并。
            if len(cluster) < 2:
                continue

            merge_plans.append(
                create_merge_plan(
                    cluster,
                    relationships,
                    similarity_matrix,
                )
            )

            group_merge_count += 1

        if group_merge_count > 0:
            print(
                f"Pair {group_number}: "
                f"relationships="
                f"{len(relationships)}, "
                f"merge clusters="
                f"{group_merge_count}"
            )

    relationships_to_remove = sum(
        len(plan["duplicate_ids"])
        for plan in merge_plans
    )

    print(
        "Semantic merge clusters found: "
        f"{len(merge_plans)}"
    )

    print(
        "Relationships scheduled for removal: "
        f"{relationships_to_remove}"
    )

    return {
        "merge_plans": merge_plans,
        "merge_cluster_count": len(
            merge_plans
        ),
        "relationships_to_remove": (
            relationships_to_remove
        ),
        "status": "semantic_analysis_completed",
    }


def route_after_analysis(
    state: RelationshipMergeState,
) -> Literal[
    "dry_run",
    "nothing_to_merge",
    "apply_merges",
]:
    if state["dry_run"]:
        return "dry_run"

    if not state["merge_plans"]:
        return "nothing_to_merge"

    return "apply_merges"


def dry_run_node(
    state: RelationshipMergeState,
) -> Dict[str, Any]:
    """仅分析，不修改 Neo4j。"""
    print(
        "Dry-run completed. "
        "No relationships were changed."
    )

    return {
        "merged_cluster_count": 0,
        "relationships_removed": 0,
        "remaining_candidate_pairs": (
            state["candidate_pair_count"]
        ),
        "status": "dry_run_completed",
    }


def nothing_to_merge_node(
    state: RelationshipMergeState,
) -> Dict[str, Any]:
    """处理没有高相似度关系的情况。"""
    print(
        "No semantically similar relationship "
        "clusters need to be merged."
    )

    return {
        "merged_cluster_count": 0,
        "relationships_removed": 0,
        "remaining_candidate_pairs": (
            state["candidate_pair_count"]
        ),
        "status": "nothing_to_merge",
    }


def apply_merges_node(
    state: RelationshipMergeState,
) -> Dict[str, Any]:
    """将关系合并计划写入 Neo4j。"""
    with Neo4jClient(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
    ) as client:
        statistics = apply_merge_plans(
            driver=client.get_driver(),
            plans=state["merge_plans"],
            batch_size=state["write_batch_size"],
        )

    print(
        "Merged semantic clusters: "
        f"{statistics.merged_cluster_count}"
    )

    print(
        "Relationships removed: "
        f"{statistics.relationships_removed}"
    )

    return {
        "merged_cluster_count": (
            statistics.merged_cluster_count
        ),
        "relationships_removed": (
            statistics.relationships_removed
        ),
        "status": "merges_applied",
    }


def verify_result_node(
    state: RelationshipMergeState,
) -> Dict[str, Any]:
    """
    重新统计仍有两条及以上关系的节点对。

    剩余节点对不一定表示失败，因为语义相似度较低的
    不同关系本来就应该保留。
    """
    with Neo4jClient(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
    ) as client:
        remaining_groups = (
            load_relationship_groups(
                client.get_driver()
            )
        )

    remaining_count = len(
        remaining_groups
    )

    print(
        "Remaining node pairs with multiple "
        f"relationships: {remaining_count}"
    )

    print(
        "Low-similarity relationships were retained."
    )

    return {
        "remaining_candidate_pairs": remaining_count,
        "status": "completed",
    }


# ---------- LangGraph 构建 ----------


graph_builder = StateGraph(
    RelationshipMergeState
)

graph_builder.add_node(
    "initialize",
    initialize_relationship_merge_node,
)

graph_builder.add_node(
    "scan_relationships",
    scan_relationships_node,
)

graph_builder.add_node(
    "analyze_relationships",
    analyze_relationships_node,
)

graph_builder.add_node(
    "dry_run",
    dry_run_node,
)

graph_builder.add_node(
    "nothing_to_merge",
    nothing_to_merge_node,
)

graph_builder.add_node(
    "apply_merges",
    apply_merges_node,
)

graph_builder.add_node(
    "verify_result",
    verify_result_node,
)

graph_builder.set_entry_point(
    "initialize"
)

graph_builder.add_edge(
    "initialize",
    "scan_relationships",
)

graph_builder.add_edge(
    "scan_relationships",
    "analyze_relationships",
)

graph_builder.add_conditional_edges(
    "analyze_relationships",
    route_after_analysis,
    {
        "dry_run": "dry_run",
        "nothing_to_merge": "nothing_to_merge",
        "apply_merges": "apply_merges",
    },
)

graph_builder.add_edge(
    "dry_run",
    END,
)

graph_builder.add_edge(
    "nothing_to_merge",
    END,
)

graph_builder.add_edge(
    "apply_merges",
    "verify_result",
)

graph_builder.add_edge(
    "verify_result",
    END,
)

MergeRepeatRelationshipsGraph = (
    graph_builder.compile()
)


# ---------- 命令行入口 ----------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge semantically similar relationships "
            "between the same unordered node pair."
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=(
            "Semantic similarity threshold; "
            f"default: "
            f"{DEFAULT_SIMILARITY_THRESHOLD}"
        ),
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=(
            "Progress group size for relationship "
            "embedding; "
            f"default: "
            f"{DEFAULT_EMBEDDING_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=DEFAULT_WRITE_BATCH_SIZE,
        help=(
            "Number of merge plans written to Neo4j "
            "per batch; "
            f"default: "
            f"{DEFAULT_WRITE_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Analyze relationships without "
            "modifying Neo4j."
        ),
    )

    arguments = parser.parse_args()

    if not 0.0 <= arguments.threshold <= 1.0:
        parser.error(
            "--threshold must be between 0 and 1"
        )

    if arguments.embedding_batch_size <= 0:
        parser.error(
            "--embedding-batch-size must be "
            "greater than 0"
        )

    if arguments.write_batch_size <= 0:
        parser.error(
            "--write-batch-size must be "
            "greater than 0"
        )

    return arguments


def main() -> None:
    arguments = parse_arguments()

    initial_state: RelationshipMergeState = {
        "similarity_threshold": (
            arguments.threshold
        ),
        "embedding_batch_size": (
            arguments.embedding_batch_size
        ),
        "write_batch_size": (
            arguments.write_batch_size
        ),
        "dry_run": arguments.dry_run,
    }

    final_state = (
        MergeRepeatRelationshipsGraph.invoke(
            initial_state
        )
    )

    print(
        "\nRelationship semantic merge completed:"
    )

    print(
        "  Candidate node pairs: "
        f"{final_state['candidate_pair_count']}"
    )

    print(
        "  Candidate relationships: "
        f"{final_state['candidate_relationship_count']}"
    )

    print(
        "  Semantic merge clusters: "
        f"{final_state['merge_cluster_count']}"
    )

    print(
        "  Relationships scheduled for removal: "
        f"{final_state['relationships_to_remove']}"
    )

    print(
        "  Relationships actually removed: "
        f"{final_state['relationships_removed']}"
    )

    print(
        "  Remaining multi-relationship pairs: "
        f"{final_state['remaining_candidate_pairs']}"
    )

    print(
        f"  Status: {final_state['status']}"
    )


if __name__ == "__main__":
    main()
