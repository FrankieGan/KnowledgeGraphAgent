from __future__ import annotations
from dependencies import *


# ============ Neo4j 客户端 ============
class Neo4jClient:
    """Neo4j 客户端封装。"""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            print(f"✅ Connected to {self.uri}")
        except Exception as e:
            raise ConnectionError(f"Connection failed: {e}") from e

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            self.driver = None

    def query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行 Cypher 并返回查询结果。
        """
        if not self.driver:
            raise RuntimeError(
                "Connection not open. Call open() or use 'with'."
            )

        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def query_with_counters(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        执行 Cypher，并同时返回 Neo4j 的真实写入统计。
        """
        if not self.driver:
            raise RuntimeError(
                "Connection not open. Call open() or use 'with'."
            )

        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})

            records = [record.data() for record in result]
            summary = result.consume()
            counters = summary.counters

            return records, {
                "nodes_created": counters.nodes_created,
                "nodes_deleted": counters.nodes_deleted,
                "relationships_created": counters.relationships_created,
                "relationships_deleted": counters.relationships_deleted,
                "properties_set": counters.properties_set,
                "labels_added": counters.labels_added,
                "labels_removed": counters.labels_removed,
                "constraints_added": counters.constraints_added,
                "constraints_removed": counters.constraints_removed,
                "indexes_added": counters.indexes_added,
                "indexes_removed": counters.indexes_removed,
            }


# ============ 状态定义 ============
@dataclass
class GraphConstructionState:
    # ---------- 节点构建计划 ----------
    class node_construction:
        def __init__(
            self,
            source_file: str = "",
            label: str = "",
            unique_column_name: str = "",
            properties: list[str] | None = None
        ):
            self.construction_type: str = "node"
            self.source_file: str = source_file
            self.label: str = label
            self.unique_column_name: str = unique_column_name
            self.properties: list[str] = (
                properties if properties is not None else []
            )

        def toString(self) -> str:
            props_str = (
                ", ".join(self.properties)
                if self.properties
                else "none"
            )

            return (
                "Node construction:\n"
                f"  source_file: {self.source_file}\n"
                f"  label: {self.label}\n"
                f"  unique_column_name: {self.unique_column_name}\n"
                f"  properties: {props_str}"
            )

        def to_csv_line(self) -> str:
            props_str = (
                ";".join(self.properties)
                if self.properties
                else ""
            )

            return (
                f"{self.source_file},"
                f"{self.label},"
                f"{self.unique_column_name},"
                f"{props_str}"
            )

    # ---------- 关系构建计划 ----------
    class relationship_construction:
        def __init__(
            self,
            source_file: str = "",
            relationship_type: str = "",
            from_node_label: str = "",
            from_node_column: str = "",
            to_node_label: str = "",
            to_node_column: str = "",
            properties: list[str] | None = None
        ):
            self.construction_type: str = "relationship"
            self.source_file: str = source_file
            self.relationship_type: str = relationship_type
            self.from_node_label: str = from_node_label
            self.from_node_column: str = from_node_column
            self.to_node_label: str = to_node_label
            self.to_node_column: str = to_node_column
            self.properties: list[str] = (
                properties if properties is not None else []
            )

        def toString(self) -> str:
            props_str = (
                ", ".join(self.properties)
                if self.properties
                else "none"
            )

            return (
                "Relationship construction:\n"
                f"  source_file: {self.source_file}\n"
                f"  relationship_type: {self.relationship_type}\n"
                f"  from_node_label: {self.from_node_label}\n"
                f"  from_node_column: {self.from_node_column}\n"
                f"  to_node_label: {self.to_node_label}\n"
                f"  to_node_column: {self.to_node_column}\n"
                f"  properties: {props_str}"
            )

        def to_csv_line(self) -> str:
            props_str = (
                ";".join(self.properties)
                if self.properties
                else ""
            )

            return (
                f"{self.source_file},"
                f"{self.relationship_type},"
                f"{self.from_node_label},"
                f"{self.from_node_column},"
                f"{self.to_node_label},"
                f"{self.to_node_column},"
                f"{props_str}"
            )

    # ---------- 状态变量 ----------
    construction_status: str = "pending"
    messages: list[str] = field(default_factory=list)
    nodes: list[node_construction] = field(default_factory=list)
    relationships: list[relationship_construction] = field(
        default_factory=list
    )

    def add_system(self, text: str) -> None:
        message = f"system: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print(message)


# ============ 通用辅助函数 ============
def read_file_data(file_path: str) -> List[Dict[str, Any]]:
    """
    读取 CSV 或 JSON 文件，返回字典列表。
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        # utf-8-sig 同时兼容有 BOM 和无 BOM 的 UTF-8 CSV
        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            rows = []
            for row in reader:
                cleaned_row = {}

                for key, value in row.items():
                    if key is None:
                        continue

                    cleaned_key = key.strip()

                    if isinstance(value, str):
                        cleaned_value = value.strip()
                    else:
                        cleaned_value = value

                    cleaned_row[cleaned_key] = cleaned_value

                rows.append(cleaned_row)

            return rows

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

        raise ValueError(
            f"Unsupported JSON format in {file_path}"
        )

    raise ValueError(f"Unsupported file type: {ext}")


def quote_identifier(identifier: str) -> str:
    """
    安全引用 Neo4j 标签、属性、关系类型和约束名称。

    例如：
        编号[主键] -> `编号[主键]`
    """
    if identifier is None:
        raise ValueError("Neo4j identifier cannot be None.")

    identifier = str(identifier).strip()

    if not identifier:
        raise ValueError("Neo4j identifier cannot be empty.")

    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"


def clean_plan_value(value: Any) -> str:
    """
    清理构建计划中的字符串值。
    """
    if value is None:
        return ""

    return str(value).strip()


def clean_property_list(value: str) -> list[str]:
    """
    将分号分隔的属性字符串转换为列表。
    """
    if not value:
        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


def get_node_plan_by_label(
    state: GraphConstructionState,
    label: str
) -> Optional[GraphConstructionState.node_construction]:
    """
    根据节点标签查找节点构建计划。
    """
    matches = [
        node
        for node in state.nodes
        if node.label == label
    ]

    if len(matches) == 1:
        return matches[0]

    return None


# ============ 执行图谱构建 ============
def run_node(
    state: GraphConstructionState
) -> GraphConstructionState:
    """
    按照已加载的节点和关系计划构建知识图谱。
    """
    if not state.nodes and not state.relationships:
        state.construction_status = "skipped"
        state.add_system(
            "No nodes or relationships to import. "
            "Skipping graph construction."
        )
        return state

    with Neo4jClient() as db:
        state.add_system("Starting graph construction...")

        # ============ 创建节点唯一性约束 ============
        def create_uniqueness_constraint(
            label: str,
            unique_property_key: str
        ) -> Dict[str, Any]:
            """
            为节点标签和唯一属性创建唯一性约束。
            """
            constraint_name = (
                f"{label}_{unique_property_key}_constraint"
            )

            quoted_constraint = quote_identifier(constraint_name)
            quoted_label = quote_identifier(label)
            quoted_property = quote_identifier(unique_property_key)

            cypher = f"""
                CREATE CONSTRAINT {quoted_constraint} IF NOT EXISTS
                FOR (n:{quoted_label})
                REQUIRE n.{quoted_property} IS UNIQUE
            """

            try:
                db.query(cypher)
                return {"status": "success"}
            except Exception as e:
                return {
                    "status": "error",
                    "error_message": str(e)
                }

        # ============ 存储节点 ============
        def store_node(
            construction: GraphConstructionState.node_construction
        ) -> Dict[str, Any]:
            """
            从 CSV/JSON 文件导入节点。
            """
            label = construction.label
            unique_column = construction.unique_column_name

            if not label:
                return {
                    "status": "error",
                    "error_message": "Node label cannot be empty."
                }

            if not unique_column:
                return {
                    "status": "error",
                    "error_message": (
                        f"Unique column cannot be empty for node "
                        f"label '{label}'."
                    )
                }

            # 1. 创建唯一性约束
            constraint_result = create_uniqueness_constraint(
                label,
                unique_column
            )

            if constraint_result.get("status") == "error":
                return constraint_result

            # 2. 读取源文件
            file_path = os.path.join(
                IMPORT_STRC_DIR,
                construction.source_file
            )
            abs_path = os.path.abspath(file_path)

            try:
                rows = read_file_data(abs_path)
            except Exception as e:
                return {
                    "status": "error",
                    "error_message": f"Failed to read file: {e}"
                }

            if not rows:
                return {
                    "status": "success",
                    "rows_read": 0,
                    "rows_imported": 0,
                    "rows_invalid": 0,
                    "nodes_created": 0
                }

            # 3. 校验字段
            source_headers = set(rows[0].keys())

            if unique_column not in source_headers:
                return {
                    "status": "error",
                    "error_message": (
                        f"Node source file '{construction.source_file}' "
                        f"is missing unique column '{unique_column}'. "
                        f"Available columns: "
                        f"{sorted(source_headers)}"
                    )
                }

            missing_properties = (
                set(construction.properties) - source_headers
            )

            if missing_properties:
                return {
                    "status": "error",
                    "error_message": (
                        f"Node source file '{construction.source_file}' "
                        f"is missing property columns: "
                        f"{sorted(missing_properties)}"
                    )
                }

            # 4. 只保留计划要求的属性
            allowed_keys = set(construction.properties)
            allowed_keys.add(unique_column)

            prepared_rows = []
            invalid_rows = []

            for line_number, row in enumerate(rows, start=2):
                unique_value = row.get(unique_column)

                if isinstance(unique_value, str):
                    unique_value = unique_value.strip()

                if unique_value in (None, ""):
                    invalid_rows.append(line_number)
                    continue

                filtered_row = {}

                for key in allowed_keys:
                    value = row.get(key)

                    if isinstance(value, str):
                        value = value.strip()

                    # Neo4j 属性不能存储 null。
                    # 空字符串是否保留可按需求调整；这里保留非空值。
                    if value not in (None, ""):
                        filtered_row[key] = value

                filtered_row[unique_column] = unique_value
                prepared_rows.append(filtered_row)

            if invalid_rows:
                state.add_system(
                    f"⚠️ Node {label}: skipped "
                    f"{len(invalid_rows)} rows because "
                    f"'{unique_column}' was empty."
                )

            # 5. 批量创建节点
            batch_size = 1000
            total_imported = 0
            total_created = 0

            quoted_label = quote_identifier(label)
            quoted_unique_column = quote_identifier(unique_column)

            cypher = f"""
                UNWIND $rows AS row

                MERGE (
                    n:{quoted_label} {{
                        {quoted_unique_column}: row.__unique_value
                    }}
                )

                SET n += row.__properties

                RETURN count(n) AS matched
            """

            internal_rows = [
                {
                    "__unique_value": row[unique_column],
                    "__properties": row
                }
                for row in prepared_rows
            ]

            for i in range(0, len(internal_rows), batch_size):
                batch = internal_rows[i:i + batch_size]

                try:
                    result, counters = db.query_with_counters(
                        cypher,
                        parameters={"rows": batch}
                    )
                except Exception as e:
                    return {
                        "status": "error",
                        "error_message": (
                            f"Batch node import failed: {e}"
                        )
                    }

                matched = (
                    result[0].get("matched", 0)
                    if result
                    else 0
                )
                created = counters.get("nodes_created", 0)

                total_imported += matched
                total_created += created

                state.add_system(
                    f"Node {label}, batch "
                    f"{i // batch_size + 1}: "
                    f"submitted={len(batch)}, "
                    f"matched={matched}, "
                    f"created={created}"
                )

            return {
                "status": "success",
                "rows_read": len(rows),
                "rows_imported": total_imported,
                "rows_invalid": len(invalid_rows),
                "nodes_created": total_created
            }

        # ============ 存储关系 ============
        def store_relationship(
            construction:
            GraphConstructionState.relationship_construction
        ) -> Dict[str, Any]:
            """
            使用现有七列关系计划格式导入关系。

            关系计划语义：

            source_file:
                关系数据来源，也是源节点的数据来源。

            from_node_label:
                源节点标签。

            from_node_column:
                源文件中用于指向目标节点的外键列。

            to_node_label:
                目标节点标签。

            to_node_column:
                目标节点上用于接受外键匹配的属性。

            源节点自身的匹配属性不写在关系计划中，
            而是通过节点计划中的 unique_column_name 自动获得。

            例如：

            楼层.csv,PART_OF_BUILDING,Floor,所属,
            Building,编号[主键],

            表示：

            Floor.编号[主键] = 楼层.csv.编号[主键]
            Building.编号[主键] = 楼层.csv.所属
            """
            source_file = construction.source_file
            relationship_type = construction.relationship_type
            from_label = construction.from_node_label
            foreign_key_column = construction.from_node_column
            to_label = construction.to_node_label
            target_property = construction.to_node_column

            if not relationship_type:
                return {
                    "status": "error",
                    "error_message": (
                        "Relationship type cannot be empty."
                    )
                }

            # 1. 查找源节点计划
            from_node_plans = [
                node
                for node in state.nodes
                if node.label == from_label
            ]

            if not from_node_plans:
                return {
                    "status": "error",
                    "error_message": (
                        f"Cannot find node construction plan for "
                        f"source label '{from_label}'."
                    )
                }

            if len(from_node_plans) > 1:
                return {
                    "status": "error",
                    "error_message": (
                        f"Multiple node construction plans use "
                        f"source label '{from_label}'. "
                        f"The source unique column is ambiguous."
                    )
                }

            from_node_plan = from_node_plans[0]
            source_unique_column = (
                from_node_plan.unique_column_name
            )

            # 2. 查找并校验目标节点计划
            to_node_plans = [
                node
                for node in state.nodes
                if node.label == to_label
            ]

            if not to_node_plans:
                return {
                    "status": "error",
                    "error_message": (
                        f"Cannot find node construction plan for "
                        f"target label '{to_label}'."
                    )
                }

            if len(to_node_plans) > 1:
                return {
                    "status": "error",
                    "error_message": (
                        f"Multiple node construction plans use "
                        f"target label '{to_label}'."
                    )
                }

            to_node_plan = to_node_plans[0]

            target_node_properties = set(
                to_node_plan.properties
            )
            target_node_properties.add(
                to_node_plan.unique_column_name
            )

            if target_property not in target_node_properties:
                return {
                    "status": "error",
                    "error_message": (
                        f"Target property '{target_property}' is not "
                        f"imported for target node label '{to_label}'. "
                        f"Available imported properties: "
                        f"{sorted(target_node_properties)}"
                    )
                }

            # 3. 读取关系源文件
            file_path = os.path.join(
                IMPORT_STRC_DIR,
                source_file
            )
            abs_path = os.path.abspath(file_path)

            try:
                rows = read_file_data(abs_path)
            except Exception as e:
                return {
                    "status": "error",
                    "error_message": (
                        f"Failed to read relationship source file: {e}"
                    )
                }

            if not rows:
                return {
                    "status": "success",
                    "rows_read": 0,
                    "rows_submitted": 0,
                    "rows_invalid": 0,
                    "relationships_matched": 0,
                    "relationships_created": 0,
                    "relationships_unmatched": 0
                }

            # 4. 校验关系源文件字段
            source_headers = set(rows[0].keys())

            required_columns = {
                source_unique_column,
                foreign_key_column
            }

            missing_columns = (
                required_columns - source_headers
            )

            if missing_columns:
                return {
                    "status": "error",
                    "error_message": (
                        f"Relationship source file '{source_file}' "
                        f"is missing required columns: "
                        f"{sorted(missing_columns)}. "
                        f"Available columns: "
                        f"{sorted(source_headers)}"
                    )
                }

            missing_relationship_properties = (
                set(construction.properties) - source_headers
            )

            if missing_relationship_properties:
                return {
                    "status": "error",
                    "error_message": (
                        f"Relationship source file '{source_file}' "
                        f"is missing relationship property columns: "
                        f"{sorted(missing_relationship_properties)}"
                    )
                }

            # 5. 转换为统一的内部数据结构
            prepared_rows = []
            invalid_rows = []

            for line_number, row in enumerate(rows, start=2):
                source_value = row.get(source_unique_column)
                target_value = row.get(foreign_key_column)

                if isinstance(source_value, str):
                    source_value = source_value.strip()

                if isinstance(target_value, str):
                    target_value = target_value.strip()

                if source_value in (None, ""):
                    invalid_rows.append({
                        "line": line_number,
                        "reason": (
                            f"empty source identifier "
                            f"'{source_unique_column}'"
                        )
                    })
                    continue

                if target_value in (None, ""):
                    invalid_rows.append({
                        "line": line_number,
                        "reason": (
                            f"empty target foreign key "
                            f"'{foreign_key_column}'"
                        )
                    })
                    continue

                relationship_properties = {}

                for property_name in construction.properties:
                    property_value = row.get(property_name)

                    if isinstance(property_value, str):
                        property_value = property_value.strip()

                    if property_value not in (None, ""):
                        relationship_properties[property_name] = (
                            property_value
                        )

                prepared_rows.append({
                    "__source_value": source_value,
                    "__target_value": target_value,
                    "__properties": relationship_properties
                })

            if invalid_rows:
                state.add_system(
                    f"⚠️ Relationship {relationship_type}: "
                    f"skipped {len(invalid_rows)} rows with "
                    f"empty source identifier or target foreign key."
                )

                # 日志只展示前 10 条，避免大量刷屏
                for invalid_row in invalid_rows[:10]:
                    state.add_system(
                        f"  Invalid row "
                        f"{invalid_row['line']}: "
                        f"{invalid_row['reason']}"
                    )

                if len(invalid_rows) > 10:
                    state.add_system(
                        f"  ... and "
                        f"{len(invalid_rows) - 10} more invalid rows."
                    )

            # 6. 批量匹配并创建关系
            batch_size = 1000

            total_submitted = len(prepared_rows)
            total_matched = 0
            total_created = 0

            quoted_from_label = quote_identifier(from_label)
            quoted_source_unique = quote_identifier(
                source_unique_column
            )
            quoted_to_label = quote_identifier(to_label)
            quoted_target_property = quote_identifier(
                target_property
            )
            quoted_relationship_type = quote_identifier(
                relationship_type
            )

            cypher = f"""
                UNWIND $rows AS row

                MATCH (from_node:{quoted_from_label})
                WHERE from_node.{quoted_source_unique}
                    = row.__source_value

                MATCH (to_node:{quoted_to_label})
                WHERE to_node.{quoted_target_property}
                    = row.__target_value

                MERGE (
                    from_node
                )-[r:{quoted_relationship_type}]->(
                    to_node
                )

                SET r += row.__properties

                RETURN count(r) AS matched
            """

            for i in range(
                0,
                total_submitted,
                batch_size
            ):
                batch = prepared_rows[i:i + batch_size]

                try:
                    result, counters = db.query_with_counters(
                        cypher,
                        parameters={"rows": batch}
                    )
                except Exception as e:
                    return {
                        "status": "error",
                        "error_message": (
                            f"Batch relationship import failed: {e}"
                        )
                    }

                matched = (
                    result[0].get("matched", 0)
                    if result
                    else 0
                )
                created = counters.get(
                    "relationships_created",
                    0
                )

                total_matched += matched
                total_created += created

                state.add_system(
                    f"Relationship {relationship_type}, batch "
                    f"{i // batch_size + 1}: "
                    f"submitted={len(batch)}, "
                    f"matched={matched}, "
                    f"created={created}"
                )

            total_unmatched = (
                total_submitted - total_matched
            )

            # 7. 如果有匹配失败，执行诊断查询
            if total_unmatched > 0:
                diagnosis_cypher = f"""
                    UNWIND $rows AS row

                    OPTIONAL MATCH (
                        from_node:{quoted_from_label}
                    )
                    WHERE from_node.{quoted_source_unique}
                        = row.__source_value

                    OPTIONAL MATCH (
                        to_node:{quoted_to_label}
                    )
                    WHERE to_node.{quoted_target_property}
                        = row.__target_value

                    RETURN
                        count(*) AS submitted,

                        count(from_node) AS source_matched,

                        count(to_node) AS target_matched,

                        count(
                            CASE
                                WHEN from_node IS NOT NULL
                                 AND to_node IS NOT NULL
                                THEN 1
                            END
                        ) AS both_matched,

                        collect(
                            CASE
                                WHEN from_node IS NULL
                                THEN row.__source_value
                            END
                        )[0..10] AS missing_source_examples,

                        collect(
                            CASE
                                WHEN to_node IS NULL
                                THEN row.__target_value
                            END
                        )[0..10] AS missing_target_examples
                """

                try:
                    diagnosis_result = db.query(
                        diagnosis_cypher,
                        parameters={"rows": prepared_rows}
                    )
                except Exception as e:
                    state.add_system(
                        f"⚠️ Could not run relationship diagnosis "
                        f"for {relationship_type}: {e}"
                    )
                    diagnosis_result = []

                if diagnosis_result:
                    diagnosis = diagnosis_result[0]

                    missing_source_examples = [
                        value
                        for value in diagnosis.get(
                            "missing_source_examples",
                            []
                        )
                        if value is not None
                    ]

                    missing_target_examples = [
                        value
                        for value in diagnosis.get(
                            "missing_target_examples",
                            []
                        )
                        if value is not None
                    ]

                    state.add_system(
                        f"⚠️ Relationship {relationship_type} "
                        f"diagnosis: "
                        f"submitted="
                        f"{diagnosis.get('submitted', 0)}, "
                        f"source_matched="
                        f"{diagnosis.get('source_matched', 0)}, "
                        f"target_matched="
                        f"{diagnosis.get('target_matched', 0)}, "
                        f"both_matched="
                        f"{diagnosis.get('both_matched', 0)}"
                    )

                    if missing_source_examples:
                        state.add_system(
                            f"  Missing {from_label} examples "
                            f"for property "
                            f"'{source_unique_column}': "
                            f"{missing_source_examples}"
                        )

                    if missing_target_examples:
                        state.add_system(
                            f"  Missing {to_label} examples "
                            f"for property "
                            f"'{target_property}': "
                            f"{missing_target_examples}"
                        )

            return {
                "status": "success",
                "rows_read": len(rows),
                "rows_submitted": total_submitted,
                "rows_invalid": len(invalid_rows),
                "relationships_matched": total_matched,
                "relationships_created": total_created,
                "relationships_unmatched": total_unmatched
            }

        # ============ 先导入所有节点 ============
        for node in state.nodes:
            state.add_system(
                f"Importing node: {node.label}"
            )

            result = store_node(node)

            if result.get("status") == "error":
                state.construction_status = "failed"
                state.add_system(
                    f"❌ Error importing node {node.label}: "
                    f"{result.get('error_message')}"
                )
                return state

            state.add_system(
                f"✅ Node {node.label}: "
                f"read={result.get('rows_read', 0)}, "
                f"imported={result.get('rows_imported', 0)}, "
                f"created={result.get('nodes_created', 0)}, "
                f"invalid={result.get('rows_invalid', 0)}"
            )

        # ============ 再导入所有关系 ============
        for relationship in state.relationships:
            state.add_system(
                f"Importing relationship: "
                f"{relationship.relationship_type}"
            )

            result = store_relationship(relationship)

            if result.get("status") == "error":
                state.construction_status = "failed"
                state.add_system(
                    f"❌ Error importing relationship "
                    f"{relationship.relationship_type}: "
                    f"{result.get('error_message')}"
                )
                #return state

            state.add_system(
                f"✅ Relationship "
                f"{relationship.relationship_type}: "
                f"read={result.get('rows_read', 0)}, "
                f"submitted={result.get('rows_submitted', 0)}, "
                f"matched={result.get('relationships_matched', 0)}, "
                f"created={result.get('relationships_created', 0)}, "
                f"unmatched="
                f"{result.get('relationships_unmatched', 0)}, "
                f"invalid={result.get('rows_invalid', 0)}"
            )

        if state.construction_status != "failed":
            state.construction_status = "completed"
            state.add_system("🎉 Graph construction completed successfully!")

    return state


# ============ 加载构建计划 ============
def import_node(
    state: GraphConstructionState
) -> GraphConstructionState:
    """
    从 STRUCTURED_NODE_CONSTRUCTION_PATH 和 STRUCTURED_RELATION_CONSTRUCTION_PATH
    CSV 文件加载节点与关系构建计划。
    """

    # 避免同一状态重复执行时重复追加计划
    state.nodes.clear()
    state.relationships.clear()

    # ============ 加载节点计划 ============
    try:
        with open(
            STRUCTURED_NODE_CONSTRUCTION_PATH,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for line_number, row in enumerate(
                reader,
                start=2
            ):
                # 跳过完全空白的行
                if not row or not any(
                    str(value).strip()
                    for value in row
                ):
                    continue

                if len(row) < 4:
                    state.add_system(
                        f"⚠️ Invalid node construction row "
                        f"{line_number}: expected at least "
                        f"4 columns, got {len(row)}."
                    )
                    continue

                source_file = clean_plan_value(row[0])
                label = clean_plan_value(row[1])
                unique_column_name = clean_plan_value(row[2])
                properties = clean_property_list(row[3])

                if not source_file:
                    state.add_system(
                        f"⚠️ Invalid node construction row "
                        f"{line_number}: source_file is empty."
                    )
                    continue

                if not label:
                    state.add_system(
                        f"⚠️ Invalid node construction row "
                        f"{line_number}: label is empty."
                    )
                    continue

                if not unique_column_name:
                    state.add_system(
                        f"⚠️ Invalid node construction row "
                        f"{line_number}: "
                        f"unique_column_name is empty."
                    )
                    continue

                node = state.node_construction(
                    source_file=source_file,
                    label=label,
                    unique_column_name=unique_column_name,
                    properties=properties
                )

                state.nodes.append(node)

        state.add_system(
            f"Loaded {len(state.nodes)} node construction "
            f"plans from '{STRUCTURED_NODE_CONSTRUCTION_PATH}'."
        )

    except FileNotFoundError:
        state.add_system(
            f"⚠️ Node construction file "
            f"'{STRUCTURED_NODE_CONSTRUCTION_PATH}' not found. "
            f"Starting with empty nodes."
        )
    except Exception as e:
        state.add_system(
            f"⚠️ Error reading node construction file: {e}"
        )

    # ============ 加载关系计划 ============
    try:
        with open(
            STRUCTURED_RELATION_CONSTRUCTION_PATH,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for line_number, row in enumerate(
                reader,
                start=2
            ):
                # 跳过完全空白的行
                if not row or not any(
                    str(value).strip()
                    for value in row
                ):
                    continue

                if len(row) < 7:
                    state.add_system(
                        f"⚠️ Invalid relationship construction "
                        f"row {line_number}: expected at least "
                        f"7 columns, got {len(row)}."
                    )
                    continue

                source_file = clean_plan_value(row[0])
                relationship_type = clean_plan_value(row[1])
                from_label = clean_plan_value(row[2])
                from_column = clean_plan_value(row[3])
                to_label = clean_plan_value(row[4])
                to_column = clean_plan_value(row[5])
                properties = clean_property_list(row[6])

                required_values = {
                    "source_file": source_file,
                    "relationship_type": relationship_type,
                    "from_node_label": from_label,
                    "from_node_column": from_column,
                    "to_node_label": to_label,
                    "to_node_column": to_column
                }

                empty_fields = [
                    name
                    for name, value in required_values.items()
                    if not value
                ]

                if empty_fields:
                    state.add_system(
                        f"⚠️ Invalid relationship construction "
                        f"row {line_number}: empty fields "
                        f"{empty_fields}."
                    )
                    continue

                relationship = state.relationship_construction(
                    source_file=source_file,
                    relationship_type=relationship_type,
                    from_node_label=from_label,
                    from_node_column=from_column,
                    to_node_label=to_label,
                    to_node_column=to_column,
                    properties=properties
                )

                state.relationships.append(relationship)

        state.add_system(
            f"Loaded {len(state.relationships)} relationship "
            f"construction plans from "
            f"'{STRUCTURED_RELATION_CONSTRUCTION_PATH}'."
        )

    except FileNotFoundError:
        state.add_system(
            f"⚠️ Relationship construction file "
            f"'{STRUCTURED_RELATION_CONSTRUCTION_PATH}' not found. "
            f"Starting with empty relationships."
        )
    except Exception as e:
        state.add_system(
            f"⚠️ Error reading relationship construction "
            f"file: {e}"
        )

    return state


# ============ LangGraph 图构建 ============
graph_builder = StateGraph(GraphConstructionState)

graph_builder.add_node("import", import_node)
graph_builder.add_node("run", run_node)

graph_builder.set_entry_point("import")
graph_builder.add_edge("import", "run")
graph_builder.add_edge("run", END)

StructuredGraphConstructionGraph = graph_builder.compile()


# ============ 程序入口 ============
if __name__ == "__main__":
    initial_state = GraphConstructionState()
    final_state = StructuredGraphConstructionGraph.invoke(
        initial_state
    )
