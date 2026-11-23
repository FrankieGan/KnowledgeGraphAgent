from __future__ import annotations
from dependencies import *

@dataclass
class StructuredProposalAgentState(BudgetState):
    # 内部类：节点构建
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
            self.properties: list[str] = properties if properties is not None else []

        def toString(self) -> str:
            props_str = ", ".join(self.properties) if self.properties else "none"
            return (
                f"Node construction:\n"
                f"  source_file: {self.source_file}\n"
                f"  label: {self.label}\n"
                f"  unique_column_name: {self.unique_column_name}\n"
                f"  properties: {props_str}"
            )

        def to_csv_line(self) -> str:
            props_str = ";".join(self.properties) if self.properties else ""
            return f"{self.source_file},{self.label},{self.unique_column_name},{props_str}"

    # 内部类：关系构建
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
            self.properties: list[str] = properties if properties is not None else []

        def toString(self) -> str:
            props_str = ", ".join(self.properties) if self.properties else "none"
            return (
                f"Relationship construction:\n"
                f"  source_file: {self.source_file}\n"
                f"  relationship_type: {self.relationship_type}\n"
                f"  from_node_label: {self.from_node_label}\n"
                f"  from_node_column: {self.from_node_column}\n"
                f"  to_node_label: {self.to_node_label}\n"
                f"  to_node_column: {self.to_node_column}\n"
                f"  properties: {props_str}"
            )

        def to_csv_line(self) -> str:
            props_str = ";".join(self.properties) if self.properties else ""
            return f"{self.source_file},{self.relationship_type},{self.from_node_label},{self.from_node_column},{self.to_node_label},{self.to_node_column},{props_str}"

    # ---------- 状态变量 ----------
    messages: list[str] = field(default_factory=list)
    approved_user_goal: str = ""
    approved_files: list[str] = field(default_factory=list)
    nodes: list[node_construction] = field(default_factory=list)
    relationships: list[relationship_construction] = field(default_factory=list)
    isComplete: bool = False
    feedback: str = ""               # Critic 反馈
    userFeedback: str = ""           # 新增：用户直接提供的反馈

    _current_instance: "StructuredProposalAgentState | None" = None

    def __post_init__(self):
        StructuredProposalAgentState._current_instance = self

    def constructionToString(self) -> str:
        if not self.nodes and not self.relationships:
            return "No construction plan defined yet."
        parts = []
        if self.nodes:
            parts.append("--- Nodes ---")
            for idx, node in enumerate(self.nodes, start=1):
                parts.append(f"Node {idx}:")
                parts.append(node.toString())
        if self.relationships:
            parts.append("--- Relationships ---")
            for idx, rel in enumerate(self.relationships, start=1):
                parts.append(f"Relationship {idx}:")
                parts.append(rel.toString())
        return "\n".join(parts)

    def add_user(self, text: str) -> None:
        message = f"user: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print("Proposal$ " + message)

    def add_agent(self, text: str) -> None:
        message = f"agent: {text}"
        self.messages.append(message)
        print("Proposal$ " + message)

    def add_system(self, text: str) -> None:
        message = f"system: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print("Proposal$ " + message)


# ---------- 节点 ----------
def import_node(state: StructuredProposalAgentState) -> StructuredProposalAgentState:
    try:
        with open(GOAL_SAVE_PATH, "r", encoding="utf-8") as f:
            goal = f.read().strip()
        state.approved_user_goal = goal
        print(f"✅ Loaded approved user goal: '{goal}'")
    except FileNotFoundError:
        state.approved_user_goal = ""
        print(f"⚠️ Goal file '{GOAL_SAVE_PATH}' not found. Starting with empty goal.")
    except Exception as e:
        state.approved_user_goal = ""
        print(f"⚠️ Error reading goal file: {e}")

    try:
        with open(SAVE_STRC_SUGGESTED_PATH, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
        state.approved_files = files
        print(f"✅ Loaded {len(files)} approved files: {files}")
    except FileNotFoundError:
        state.approved_files = []
        print(f"⚠️ Suggested files file '{SAVE_STRC_SUGGESTED_PATH}' not found. Starting with empty list.")
    except Exception as e:
        state.approved_files = []
        print(f"⚠️ Error reading suggested files: {e}")

    # 加载 critic 反馈（原有）
    try:
        with open(STRUCTURED_FEEDBACK_PATH, "r", encoding="utf-8") as f:
            feedback = f.read().strip()
        state.feedback = feedback
        print(f"✅ Loaded feedback: '{feedback}'")
        state.add_system(f"Feedback loaded: {feedback}")
    except FileNotFoundError:
        state.feedback = "no feedback"
        state.add_system("No feedback available.")
        print(f"ℹ️ Feedback file '{STRUCTURED_FEEDBACK_PATH}' not found. Using 'no feedback'.")
    except Exception as e:
        state.feedback = "no feedback"
        state.add_system("No feedback available (error reading file).")
        print(f"⚠️ Error reading feedback file: {e}")

    # ---------- 新增：加载用户反馈 ----------
    try:
        with open(STRUCTURED_USER_FEEDBACK_PATH, "r", encoding="utf-8") as f:
            user_feedback = f.read().strip()
        state.userFeedback = user_feedback
        state.add_system(f"User feedback loaded: {user_feedback}")
        print(f"✅ Loaded user feedback: '{user_feedback}'")
    except FileNotFoundError:
        state.userFeedback = ""
        state.add_system("No user feedback available.")
        print(f"ℹ️ User feedback file '{STRUCTURED_USER_FEEDBACK_PATH}' not found.")
    except Exception as e:
        state.userFeedback = ""
        state.add_system("No user feedback (error reading file).")
        print(f"⚠️ Error reading user feedback file: {e}")

    # 加载节点构建计划
    try:
        with open(STRUCTURED_NODE_CONSTRUCTION_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                source_file, label, unique_column_name, props_str = parts[0], parts[1], parts[2], parts[3]
                properties = props_str.split(';') if props_str else []
                node = state.node_construction(
                    source_file=source_file,
                    label=label,
                    unique_column_name=unique_column_name,
                    properties=properties
                )
                state.nodes.append(node)
        print(f"✅ Loaded {len(state.nodes)} nodes from '{STRUCTURED_NODE_CONSTRUCTION_PATH}'.")
    except FileNotFoundError:
        print(f"ℹ️ Node file '{STRUCTURED_NODE_CONSTRUCTION_PATH}' not found. Starting with empty nodes.")
    except Exception as e:
        print(f"⚠️ Error reading node file: {e}")

    # 加载关系构建计划
    try:
        with open(STRUCTURED_RELATION_CONSTRUCTION_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                source_file, rel_type, from_label, from_col, to_label, to_col, props_str = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
                properties = props_str.split(';') if props_str else []
                rel = state.relationship_construction(
                    source_file=source_file,
                    relationship_type=rel_type,
                    from_node_label=from_label,
                    from_node_column=from_col,
                    to_node_label=to_label,
                    to_node_column=to_col,
                    properties=properties
                )
                state.relationships.append(rel)
        print(f"✅ Loaded {len(state.relationships)} relationships from '{STRUCTURED_RELATION_CONSTRUCTION_PATH}'.")
    except FileNotFoundError:
        print(f"ℹ️ Relationship file '{STRUCTURED_RELATION_CONSTRUCTION_PATH}' not found. Starting with empty relationships.")
    except Exception as e:
        print(f"⚠️ Error reading relationship file: {e}")

    return state


def save_node(state: StructuredProposalAgentState) -> StructuredProposalAgentState:
    # 保存节点
    try:
        with open(STRUCTURED_NODE_CONSTRUCTION_PATH, "w", encoding="utf-8") as f:
            f.write("source_file,label,unique_column_name,properties\n")
            for node in state.nodes:
                f.write(node.to_csv_line() + "\n")
        print(f"✅ Nodes saved to '{STRUCTURED_NODE_CONSTRUCTION_PATH}'.")
    except Exception as e:
        print(f"⚠️ Error saving nodes: {e}")

    # 保存关系
    try:
        with open(STRUCTURED_RELATION_CONSTRUCTION_PATH, "w", encoding="utf-8") as f:
            f.write("source_file,relationship_type,from_node_label,from_node_column,to_node_label,to_node_column,properties\n")
            for rel in state.relationships:
                f.write(rel.to_csv_line() + "\n")
        print(f"✅ Relationships saved to '{STRUCTURED_RELATION_CONSTRUCTION_PATH}'.")
    except Exception as e:
        print(f"⚠️ Error saving relationships: {e}")

    # 删除反馈文件
    try:
        if os.path.exists(STRUCTURED_FEEDBACK_PATH):
            os.remove(STRUCTURED_FEEDBACK_PATH)
            print(f"🗑️ Feedback file '{STRUCTURED_FEEDBACK_PATH}' deleted.")
        else:
            print(f"ℹ️ Feedback file '{STRUCTURED_FEEDBACK_PATH}' does not exist, skipping deletion.")
    except Exception as e:
        print(f"⚠️ Error deleting feedback file: {e}")

    state.isComplete = False
    return state


def rest_node(state: StructuredProposalAgentState) -> StructuredProposalAgentState:
    state.add_system("You can talk here but you should call a tool.")
    return state


# ---------- 工具定义 ----------
@tool
def get_approved_user_goal() -> str:
    """

    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    return state.approved_user_goal


@tool
def get_approved_files() -> list[str]:
    """

    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return ["Error: No state instance available."]
    return state.approved_files


@tool
def sample_file(file_names: list[str]) -> dict[str, str]:
    """

    :param file_names:
    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return {"error": "No state instance available."}
    result = {}
    for fname in file_names:
        file_path = os.path.join(IMPORT_STRC_DIR, fname)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= MAX_READ:
                        break
                    lines.append(line.rstrip('\n'))
                content = "\n".join(lines)
                if len(lines) >= MAX_READ:
                    content += "\n... (truncated)"
                result[fname] = content
        except FileNotFoundError:
            result[fname] = f"Error: File '{fname}' not found."
        except Exception as e:
            result[fname] = f"Error reading file: {e}"
    return result





@tool
def search_file(file_names: list[str], search_string: str) -> list[str]:
    """
    Search files for a string and return up to MAX_SEARCH matches.

    :param file_names: Names of the files to search.
    :param search_string: String to search for.
    :return: Search results, optionally followed by a truncation notice.
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return ["Error: No state instance available."]

    matches_by_file: dict[str, list[str]] = {}
    messages: list[str] = []
    match_count = 0
    results_truncated = False

    for fname in file_names:
        file_path = os.path.join(IMPORT_STRC_DIR, fname)

        try:
            file_matches = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    if search_string not in line:
                        continue

                    if match_count >= MAX_SEARCH:
                        results_truncated = True
                        break

                    file_matches.append(
                        f"Line {line_num}: {line.strip()}"
                    )
                    match_count += 1

            if file_matches:
                matches_by_file[fname] = file_matches
            elif not results_truncated:
                messages.append(f"{fname}: No result found.")

            if results_truncated:
                break

        except FileNotFoundError:
            messages.append(f"{fname}: File not found.")
        except Exception as e:
            messages.append(f"{fname}: Error reading file - {e}")

    results = []

    for fname, file_matches in matches_by_file.items():
        results.append(f"--- {fname} ---")
        results.extend(file_matches)

    results.extend(messages)

    if results_truncated:
        results.append(
            f"Results truncated: only the first {MAX_SEARCH} matches are shown."
        )

    return results if results else ["No files searched."]

@tool
def propose_node_construction(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: list[str] | None = None
) -> str:
    """

    :param source_file:
    :param label:
    :param unique_column_name:
    :param properties:
    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    for node in state.nodes:
        if node.label == label:
            return f"Error: Node with label '{label}' already exists."
    new_node = state.node_construction(
        source_file=source_file,
        label=label,
        unique_column_name=unique_column_name,
        properties=properties or []
    )
    state.nodes.append(new_node)
    return f"Node construction added: {label}"


@tool
def propose_relationship_construction(
    source_file: str,
    relationship_type: str,
    from_node_label: str,
    from_node_column: str,
    to_node_label: str,
    to_node_column: str,
    properties: list[str] | None = None
) -> str:
    """

    :param source_file:
    :param relationship_type:
    :param from_node_label:
    :param from_node_column:
    :param to_node_label:
    :param to_node_column:
    :param properties:
    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    for rel in state.relationships:
        if rel.from_node_label == from_node_label and rel.to_node_label == to_node_label:
            return f"Error: Relationship from '{from_node_label}' to '{to_node_label}' already exists."
    new_rel = state.relationship_construction(
        source_file=source_file,
        relationship_type=relationship_type,
        from_node_label=from_node_label,
        from_node_column=from_node_column,
        to_node_label=to_node_label,
        to_node_column=to_node_column,
        properties=properties or []
    )
    state.relationships.append(new_rel)
    return f"Relationship construction added: {relationship_type} ({from_node_label} -> {to_node_label})"


@tool
def remove_node_construction(label: str) -> str:
    """

    :param label:
    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    initial_len = len(state.nodes)
    state.nodes = [node for node in state.nodes if node.label != label]
    if len(state.nodes) == initial_len:
        return f"No node with label '{label}' found."
    return f"Node construction for '{label}' removed."


@tool
def remove_relationship_construction(from_node_label: str, to_node_label: str) -> str:
    """

    :param from_node_label:
    :param to_node_label:
    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    initial_len = len(state.relationships)
    state.relationships = [rel for rel in state.relationships
                           if not (rel.from_node_label == from_node_label and rel.to_node_label == to_node_label)]
    if len(state.relationships) == initial_len:
        return f"No relationship from '{from_node_label}' to '{to_node_label}' found."
    return f"Relationship construction from '{from_node_label}' to '{to_node_label}' removed."


@tool
def get_proposed_construction_plan() -> str:
    """

    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    return state.constructionToString()


@tool
def construction_complete() -> str:
    """

    :return:
    """
    state = StructuredProposalAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    if not state.nodes and not state.relationships:
        return "Error: Cannot mark complete with an empty construction plan."
    state.isComplete = True
    return "Construction plan marked as complete."


tools = [
    get_approved_user_goal,
    get_approved_files,
    sample_file,
    search_file,
    propose_node_construction,
    propose_relationship_construction,
    remove_node_construction,
    remove_relationship_construction,
    get_proposed_construction_plan,
    construction_complete,
]

tool_map = {t.name: t for t in tools}


model = LLM_CHAT_COMPLETION


def history_to_chat_messages(state: StructuredProposalAgentState) -> list[tuple[str, str]]:
    chat_messages = []
    for message in state.messages:
        if message.startswith("user: "):
            chat_messages.append(("user", message.removeprefix("user: ")))
        elif message.startswith("agent: "):
            chat_messages.append(("assistant", message.removeprefix("agent: ")))
        elif message.startswith("system: "):
            chat_messages.append(("system", message.removeprefix("system: ")))
    return chat_messages


def agent_node(state: StructuredProposalAgentState) -> StructuredProposalAgentState:
    def parse_agent_output(content) -> str:
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    item_text = item.get("text") or item.get("content") or ""
                    if isinstance(item_text, str):
                        parts.append(item_text)
            text = "\n".join(parts)
        else:
            text = str(content)
        return (
            text.replace("```text", "")
                .replace("```python", "")
                .replace("```json", "")
                .replace("```", "")
                .strip()
        )

    system_prompt = (
        "You are an expert at knowledge graph modeling with property graphs. "
        "Your task is to propose an appropriate schema by specifying construction rules which transform approved files into nodes or relationships. "
        "The resulting schema should describe a knowledge graph based on the user goal.\n\n"
        "**Consider feedback if it is available:**\n"
        "<feedback>\n"
        "{feedback}\n"
        "</feedback>\n\n"
        "Every file in the approved files list will become either a node or a relationship. "
        "Determining whether a file likely represents a node or a relationship is based on hints from the filename (is it a single thing or two things) and the identifiers found within the file.\n\n"
        "Because unique identifiers are so important for determining the structure of the graph, "
        "always verify the uniqueness of suspected unique identifiers using the 'search_file' tool.\n\n"
        "**General guidance for identifying a node or a relationship:**\n"
        "- If the file name is singular and has only 1 unique identifier it is likely a node.\n"
        "- If the file name is a combination of two things, it is likely a full relationship.\n"
        "- If the file name sounds like a node, but there are multiple unique identifiers, that is likely a node with reference relationships.\n\n"
        "**Design rules for nodes:**\n"
        "- Nodes will have unique identifiers.\n"
        "- Nodes _may_ have identifiers that are used as reference relationships.\n\n"
        "**Design rules for relationships:**\n"
        "Relationships appear in two ways: full relationships and reference relationships.\n\n"
        "Full relationships:\n"
        "- Full relationships appear in dedicated relationship files, often having a filename that references two nodes.\n"
        "- Full relationships typically have references to a source and destination node.\n"
        "- Full relationships _do not have_ unique identifiers, but instead have references to the primary keys of the source and destination nodes.\n"
        "- The absence of a single, unique identifier is a strong indicator that a file is a full relationship.\n\n"
        "Reference relationships:\n"
        "- Reference relationships appear as foreign key references in node files.\n"
        "- Reference relationship foreign key column names often hint at the destination node and relationship type.\n"
        "- References may be hierarchical container relationships, with terminology revealing parent-child, 'has', 'contains', membership, or similar relationship.\n"
        "- References may be peer relationships, that is often a self-reference to a similar class of nodes. For example, 'knows' or 'see also'.\n\n"
        "**Mandatory column and property matching rules for the current importer:**\n"
        "- Inspect the actual source-file headers with 'sample_file' and verify mappings with 'search_file'; copy column names exactly, including case and suffixes such as '[主键]'. Never invent or rename source columns.\n"
        "- For each node construction, unique_column_name and every entry in properties must exist in that node's source_file. A node imports only unique_column_name plus properties; other source columns do not automatically become node properties.\n"
        "- A relationship's from_node_label and to_node_label must each resolve to exactly one proposed node construction.\n"
        "- from_node_column is the FOREIGN KEY column in the relationship's source_file whose values identify the TARGET node. It is not the source node's own matching property and need not be imported as a source node property.\n"
        "- to_node_column is the TARGET NODE PROPERTY used to match that foreign key. It must exist in the target node's original source file AND be included in that node construction's unique_column_name or properties; normally use the target's unique_column_name. Do not use the source foreign-key column name as a target property unless that exact property is actually imported on the target.\n"
        "- The importer matches the source node automatically using its node construction's unique_column_name. That column must also exist in the relationship's source_file, and its row values must identify the intended source nodes. Every relationship properties entry must exist in the relationship's source_file.\n"
        "- Verify both joins against real values: source_node[unique_column_name] = row[unique_column_name], and target_node[to_node_column] = row[from_node_column]. A shared column name alone does not prove that identifiers refer to the same nodes.\n"
        "- For example, Department rows with '编号[主键]' and '所属学校编号' referencing School nodes keyed by '编号[主键]' require from_node_column='所属学校编号' and to_node_column='编号[主键]'.\n"
        "- For Course nodes importing '编号[主键]', '名称', '学分', and '学时', '先修课程编号' from a relationship file is a foreign-key column, not an imported Course property. Also, a separate prerequisite record's own '编号[主键]' identifies that record, not a Course. Do not propose a direct Course-to-Course rule that violates either join; use a justified association-node construction when supported by the source data, or report the unsupported mapping.\n"
        "- Before calling 'construction_complete', review every relationship against both node plans and the actual file headers and values; correct all mismatches.\n\n"
        "The resulting schema should be a connected graph, with no isolated components.\n\n"
        "**Tools at your disposal:**\n"
        "- get_approved_user_goal(): retrieve the approved user goal.\n"
        "- get_approved_files(): retrieve the list of approved files.\n"
        "- get_proposed_construction_plan(): retrieve the current construction plan.\n"
        "- sample_file(file_names: list[str]): preview the first 100 lines of specified files.\n"
        "- search_file(file_names: list[str], search_string: str): search for a string in the specified files.\n"
        "- propose_node_construction(source_file, label, unique_column_name, properties): add a node construction.\n"
        "- propose_relationship_construction(source_file, relationship_type, from_node_label, from_node_column, to_node_label, to_node_column, properties): add a relationship construction.\n"
        "- remove_node_construction(label): remove a node construction by label.\n"
        "- remove_relationship_construction(from_node_label, to_node_label): remove a relationship construction.\n"
        "- construction_complete(): mark the plan as complete.\n\n"
        "**Workflow:**\n"
        "1. Get goal and files.\n"
        "2. For each file, decide node or relationship using 'sample_file' and 'search_file' to verify unique identifiers.\n"
        "3. Propose constructions with the appropriate tools.\n"
        "4. Use 'get_proposed_construction_plan' to review.\n"
        "5. When done, call 'construction_complete'.\n\n"
        "**Critical rules:**\n"
        "- Only one tool call per response.\n"
        "- Entire response must be exactly: [tool_call] tool_name(args)\n"
        "- No extra text before or after the tool call.\n"
        "- Use natural language for conversation (no tool needed).\n\n"
        "Examples of valid tool calls:\n"
        "[tool_call] get_approved_user_goal()\n"
        "[tool_call] get_approved_files()\n"
        "[tool_call] sample_file(['customers.csv', 'orders.csv'])\n"
        "[tool_call] search_file(['customers.csv'], 'customer_id')\n"
        "[tool_call] propose_node_construction('customers.csv', 'Customer', 'customer_id', ['name', 'email'])\n"
        "[tool_call] propose_relationship_construction('orders.csv', 'PlacedBy', 'Order', 'customer_id', 'Customer', 'customer_id', ['order_date'])\n"
        "[tool_call] get_proposed_construction_plan()\n"
        "[tool_call] construction_complete()"
    )

    messages = [
        SystemMessage(content=system_prompt),
        *history_to_chat_messages(state),
    ]

    response = model.invoke(messages)
    agent_text = parse_agent_output(response.content)
    state.add_agent(agent_text)
    return state


def tool_node(state: StructuredProposalAgentState) -> StructuredProposalAgentState:
    def parse_tool_call(text: str) -> tuple[str | None, list, dict]:
        if TOOL_CALL_MARKER not in text:
            return None, [], {}
        raw_text = text.removeprefix("agent: ").strip()
        call_start = raw_text.find(TOOL_CALL_MARKER)
        call_text = raw_text[call_start:].strip()
        call_text = call_text.removeprefix(TOOL_CALL_MARKER).strip()
        try:
            parsed = ast.parse(call_text, mode="eval")
        except SyntaxError:
            return None, [], {}
        if not isinstance(parsed.body, ast.Call):
            return None, [], {}
        call = parsed.body
        if not isinstance(call.func, ast.Name):
            return None, [], {}
        tool_name = call.func.id
        positional_args = [ast.literal_eval(arg) for arg in call.args]
        keyword_args = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in call.keywords
            if keyword.arg is not None
        }
        return tool_name, positional_args, keyword_args

    def build_tool_input(tool_name: str, positional_args: list, keyword_args: dict) -> dict:
        selected_tool = tool_map[tool_name]
        if keyword_args:
            return keyword_args
        tool_arg_names = list(selected_tool.args.keys())
        return {
            arg_name: arg_value
            for arg_name, arg_value in zip(tool_arg_names, positional_args)
        }

    latest_message = state.messages[-1]
    tool_name, positional_args, keyword_args = parse_tool_call(latest_message)

    if tool_name is None:
        state.add_system("Invalid tool call.")
        return state
    if tool_name not in tool_map:
        state.add_system(f"Unknown tool: {tool_name}")
        return state

    try:
        tool_input = build_tool_input(tool_name, positional_args, keyword_args)
        result = tool_map[tool_name].invoke(tool_input)
        state.add_system(f"{tool_name} returned: {result}")
    except Exception as error:
        state.add_system(f"{tool_name} failed: {error}")

    return state


# ---------- 条件函数 ----------
def should_continue(state: StructuredProposalAgentState) -> Literal["tool", "rest"]:
    latest_message = state.messages[-1]
    if TOOL_CALL_MARKER in latest_message:
        return "tool"
    else:
        return "rest"


def tool_should_continue(state: StructuredProposalAgentState) -> Literal["save", "agent"]:
    if state.isComplete:
        return "save"
    else:
        return "agent"


# ---------- 图构建 ----------
structuredProposalAgentGraph = build_agent_graph(
    globals(), "structuredProposalAgent", StructuredProposalAgentState,
    complete=lambda state: state.isComplete,
    outputs=lambda state: {
        STRUCTURED_NODE_CONSTRUCTION_PATH: csv_text(
            ["source_file", "label", "unique_column_name", "properties"],
            [(n.source_file, n.label, n.unique_column_name, ";".join(n.properties)) for n in state.nodes]),
        STRUCTURED_RELATION_CONSTRUCTION_PATH: csv_text(
            ["source_file", "relationship_type", "from_node_label", "from_node_column", "to_node_label", "to_node_column", "properties"],
            [(r.source_file, r.relationship_type, r.from_node_label, r.from_node_column,
              r.to_node_label, r.to_node_column, ";".join(r.properties)) for r in state.relationships]),
    },
    coordinator=False,
)


if __name__ == "__main__":
    state = StructuredProposalAgentState()
    state.add_user("I need to design a knowledge graph schema based on my goals and files.")
    state.add_agent("[tool_call] sampleCall()")
    final_state = structuredProposalAgentGraph.invoke(state)
    print("\nFull conversation history:")
    for message in final_state["messages"]:
        print(message)
    print("\n--- Construction plan preview ---")
    print(state.constructionToString())
