from dependencies import *


@dataclass
class StructuredCriticAgentState(BudgetState):
    # 内部类：节点构建（与 Proposal 一致）
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

    # 状态变量（移除 construction 和 construction_plan_text）
    messages: list[str] = field(default_factory=list)
    approved_user_goal: str = ""
    approved_files: list[str] = field(default_factory=list)
    nodes: list[node_construction] = field(default_factory=list)
    relationships: list[relationship_construction] = field(default_factory=list)
    feedback: str = ""
    isCriticComplete: bool = False

    _current_instance: "StructuredCriticAgentState | None" = None

    def __post_init__(self):
        StructuredCriticAgentState._current_instance = self

    def constructionToString(self) -> str:
        """Combine nodes and relationships into a single string."""
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
            print("Critic$ " + message)

    def add_agent(self, text: str) -> None:
        message = f"agent: {text}"
        self.messages.append(message)
        print("Critic$ " + message)

    def add_system(self, text: str) -> None:
        message = f"system: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print("Critic$ " + message)


# ---------- 节点 ----------
def import_node(state: StructuredCriticAgentState) -> StructuredCriticAgentState:
    # 删除旧反馈文件
    try:
        if os.path.exists(STRUCTURED_FEEDBACK_PATH):
            os.remove(STRUCTURED_FEEDBACK_PATH)
            print(f"🗑️ Deleted old feedback file '{STRUCTURED_FEEDBACK_PATH}'.")
    except Exception as e:
        print(f"⚠️ Error deleting old feedback file: {e}")

    # 加载用户目标
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

    # 加载文件列表
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

    # 加载节点构建计划 (CSV)
    try:
        with open(STRUCTURED_NODE_CONSTRUCTION_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 假设CSV格式：source_file,label,unique_column_name,properties
        for line in lines[1:]:  # 跳过表头
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

    # 加载关系构建计划 (CSV)
    try:
        with open(STRUCTURED_RELATION_CONSTRUCTION_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 假设CSV格式：source_file,relationship_type,from_node_label,from_node_column,to_node_label,to_node_column,properties
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


def save_node(state: StructuredCriticAgentState) -> StructuredCriticAgentState:
    # 保存反馈
    try:
        with open(STRUCTURED_FEEDBACK_PATH, "w", encoding="utf-8") as f:
            f.write(state.feedback)
        print(f"✅ Feedback saved to '{STRUCTURED_FEEDBACK_PATH}'.")
    except Exception as e:
        print(f"⚠️ Error saving feedback: {e}")

    state.isCriticComplete = False
    return state


def rest_node(state: StructuredCriticAgentState) -> StructuredCriticAgentState:
    state.add_system("You can talk here but you should call a tool.")
    return state


# ---------- 工具定义 ----------
@tool
def get_approved_user_goal() -> str:
    """Return the approved user goal from state."""
    state = StructuredCriticAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    return state.approved_user_goal


@tool
def get_approved_files() -> list[str]:
    """Return the list of approved file names from state."""
    state = StructuredCriticAgentState._current_instance
    if state is None:
        return ["Error: No state instance available."]
    return state.approved_files


@tool
def get_proposed_construction_plan() -> str:
    """Return the current construction plan (nodes + relationships) as a string."""
    state = StructuredCriticAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    return state.constructionToString()


@tool
def sample_file(file_names: list[str]) -> dict[str, str]:
    """Preview the first MAX_READ lines (default 100) of each specified file."""
    state = StructuredCriticAgentState._current_instance
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
def set_feedback(feedback: str) -> str:
    """Set the feedback string in the state. This feedback will be passed back to the structured proposal agent."""
    state = StructuredCriticAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    state.feedback = feedback
    return f"Feedback set to: '{feedback}'"


@tool
def critic_complete() -> str:
    """Mark the critic review as complete."""
    state = StructuredCriticAgentState._current_instance
    if state is None:
        return "Error: No state instance available."
    state.isCriticComplete = True
    return "Critic review marked as complete."


tools = [
    get_approved_user_goal,
    get_approved_files,
    get_proposed_construction_plan,
    sample_file,
    search_file,
    set_feedback,
    critic_complete,
]

tool_map = {t.name: t for t in tools}


model = LLM_CHAT_COMPLETION


def history_to_chat_messages(state: StructuredCriticAgentState) -> list[tuple[str, str]]:
    chat_messages = []
    for message in state.messages:
        if message.startswith("user: "):
            chat_messages.append(("user", message.removeprefix("user: ")))
        elif message.startswith("agent: "):
            chat_messages.append(("assistant", message.removeprefix("agent: ")))
        elif message.startswith("system: "):
            chat_messages.append(("system", message.removeprefix("system: ")))
    return chat_messages


def agent_node(state: StructuredCriticAgentState) -> StructuredCriticAgentState:
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
        "Your task is to criticize the proposed schema for relevance to the user goal and approved files.\n\n"
        "**Critique criteria:**\n"
        "- Are unique identifiers actually unique? Use the 'search_file' tool to validate. Composite identifiers are not acceptable.\n"
        "- Could any nodes be relationships instead? Double-check that unique identifiers are truly unique and not references to other nodes. Use 'search_file' to validate.\n"
        "- Can you manually trace through the source data to find the necessary information for answering a hypothetical question?\n"
        "- Is every node in the schema connected? What relationships could be missing? Every node should connect to at least one other node.\n"
        "- Are hierarchical container relationships missing?\n"
        "- Are any relationships redundant? A relationship between two nodes is redundant if it is semantically equivalent to or the inverse of another relationship between those two nodes.\n\n"
        "**Mandatory column and property matching criteria:**\n"
        "- Check actual file headers with 'sample_file' and validate identifier values with 'search_file'. Every node unique_column_name and properties entry must exactly match a column in that node's source_file, including case and suffixes such as '[主键]'.\n"
        "- For every relationship, do from_node_label and to_node_label each resolve to exactly one node construction? Treat missing or ambiguous node plans as errors.\n"
        "- Does from_node_column exist in the relationship's source_file as the foreign-key column pointing to the TARGET node? This field is not the source node's own matching property; a foreign key need not be imported as a source node property.\n"
        "- Does to_node_column exist in the target node's original file AND in that node plan's unique_column_name or properties? Reject columns present only in the relationship source file or omitted from the target node's imported properties. For example, '先修课程编号' cannot match a Course node that imports only '编号[主键]', '名称', '学分', and '学时'.\n"
        "- Does the relationship source file contain the source node plan's unique_column_name, and do its values actually identify the intended source nodes? Check both joins: source_node[unique_column_name] = row[unique_column_name], and target_node[to_node_column] = row[from_node_column]. A prerequisite record's own '编号[主键]' is not a Course identifier merely because the column names match.\n"
        "- Does every relationship properties entry exist in the relationship's source_file? Check all relationships, including self-references and association-file mappings; semantic plausibility does not override invalid property mappings.\n"
        "- Any missing column, unimported target property, or invalid identifier mapping blocks 'approve'. Feedback must identify the relationship, source file, affected label and field, the actual available columns/imported properties, and a correction supported by the inspected data. If the current importer cannot express the mapping, report that limitation rather than inventing columns.\n\n"
        "**Prepare for the task:**\n"
        "- Use 'get_approved_user_goal' to retrieve the user goal.\n"
        "- Use 'get_approved_files' to get the list of approved files.\n"
        "- Use 'get_proposed_construction_plan' to obtain the current construction plan.\n"
        "- Use 'sample_file' and 'search_file' to validate the schema design.\n\n"
        "**Workflow:**\n"
        "1. Analyze each construction rule in the proposed construction plan.\n"
        "2. Use tools to validate the construction rules for relevance and correctness.\n"
        "3. If the schema looks good, set feedback to 'approve'.\n"
        "4. If the schema has problems, provide feedback as a concise bullet list of problems (use 'set_feedback' tool to record the list).\n"
        "5. After setting feedback, call 'critic_complete' to finish the review.\n\n"
        "**Critical rules for tool usage:**\n"
        "- You can only call **one tool per response**. If you need multiple tools, call them in separate responses, one after another.\n"
        "- When you call a tool, your **entire response must consist solely** of the `[tool_call]` marker followed by the exact tool call, with no additional text before or after.\n"
        "- Do not add any text after the tool call – it must be the last thing in your response.\n"
        "- Use natural language for conversation (e.g., when you output 'valid' or 'retry' with feedback).\n\n"
        "Examples of valid tool calls:\n"
        "[tool_call] get_approved_user_goal()\n"
        "[tool_call] get_approved_files()\n"
        "[tool_call] get_proposed_construction_plan()\n"
        "[tool_call] sample_file(['customers.csv'])\n"
        "[tool_call] search_file(['customers.csv'], 'customer_id')\n"
        "[tool_call] set_feedback('- Node label missing.\\n- Duplicate relationship.')\n"
        "[tool_call] critic_complete()"
    )

    messages = [
        SystemMessage(content=system_prompt),
        *history_to_chat_messages(state),
    ]

    response = model.invoke(messages)
    agent_text = parse_agent_output(response.content)
    state.add_agent(agent_text)
    return state


def tool_node(state: StructuredCriticAgentState) -> StructuredCriticAgentState:
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


def agent_should_continue(state: StructuredCriticAgentState) -> Literal["tool", "rest"]:
    latest_message = state.messages[-1]
    if TOOL_CALL_MARKER in latest_message:
        return "tool"
    else:
        return "rest"


def tool_should_continue(state: StructuredCriticAgentState) -> Literal["save", "agent"]:
    if state.isCriticComplete:
        return "save"
    else:
        return "agent"


# ---------- 图构建 ----------
structuredCriticAgentGraph = build_agent_graph(
    globals(), "structuredCriticAgent", StructuredCriticAgentState,
    complete=lambda state: state.isCriticComplete,
    outputs=lambda state: {STRUCTURED_FEEDBACK_PATH: state.feedback},
    coordinator=False,
)


if __name__ == "__main__":
    state = StructuredCriticAgentState()
    state.add_user("Please review the construction plan.")
    final_state = structuredCriticAgentGraph.invoke(state)
