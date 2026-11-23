from dependencies import *


@dataclass
class StructuredFileSuggestionAgentState(BudgetState):
    class UserGoal:
        def __init__(self, goal: str = "", approved: bool = False):
            self.goal = goal
            self.approved = approved

    messages: list[str] = field(default_factory=list)
    approved_user_goal: str = ""
    import_files: list[str] = field(default_factory=list)
    suggested_files: list[str] = field(default_factory=list)
    is_approved: bool = False

    _current_instance: StructuredFileSuggestionAgentState | None = None

    def __post_init__(self):
        StructuredFileSuggestionAgentState._current_instance = self

    def add_user(self, text: str) -> None:
        message = f"user: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print(message)

    def add_agent(self, text: str) -> None:
        message = f"agent: {text}"
        self.messages.append(message)
        print(message)

    def add_system(self, text: str) -> None:
        message = f"system: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print(message)


# ---------- 节点定义 ----------
def user_chat_node(state: StructuredFileSuggestionAgentState) -> StructuredFileSuggestionAgentState:
    agent_message = None
    for msg in reversed(state.messages):
        if msg.startswith("agent: "):
            agent_message = msg.removeprefix("agent: ").strip()
            break

    if agent_message:
        print("\n" + "=" * 50)
        print("🤖 Agent:", agent_message)
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("🤖 Agent: (no message)")
        print("=" * 50)

    user_input = "approve" if AUTO_APPROVE else input("👤 Your reply: ").strip()
    if user_input:
        state.add_user(user_input)
    else:
        state.add_user("(empty response)")
    return state


def save_node(state: StructuredFileSuggestionAgentState) -> StructuredFileSuggestionAgentState:
    with open(SAVE_STRC_SUGGESTED_PATH, "w", encoding="utf-8") as f:
        for fname in state.suggested_files:
            f.write(fname + "\n")
    print(f"\n✅ Suggested files saved to '{SAVE_STRC_SUGGESTED_PATH}'.")
    return state


def import_node(state: StructuredFileSuggestionAgentState) -> StructuredFileSuggestionAgentState:
    state.add_user("Hi, I need help selecting files for my knowledge graph.")
    try:
        with open(GOAL_SAVE_PATH, "r", encoding="utf-8") as f:
            goal = f.read().strip()
        state.approved_user_goal = goal
        print(f"✅ Loaded approved user goal: '{goal}'")
    except FileNotFoundError:
        state.approved_user_goal = ""
        print(f"⚠️ Goal file '{GOAL_SAVE_PATH}' not found. Starting with empty goal.")

    pattern = os.path.join(IMPORT_STRC_DIR, "*")
    file_paths = glob.glob(pattern)
    file_names = [os.path.basename(p) for p in file_paths if os.path.isfile(p)]
    state.import_files = file_names
    print(f"📁 Found {len(file_names)} import files: {file_names}")
    return state


# ---------- 工具定义 ----------
@tool
def get_approved_user_goal() -> str:
    """
    Retrieve the approved user goal stored in the agent's state.
    """
    state = StructuredFileSuggestionAgentState._current_instance
    if state is None:
        return "Error: Unable to get state instance."
    return state.approved_user_goal


@tool
def list_import_files() -> list[str]:
    """
    List all available import files.
    """
    state = StructuredFileSuggestionAgentState._current_instance
    if state is None:
        return ["Error: Unable to get state instance."]
    return state.import_files


@tool
def sample_file(file_names: list[str]) -> dict[str, str]:
    """
    Preview the first MAX_READ lines of each specified file.
    """
    state = StructuredFileSuggestionAgentState._current_instance
    if state is None:
        return {"error": "Unable to get state instance."}

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
def set_suggested_files(files: list[str]) -> str:
    """
    Record the agent's recommended file list.
    """
    state = StructuredFileSuggestionAgentState._current_instance
    if state is None:
        return "Error: Unable to get state instance."
    state.suggested_files = files
    return f"Suggested files set to: {files}"


@tool
def approve_suggested_files() -> str:
    """
    Mark the suggested files as approved.
    """
    state = StructuredFileSuggestionAgentState._current_instance
    if state is None:
        return "Error: Unable to get state instance."
    if not state.suggested_files:
        return "Error: No suggested files have been set. Please call set_suggested_files first."
    state.is_approved = True
    return f"Suggested files approved: {state.suggested_files}"


tools = [
    get_approved_user_goal,
    list_import_files,
    sample_file,
    set_suggested_files,
    approve_suggested_files,
]

tool_map = {t.name: t for t in tools}


model = LLM_CHAT_COMPLETION


def history_to_chat_messages(state: StructuredFileSuggestionAgentState) -> list[tuple[str, str]]:
    chat_messages = []
    for message in state.messages:
        if message.startswith("user: "):
            chat_messages.append(("user", message.removeprefix("user: ")))
        elif message.startswith("agent: "):
            chat_messages.append(("assistant", message.removeprefix("agent: ")))
        elif message.startswith("system: "):
            chat_messages.append(("system", message.removeprefix("system: ")))
    return chat_messages


def agent_node(state: StructuredFileSuggestionAgentState) -> StructuredFileSuggestionAgentState:
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
        "You are a constructive critic AI reviewing a list of files. "
        "Your goal is to help the user select appropriate files for constructing a knowledge graph.\n\n"
        "**Task:**\n"
        "Review the file list for relevance to the 'kind_of_graph' and 'description' derived from the approved user goal.\n"
        "For any file that you're not sure about, use the 'sample_file' tool to inspect its content.\n"
        "Only consider structured data files like CSV or JSON (files with tabular or structured data).\n\n"
        "**Tools at your disposal:**\n"
        "- get_approved_user_goal(): retrieves the approved user goal (kind and description).\n"
        "- list_import_files(): returns a list of all available file names.\n"
        "- sample_file(file_names: list[str]): previews the first 100 lines of each file.\n"
        "- set_suggested_files(files: list[str]): records your recommended file list in state.\n"
        "- approve_suggested_files(): finalizes the selection and marks it as approved.\n\n"
        "**Tool call rules (critical):**\n"
        "- You can only call **ONE tool per response**. If you need multiple tools, call them in separate responses, one after another.\n"
        "- When you need to call a tool, your **entire response must consist solely** of the `[tool_call]` marker followed by the exact tool call, with no additional text before or after.\n"
        "- Do not add any text after the tool call – it must be the last thing in your response.\n"
        "- Use natural language for conversation; you do not need a tool for chatting.\n\n"
        "Examples of valid tool calls:\n"
        "[tool_call] get_approved_user_goal()\n"
        "[tool_call] list_import_files()\n"
        "[tool_call] sample_file(['data1.csv', 'data2.json'])\n"
        "[tool_call] set_suggested_files(['data1.csv', 'data2.json'])\n"
        "[tool_call] approve_suggested_files()\n\n"
        "**Workflow – repeat until finished:**\n"
        "1. Use 'get_approved_user_goal' to understand the user's goal.\n"
        "2. Use 'list_import_files' to know what files are available.\n"
        "3. Evaluate each file's relevance. For uncertain ones, use 'sample_file' to examine content.\n"
        "4. Form a shortlist of relevant files and use 'set_suggested_files' to record it.\n"
        "5. Present the suggested files to the user naturally (just speak – no tool needed for chatting) and ask for approval.\n"
        "6. If the user gives feedback, go back to step 2 with that feedback in mind, adjust your suggestions, and repeat.\n"
        "7. Only when the user explicitly says 'approve' or 'yes' to your suggested list, call 'approve_suggested_files' to finalize.\n\n"
        "Remember: talk normally, use tools only for specific actions, and never add extra text around tool calls."
    )

    messages = [
        SystemMessage(content=system_prompt),
        *history_to_chat_messages(state),
    ]

    response = model.invoke(messages)
    agent_text = parse_agent_output(response.content)
    state.add_agent(agent_text)
    return state


def tool_node(state: StructuredFileSuggestionAgentState) -> StructuredFileSuggestionAgentState:
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


def should_continue(state: StructuredFileSuggestionAgentState) -> Literal["tool", "user_chat", "save"]:
    latest_message = state.messages[-1]
    if TOOL_CALL_MARKER in latest_message:
        return "tool"
    elif state.is_approved:
        return "save"
    else:
        return "user_chat"


# ---------- 图构建 ----------
StructuredFileSuggestionAgentGraph = build_agent_graph(
    globals(), "StructuredFileSuggestionAgent", StructuredFileSuggestionAgentState,
    complete=lambda state: state.is_approved,
    outputs=lambda state: {SAVE_STRC_SUGGESTED_PATH: "".join(name + "\n" for name in state.suggested_files)},
    coordinator=False,
)


if __name__ == "__main__":
    state = StructuredFileSuggestionAgentState()
    # 添加初始用户消息以启动对话（与 User Intent Agent 类似）

    state.add_agent("[tool_call] sampleCall()")
    final_state = StructuredFileSuggestionAgentGraph.invoke(state)
