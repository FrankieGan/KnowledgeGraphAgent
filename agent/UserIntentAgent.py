from dependencies import *


@dataclass
class UserIntentAgentState(BudgetState):
    class UserGoal:
        def __init__(self, goal: str = "", approved: bool = False):
            self.goal = goal
            self.approved = approved

    messages: list[str] = field(default_factory=list)
    user_goal: UserGoal = field(default_factory=UserGoal)

    _current_instance: UserIntentAgentState | None = None

    def __post_init__(self):
        UserIntentAgentState._current_instance = self

    def _set_goal(self, goal: str) -> None:
        self.user_goal.goal = goal
        self.user_goal.approved = False

    def _set_approved(self, approved: bool = True) -> None:
        self.user_goal.approved = approved

    def _is_goal_empty(self) -> bool:
        return not bool(self.user_goal.goal.strip())

    def add_user(self, text: str) -> None:
        message = f"user: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print(message)   # debug output

    def add_agent(self, text: str) -> None:
        message = f"agent: {text}"
        self.messages.append(message)
        print(message)   # debug output

    def add_system(self, text: str) -> None:
        message = f"system: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print(message)   # debug output


# ---------- Tool definitions ----------
@tool
def set_perceived_user_goal(goal: str) -> str:
    """
    Record the agent's current perception of the user's goal, and reset the approval status to unapproved.
    Parameters:
        goal: A textual description of the user's goal (should include kind_of_graph and description).
    Returns:
        Feedback message indicating that the goal has been recorded and needs user confirmation.
    """
    state = UserIntentAgentState._current_instance
    if state is None:
        return "Error: Unable to get state instance."
    state._set_goal(goal)
    return f"Recorded user goal: '{goal}'. Please confirm or use approve_perceived_user_goal to approve."


@tool
def approve_perceived_user_goal() -> str:
    """
    Approve the currently recorded user goal. If the goal is empty, returns an error message.
    Parameters:
        None.
    Returns:
        Confirmation message with approval status if successful; otherwise an error message.
    """
    state = UserIntentAgentState._current_instance
    if state is None:
        return "Error: Unable to get state instance."
    if state._is_goal_empty():
        return "Error: No goal has been set. Please use set_perceived_user_goal first."
    state._set_approved(True)
    return f"Approved user goal: '{state.user_goal.goal}'."


tools = [
    set_perceived_user_goal,
    approve_perceived_user_goal,
]

tool_map = {t.name: t for t in tools}


model = LLM_CHAT_COMPLETION


def history_to_chat_messages(state: UserIntentAgentState) -> list[tuple[str, str]]:
    chat_messages = []
    for message in state.messages:
        if message.startswith("user: "):
            chat_messages.append(("user", message.removeprefix("user: ")))
        elif message.startswith("agent: "):
            chat_messages.append(("assistant", message.removeprefix("agent: ")))
        elif message.startswith("system: "):
            chat_messages.append(("system", message.removeprefix("system: ")))
    return chat_messages


def agent_node(state: UserIntentAgentState) -> UserIntentAgentState:
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
        "You are an expert at knowledge graph use cases.\n"
        "Your primary goal is to help the user come up with a knowledge graph use case.\n\n"
        "If the user is unsure what to do, make some suggestions based on classic use cases like:\n"
        "- social network involving friends, family, or professional relationships\n"
        "- logistics network with suppliers, customers, and partners\n"
        "- recommendation system with customers, products, and purchase patterns\n"
        "- fraud detection over multiple accounts with suspicious patterns of transactions\n"
        "- pop-culture graphs with movies, books, or music\n\n"
        "A user goal has two components:\n"
        "- kind_of_graph: at most 3 words describing the graph, for example 'social network' or 'USA freight logistics'\n"
        "- description: a few sentences about the intention of the graph, for example 'A dynamic routing and delivery system for cargo.' or 'Analysis of product dependencies and supplier alternatives.'\n\n"
        "Think carefully and collaborate with the user:\n"
        "1. Understand the user's goal, which is a kind_of_graph with description\n"
        "2. Ask clarifying questions as needed – you can just speak naturally, no special tool needed for conversation.\n"
        "3. When you think you understand their goal, use the 'set_perceived_user_goal' tool to record your perception. Feel free to call this tool multiple times to refine the goal based on user feedback.\n"
        "4. Present the perceived user goal to the user for confirmation (just say it naturally).\n"
        "5. Only when both you and the user are satisfied with the goal, use the 'approve_perceived_user_goal' tool to finalize it. Do not approve prematurely; iterate as needed.\n\n"
        "Important rules for tool usage:\n"
        "- You do NOT need to call any tool for chatting. Just output your message naturally; the system will show it to the user and wait for their reply.\n"
        "- Only use the two tools when you need to record or approve the goal.\n"
        "- When you need to call a tool, your **entire response must consist solely** of the `[tool_call]` marker and the tool call itself. Do not include any additional text before or after it. \n"
        "- If you do not need to call a tool, you may respond with natural language (this is your 'talking' mode).\n\n"
        "Tool call format (use the exact marker, with nothing else):\n"
        "[tool_call] set_perceived_user_goal('kind_of_graph: social network, description: map friendships')\n"
        "[tool_call] approve_perceived_user_goal()\n\n"
        "Decide clearly: if you need a tool, output only the tool call; otherwise, output natural language."
    )

    messages = [
        SystemMessage(content=system_prompt),
        *history_to_chat_messages(state),
    ]

    response = model.invoke(messages)
    agent_text = parse_agent_output(response.content)
    state.add_agent(agent_text)
    return state


def tool_node(state: UserIntentAgentState) -> UserIntentAgentState:
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
        state.add_system("Invalid tool call format.")
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


def user_chat_node(state: UserIntentAgentState) -> UserIntentAgentState:
    """
    Node that displays the agent's last message (if any) to the user and waits for input.
    It then adds the user's response to the conversation history.
    """
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

    user_input = input("👤 Your reply: ").strip()
    if user_input:
        state.add_user(user_input)
    else:
        state.add_user("(empty response)")

    return state


# ---------- 新增节点：保存目标到本地（无检查，因为条件边已保证批准） ----------
def save_goal_node(state: UserIntentAgentState) -> UserIntentAgentState:
    """
    Saves the approved user goal to a local file (overwrite mode).
    """
    with open(GOAL_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(state.user_goal.goal)
    print(f"\n✅ Goal saved to '{GOAL_SAVE_PATH}'.")
    return state


def should_continue(state: UserIntentAgentState) -> Literal["tool", "user_chat", "save_goal"]:
    latest_message = state.messages[-1]
    if TOOL_CALL_MARKER in latest_message:
        return "tool"
    elif state.user_goal.approved:
        return "save_goal"
    else:
        return "user_chat"


# Build the graph
UserIntentAgentGraph = build_agent_graph(
    globals(), "UserIntentAgent", UserIntentAgentState,
    complete=lambda state: state.user_goal.approved,
    outputs=lambda state: {GOAL_SAVE_PATH: state.user_goal.goal},
    coordinator=False,
)


if __name__ == "__main__":
    state = UserIntentAgentState()
    state.add_user("Hello, I want to build a knowledge graph.")

    final_state = UserIntentAgentGraph.invoke(state)