from structuredProposalAgent import structuredProposalAgentGraph
from structuredCriticAgent import structuredCriticAgentGraph
from dependencies import *

@dataclass
class StructuredProposalCoordinatorState(BudgetState):
    msg: list[str] = field(default_factory=list)
    isApproved: bool = False
    isCallLoop: bool = False
    isConflict: bool = False
    iterationNum: int = 0  # 新增变量

    _current_instance: "StructuredProposalCoordinatorState | None" = None

    def __post_init__(self):
        StructuredProposalCoordinatorState._current_instance = self

    def add_user(self, text: str) -> None:
        message = f"user: {text}"
        self.msg.append(message)
        if DEBUG_MODE:
            print(message)

    def add_agent(self, text: str) -> None:
        message = f"agent: {text}"
        self.msg.append(message)
        print(message)

    def add_system(self, text: str) -> None:
        message = f"system: {text}"
        self.msg.append(message)
        if DEBUG_MODE:
            print(message)


# ---------- 工具定义 ----------
@tool
def get_feedback() -> str:
    """Read the feedback from the local file at STRUCTURED_FEEDBACK_PATH and return its content."""
    state = StructuredProposalCoordinatorState._current_instance
    if state is None:
        return "Error: No state instance available."

    try:
        with open(STRUCTURED_FEEDBACK_PATH, "r", encoding="utf-8") as f:
            feedback = f.read()
        return feedback
    except FileNotFoundError:
        return "No feedback file found."
    except Exception as e:
        return f"Error reading feedback: {e}"


@tool
def approve_construction() -> str:
    """Set isApproved to True, indicating the construction plan is approved."""
    state = StructuredProposalCoordinatorState._current_instance
    if state is None:
        return "Error: No state instance available."
    state.isApproved = True
    return "Construction plan approved."


@tool
def callLoop() -> str:
    """Set isCallLoop to True, triggering the refinement loop."""
    state = StructuredProposalCoordinatorState._current_instance
    if state is None:
        return "Error: No state instance available."
    state.isCallLoop = True
    return "Refinement loop triggered."


@tool
def conflict(message: str) -> str:
    """Send a message to the user and set isConflict to True."""
    state = StructuredProposalCoordinatorState._current_instance
    if state is None:
        return "Error: No state instance available."
    print(f"\n🔴 Agent says to user: {message}")
    state.isConflict = True
    state.iterationNum = 0  # 重置迭代计数
    return "Conflict flag set, waiting for user input."


tools = [
    get_feedback,
    approve_construction,
    callLoop,
    conflict,
]

tool_map = {t.name: t for t in tools}

model = LLM_CHAT_COMPLETION


def history_to_chat_messages(state: StructuredProposalCoordinatorState) -> list[tuple[str, str]]:
    chat_messages = []
    for message in state.msg:
        if message.startswith("user: "):
            chat_messages.append(("user", message.removeprefix("user: ")))
        elif message.startswith("agent: "):
            chat_messages.append(("assistant", message.removeprefix("agent: ")))
        elif message.startswith("system: "):
            chat_messages.append(("system", message.removeprefix("system: ")))
    return chat_messages


# ---------- 节点定义 ----------
def agent_node(state: StructuredProposalCoordinatorState) -> StructuredProposalCoordinatorState:
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

    # 协调器系统提示（包含迭代计数说明）
    system_prompt = (
        "You are the Structured Proposal Coordinator. "
        "Your role is to manage the refinement loop between the Structured Proposal Agent and the Structured Critic Agent, "
        "and ultimately produce an approved construction plan.\n\n"

        "**Available tools:**\n"
        "- get_feedback(): Retrieves the latest feedback from the critic (reads STRUCTURED_FEEDBACK_PATH).\n"
        "- callLoop(): Triggers one iteration of the refinement loop (Proposal → Critic).\n"
        "- approve_construction(): Marks the current plan as approved (sets isApproved=True).\n"
        "- conflict(message: str): Sends a message to the user and pauses for their input (sets isConflict=True).\n\n"

        "**Workflow – follow these rules strictly:**\n"
        "1. Always start by calling 'get_feedback()' to check the current status of the critic.\n"
        "   - If the file does not exist, or the feedback is 'valid' (meaning the plan is accepted), "
        "   call 'callLoop()' to begin a new proposal-critic iteration.\n"
        "   - If the feedback is empty or the feedback explicitly confirms the plan (e.g., 'approved' or 'ok'), call 'approve_construction()' and finish.\n"
        "2. You can track the number of iterations using the state variable 'iterationNum'. Each time the loop ends (after critic), the iteration count increases by 1.\n"
        f"3. If you have already performed **{MAX_ITERATION} complete iterations** (i.e., iterationNum >= {MAX_ITERATION}) and the feedback still does not confirm the plan, call 'conflict(\"The schema has been revised twice but is not yet approved. Please decide.\")' to escalate to the user.\n"
        "4. When the user responds to the conflict message:\n"
        "   - If the user says 'approve' or indicates agreement, call 'approve_construction()'.\n"
        "   - If the user provides new ideas or changes the goal, call 'conflict' again (which resets iterationNum to 0) and then restart the loop.\n"
        "5. Always respond with one tool call at a time. After a tool returns, you will see its result, and you can decide the next step.\n\n"

        "**Critical rules for tool usage:**\n"
        "- You can only call **one tool per response**.\n"
        "- When you call a tool, your **entire response must consist solely** of the `[tool_call]` marker followed by the exact tool call, "
        "with no additional text before or after.\n"
        "- Do not add any text after the tool call – it must be the last thing in your response.\n"
        "- Use natural language only when you need to output a final answer or when the user interacts with you directly "
        "(e.g., when you use 'conflict' tool, the user will reply, and you will receive that as a user message).\n\n"

        "Examples of valid tool calls:\n"
        "[tool_call] get_feedback()\n"
        "[tool_call] callLoop()\n"
        "[tool_call] approve_construction()\n"
        "[tool_call] conflict('We need your input on the current schema.')"
    )

    messages = [
        SystemMessage(content=system_prompt),
        *history_to_chat_messages(state),
    ]

    response = model.invoke(messages)
    agent_text = parse_agent_output(response.content)
    state.add_agent(agent_text)
    return state


def tool_node(state: StructuredProposalCoordinatorState) -> StructuredProposalCoordinatorState:
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

    latest_message = state.msg[-1]
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


def rest_node(state: StructuredProposalCoordinatorState) -> StructuredProposalCoordinatorState:
    state.add_system("You can talk here but you should call a tool.")
    return state


def user_chat_node(state: StructuredProposalCoordinatorState) -> StructuredProposalCoordinatorState:
    # 删除旧的用户反馈文件
    try:
        if os.path.exists(STRUCTURED_USER_FEEDBACK_PATH):
            os.remove(STRUCTURED_USER_FEEDBACK_PATH)
            print(f"🗑️ Deleted old user feedback file '{STRUCTURED_USER_FEEDBACK_PATH}'.")
    except Exception as e:
        print(f"⚠️ Error deleting user feedback file: {e}")

    user_input = "approve" if AUTO_APPROVE else input("👤 Your reply: ").strip()
    if user_input:
        state.add_user(user_input)
        # 保存用户反馈到文件（覆盖模式）
        try:
            with open(STRUCTURED_USER_FEEDBACK_PATH, "w", encoding="utf-8") as f:
                f.write(user_input)
            print(f"✅ User feedback saved to '{STRUCTURED_USER_FEEDBACK_PATH}'.")
        except Exception as e:
            print(f"⚠️ Error saving user feedback: {e}")
    else:
        state.add_user("(empty response)")
    return state


def save_node(state: StructuredProposalCoordinatorState) -> StructuredProposalCoordinatorState:
    print("✅ Saving final construction plan...")
    state.isApproved = False
    state.isCallLoop = False
    state.isConflict = False
    state.iterationNum = 0  # 重置迭代计数
    return state


def loopEnd_node(state: StructuredProposalCoordinatorState) -> StructuredProposalCoordinatorState:
    state.isCallLoop = False
    state.iterationNum += 1
    state.add_system(f"Iteration count updated: {state.iterationNum}")
    return state


def chatEnd_node(state: StructuredProposalCoordinatorState) -> StructuredProposalCoordinatorState:
    state.isConflict = False
    return state


# ---------- 路由函数 ----------
def agent_should_continue(state: StructuredProposalCoordinatorState) -> Literal["tool", "rest"]:
    latest_message = state.msg[-1]
    if TOOL_CALL_MARKER in latest_message:
        return "tool"
    return "rest"


def tool_should_continue(state: StructuredProposalCoordinatorState) -> Literal[
    "structuredProposalAgentGraph", "save", "user_chat", "agent"
]:
    if state.isCallLoop:
        return "structuredProposalAgentGraph"
    elif state.isApproved:
        return "save"
    elif state.isConflict:
        return "user_chat"
    else:
        return "agent"


# ---------- 图构建 ----------
StructuredProposalCoordinatorGraph = build_agent_graph(
    globals(), "StructuredProposalCoordinator", StructuredProposalCoordinatorState,
    complete=lambda state: state.isApproved,
    outputs=lambda state: {},
    coordinator=True,
)


if __name__ == "__main__":
    state = StructuredProposalCoordinatorState()
    state.add_user("I want to create a construction plan.")
    state.add_agent("[tool_call] sampleCall()")
    final_state = StructuredProposalCoordinatorGraph.invoke(state)
