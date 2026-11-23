from dependencies import *
from agent_runtime import stage_node, managed_main_graph
from UserIntentAgent import UserIntentAgentGraph
from StructuredFileSuggestionAgent import StructuredFileSuggestionAgentGraph
from UnstructuredFileSuggestionAgent import UnstructuredFileSuggestionAgentGraph
from StructuredProposalCoordinator import StructuredProposalCoordinatorGraph
from unstructuredNodeConstruction import unstructuredNodeConstructionGraph
from unstructuredRelationConstruction import unstructuredRelationConstructionGraph
from StructuredGraphConstruction import StructuredGraphConstructionGraph
from UnstructuredGraphConstruction import UnstructuredGraphConstructionGraph
from MergeGraph import MergeGraphGraph
from ClearNodes import NodeCleanerGraph
from MergeSameAsNodes import PhysicalMergeGraph
from MergeRepeatRelationships import MergeRepeatRelationshipsGraph
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings


@dataclass
class AgentState:
    c: int = 0
    stages: dict[str, str] = field(default_factory=dict)

# File name, graph node name, and runnable.
# Structured construction runs before unstructured construction.
WORKFLOW = (
    ("UserIntentAgent", "UserIntentAgent", UserIntentAgentGraph),
    ("StructuredFileSuggestionAgent", "StructuredFileSuggestionAgent", StructuredFileSuggestionAgentGraph),
    ("UnstructuredFileSuggestionAgent", "UnstructuredFileSuggestionAgent", UnstructuredFileSuggestionAgentGraph),
    ("StructuredProposalCoordinator", "StructuredProposalCoordinator", StructuredProposalCoordinatorGraph),
    ("unstructuredNodeConstruction", "unstructuredNodeConstruction", unstructuredNodeConstructionGraph),
    ("unstructuredRelationConstruction", "unstructuredRelationConstruction", unstructuredRelationConstructionGraph),
    ("StructuredGraphConstruction", "StructuredGraphConstruction", StructuredGraphConstructionGraph),
    ("UnstructuredGraphConstruction", "UnstructuredGraphConstruction", UnstructuredGraphConstructionGraph),
    ("MergeGraph", "MergeGraph", MergeGraphGraph),
    ("MergeSameAsNodes", "PhysicalMerge", PhysicalMergeGraph),
    ("MergeRepeatRelationships", "MergeRepeatRelationships", MergeRepeatRelationshipsGraph),
    ("ClearNodes", "NodeCleaner", NodeCleanerGraph),
)


def resolve_start_file(value: str) -> str:
    name = value.strip().casefold()
    if not name:
        return WORKFLOW[0][0]
    if name.isdecimal():
        number = int(name)
        if 1 <= number <= len(WORKFLOW):
            return WORKFLOW[number - 1][0]
        raise ValueError(f"Choose a number from 1 to {len(WORKFLOW)}.")
    if name.endswith(".py"):
        name = name[:-3]
    for filename, _, _ in WORKFLOW:
        if filename.casefold() == name:
            return filename
    raise ValueError(f"Unknown selection: {value!r}. Choose a number or filename from the list.")


def build_main_graph(start_file: str = ""):
    filename = resolve_start_file(start_file)
    start_index = next(i for i, step in enumerate(WORKFLOW) if step[0] == filename)
    selected = WORKFLOW[start_index:]
    builder = StateGraph(AgentState)
    for _, name, runnable in selected:
        builder.add_node(name, stage_node(name, runnable))
    builder.set_entry_point(selected[0][1])
    for previous, following in zip(selected, selected[1:]):
        builder.add_edge(previous[1], following[1])
    builder.add_edge(selected[-1][1], END)
    return managed_main_graph(builder.compile())


def create_start_session(**kwargs):
    completer = WordCompleter(
        [filename + ".py" for filename, _, _ in WORKFLOW],
        ignore_case=True,
        sentence=True,
    )
    bindings = KeyBindings()

    @bindings.add("tab")
    def complete_next(event):
        if event.current_buffer.complete_state:
            event.current_buffer.complete_next()
        else:
            event.current_buffer.start_completion(select_first=True)

    @bindings.add("s-tab")
    def complete_previous(event):
        if event.current_buffer.complete_state:
            event.current_buffer.complete_previous()
        else:
            event.current_buffer.start_completion(select_last=True)

    return PromptSession(completer=completer, key_bindings=bindings,
                         complete_while_typing=False, **kwargs)


def select_start_file() -> str:
    print("Available starting files (in execution order):")
    for number, (filename, _, _) in enumerate(WORKFLOW, start=1):
        print(f"  {number:2d}. {filename}.py")
    print(f"Enter a number (1-{len(WORKFLOW)}) or filename. Press Enter to run from the beginning.")
    print("Existing files and database contents are used when starting later.")
    if sys.stdin.isatty() and sys.stdout.isatty():
        print("Type a filename prefix and press Tab to complete. Tab / Shift+Tab cycle through matches.")
        read_selection = create_start_session().prompt
    else:
        # Redirected input and non-terminal IDE consoles keep plain input support.
        read_selection = input
    while True:
        try:
            return resolve_start_file(read_selection("Starting number or filename: "))
        except ValueError as error:
            print(f"Invalid selection: {error}")


# Importing this module remains non-interactive for tests and programmatic use.
graph = build_main_graph()


if __name__ == "__main__":
    start_file = select_start_file()
    print(f"Starting workflow from {start_file}.py")
    final_state = build_main_graph(start_file).invoke(AgentState())
