from __future__ import annotations
from dependencies import *


# ============================================================
# sample_file 配置
# ============================================================

SUPPORTED_SAMPLE_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
}

SAMPLE_MAX_LINES = int(
    globals().get("MAX_READ", 100)
)

SAMPLE_MAX_CHARACTERS = 12000
SAMPLE_MAX_PDF_PAGES = 5
MIN_PDF_NATIVE_CHARACTERS = 40

SAMPLE_OCR_LANGUAGES = "chi_sim+eng"
SAMPLE_OCR_DPI = 250


# ============================================================
# 当前状态引用
# ============================================================

# 不要将该变量放入 dataclass。
_CURRENT_UNSTRUCTURED_RELATION_STATE: Any = None


# ============================================================
# 状态定义
# ============================================================

@dataclass
class UnstructuredRelationConstructionState(BudgetState):
    class UnstructuredRelation:
        def __init__(
            self,
            subj: str = "",
            predicate: str = "",
            obj: str = ""
        ):
            self.subj = subj
            self.predicate = predicate
            self.obj = obj

        def __str__(self) -> str:
            return (
                f"({self.subj}, "
                f"{self.predicate}, "
                f"{self.obj})"
            )

        def to_csv_line(self) -> str:
            return (
                f"{self.subj},"
                f"{self.predicate},"
                f"{self.obj}"
            )

    messages: list[str] = field(
        default_factory=list
    )

    approved_user_goal: str = ""

    approved_files: list[str] = field(
        default_factory=list
    )

    approved_unstructured_nodes: list[str] = field(
        default_factory=list
    )

    proposed_unstructured_relations: list[UnstructuredRelation] = field(
        default_factory=list
    )

    isApproved: bool = False

    def __post_init__(self):
        global _CURRENT_UNSTRUCTURED_RELATION_STATE
        _CURRENT_UNSTRUCTURED_RELATION_STATE = self

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


def activate_state(
    state: UnstructuredRelationConstructionState
) -> None:
    global _CURRENT_UNSTRUCTURED_RELATION_STATE
    _CURRENT_UNSTRUCTURED_RELATION_STATE = state


def get_current_unstructured_relation_state(
) -> UnstructuredRelationConstructionState | None:
    return _CURRENT_UNSTRUCTURED_RELATION_STATE


# ============================================================
# 通用辅助函数
# ============================================================

def unique_preserving_order(
    values: list[str]
) -> list[str]:
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def normalize_preview_text(
    text: str
) -> str:
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def useful_character_count(
    text: str
) -> int:
    if not text:
        return 0

    return sum(
        1
        for character in text
        if character.isalnum()
    )


def native_pdf_text_is_usable(
    text: str
) -> bool:
    text = normalize_preview_text(text)

    if not text:
        return False

    if (
        useful_character_count(text)
        < MIN_PDF_NATIVE_CHARACTERS
    ):
        return False

    printable_count = sum(
        1
        for character in text
        if (
            character.isprintable()
            or character in "\n\t"
        )
    )

    printable_ratio = (
        printable_count / len(text)
        if text
        else 0
    )

    return printable_ratio >= 0.85


def truncate_preview(
    text: str,
    max_lines: int = SAMPLE_MAX_LINES,
    max_characters: int = SAMPLE_MAX_CHARACTERS
) -> str:
    text = normalize_preview_text(text)

    if not text:
        return "(no readable text)"

    lines = text.splitlines()
    selected_lines = lines[:max_lines]

    preview = "\n".join(
        selected_lines
    ).strip()

    truncated_by_lines = (
        len(lines) > max_lines
    )

    truncated_by_characters = (
        len(preview) > max_characters
    )

    if truncated_by_characters:
        preview = (
            preview[:max_characters]
            .rstrip()
        )

    if (
        truncated_by_lines
        or truncated_by_characters
    ):
        preview += "\n... (truncated)"

    return preview


# ============================================================
# 文件路径安全校验
# ============================================================

def resolve_sample_file_path(
    file_name: str,
    state: UnstructuredRelationConstructionState
) -> Path:
    """
    只允许读取 approved_files 中且位于 IMPORT_UNSTRC_DIR
    内部的文件。
    """
    if not file_name:
        raise ValueError(
            "File name cannot be empty."
        )

    if file_name not in state.approved_files:
        raise ValueError(
            f"File '{file_name}' is not in "
            f"the approved file list."
        )

    import_root = Path(
        IMPORT_UNSTRC_DIR
    ).resolve()

    requested_path = Path(file_name)

    if requested_path.is_absolute():
        resolved_path = (
            requested_path.resolve()
        )
    else:
        resolved_path = (
            import_root / requested_path
        ).resolve()

    try:
        common_path = Path(
            os.path.commonpath(
                [
                    str(import_root),
                    str(resolved_path),
                ]
            )
        )

    except ValueError as e:
        raise ValueError(
            f"Invalid file path "
            f"'{file_name}': {e}"
        ) from e

    if common_path != import_root:
        raise ValueError(
            f"File path escapes import "
            f"directory: {file_name}"
        )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"File '{file_name}' not found."
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Path '{file_name}' is not a file."
        )

    suffix = resolved_path.suffix.lower()

    if suffix not in SUPPORTED_SAMPLE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: "
            f"{sorted(SUPPORTED_SAMPLE_EXTENSIONS)}"
        )

    return resolved_path


# ============================================================
# TXT / Markdown 预览
# ============================================================

def sample_text_file(
    file_path: Path
) -> str:
    try:
        file_bytes = file_path.read_bytes()

    except Exception as e:
        raise ValueError(
            f"Unable to read file bytes: {e}"
        ) from e

    decode_errors = []

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
    ):
        try:
            text = file_bytes.decode(
                encoding
            )

            return (
                f"[File type: "
                f"{file_path.suffix.lower()}]\n"
                f"[Encoding: {encoding}]\n\n"
                f"{truncate_preview(text)}"
            )

        except UnicodeDecodeError as e:
            decode_errors.append(
                f"{encoding}: {e}"
            )

    raise ValueError(
        "Unable to decode text file. "
        f"Errors: {decode_errors}"
    )


# ============================================================
# PDF / OCR 预览
# ============================================================

def render_pdf_page(
    page: fitz.Page
) -> Image.Image:
    zoom = SAMPLE_OCR_DPI / 72.0

    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image_bytes = pixmap.tobytes(
        "png"
    )

    with Image.open(
        io.BytesIO(image_bytes)
    ) as opened_image:
        image = opened_image.convert(
            "RGB"
        )

    return image


def run_page_ocr(
    page: fitz.Page
) -> str:
    image = render_pdf_page(page)

    try:
        grayscale = image.convert("L")

        enhanced = ImageOps.autocontrast(
            grayscale
        )

        text = pytesseract.image_to_string(
            enhanced,
            lang=SAMPLE_OCR_LANGUAGES,
            config="--oem 3 --psm 3"
        )

        return normalize_preview_text(
            text
        )

    finally:
        image.close()


def sample_pdf_file(
    file_path: Path
) -> str:
    try:
        document = fitz.open(
            str(file_path)
        )

    except Exception as e:
        raise ValueError(
            f"Unable to open PDF: {e}"
        ) from e

    try:
        if document.needs_pass:
            raise ValueError(
                "PDF is encrypted and requires "
                "a password."
            )

        if document.page_count == 0:
            return (
                "[File type: PDF]\n"
                "(PDF contains no pages)"
            )

        metadata = document.metadata or {}

        title = str(
            metadata.get("title") or ""
        ).strip()

        pages_to_read = min(
            document.page_count,
            SAMPLE_MAX_PDF_PAGES
        )

        page_previews = []
        native_pages = []
        ocr_pages = []
        ocr_failed_pages = []

        for page_index in range(
            pages_to_read
        ):
            page = document.load_page(
                page_index
            )

            page_number = page_index + 1

            native_text = (
                normalize_preview_text(
                    page.get_text("text")
                )
            )

            selected_text = native_text
            method = "native text"

            if native_pdf_text_is_usable(
                native_text
            ):
                native_pages.append(
                    page_number
                )

            else:
                try:
                    ocr_text = run_page_ocr(
                        page
                    )

                    if ocr_text:
                        selected_text = ocr_text
                        method = "OCR"

                        ocr_pages.append(
                            page_number
                        )

                    else:
                        method = (
                            "OCR returned empty"
                        )

                        ocr_failed_pages.append(
                            page_number
                        )

                except Exception as e:
                    method = (
                        f"OCR failed: {e}"
                    )

                    ocr_failed_pages.append(
                        page_number
                    )

            if not selected_text:
                selected_text = (
                    "(no readable text "
                    "on this page)"
                )

            page_previews.append(
                f"--- Page {page_number} "
                f"[{method}] ---\n"
                f"{selected_text}"
            )

        combined_text = "\n\n".join(
            page_previews
        )

        preview = truncate_preview(
            combined_text
        )

        header_lines = [
            "[File type: PDF]",
            (
                f"[Total pages: "
                f"{document.page_count}]"
            ),
            (
                f"[Previewed pages: "
                f"{pages_to_read}]"
            ),
        ]

        if title:
            header_lines.append(
                f"[Title: {title}]"
            )

        if native_pages:
            header_lines.append(
                f"[Native-text pages: "
                f"{native_pages}]"
            )

        if ocr_pages:
            header_lines.append(
                f"[OCR pages: {ocr_pages}]"
            )

        if ocr_failed_pages:
            header_lines.append(
                f"[OCR failed/empty pages: "
                f"{ocr_failed_pages}]"
            )

        if document.page_count > pages_to_read:
            header_lines.append(
                "[Additional PDF pages "
                "were not previewed]"
            )

        return (
            "\n".join(header_lines)
            + "\n\n"
            + preview
        )

    finally:
        document.close()


def read_file_preview(
    file_path: Path
) -> str:
    suffix = file_path.suffix.lower()

    if suffix in {
        ".txt",
        ".md",
        ".markdown",
    }:
        return sample_text_file(
            file_path
        )

    if suffix == ".pdf":
        return sample_pdf_file(
            file_path
        )

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )


# ============================================================
# import_node
# ============================================================

def import_node(
    state: UnstructuredRelationConstructionState
) -> UnstructuredRelationConstructionState:
    activate_state(state)

    state.isApproved = False

    # --------------------------------------------------------
    # 加载用户目标
    # --------------------------------------------------------

    try:
        with open(
            GOAL_SAVE_PATH,
            "r",
            encoding="utf-8-sig"
        ) as f:
            goal = f.read().strip()

        state.approved_user_goal = goal

        state.add_system(
            f"✅ Loaded approved user goal: "
            f"'{goal}'"
        )

    except FileNotFoundError:
        state.approved_user_goal = ""

        state.add_system(
            f"⚠️ Goal file "
            f"'{GOAL_SAVE_PATH}' not found. "
            f"Starting with empty goal."
        )

    except Exception as e:
        state.approved_user_goal = ""

        state.add_system(
            f"⚠️ Error reading goal file: {e}"
        )

    # --------------------------------------------------------
    # 加载已批准的文件
    # --------------------------------------------------------

    try:
        with open(
            SAVE_UNSTRC_SUGGESTED_PATH,
            "r",
            encoding="utf-8-sig"
        ) as f:
            files = [
                line.strip()
                for line in f
                if line.strip()
            ]

        files = unique_preserving_order(
            files
        )

        supported_files = []
        unsupported_files = []

        for file_name in files:
            suffix = Path(
                file_name
            ).suffix.lower()

            if (
                suffix
                in SUPPORTED_SAMPLE_EXTENSIONS
            ):
                supported_files.append(
                    file_name
                )
            else:
                unsupported_files.append(
                    file_name
                )

        state.approved_files = (
            supported_files
        )

        state.add_system(
            f"✅ Loaded "
            f"{len(supported_files)} "
            f"approved supported files: "
            f"{supported_files}"
        )

        if unsupported_files:
            state.add_system(
                f"⚠️ Ignored unsupported "
                f"approved files: "
                f"{unsupported_files}"
            )

    except FileNotFoundError:
        state.approved_files = []

        state.add_system(
            f"⚠️ Suggested files file "
            f"'{SAVE_UNSTRC_SUGGESTED_PATH}' "
            f"not found. Starting with "
            f"empty list."
        )

    except Exception as e:
        state.approved_files = []

        state.add_system(
            f"⚠️ Error reading suggested "
            f"files: {e}"
        )

    # --------------------------------------------------------
    # 加载已批准的 unstructured_node 类型
    # --------------------------------------------------------

    try:
        with open(
            UNSTRUCTURED_NODE_CONSTRUCTION_PATH,
            "r",
            encoding="utf-8-sig"
        ) as f:
            unstructured_nodes = [
                line.strip()
                for line in f
                if line.strip()
            ]

        state.approved_unstructured_nodes = (
            unique_preserving_order(
                unstructured_nodes
            )
        )

        state.add_system(
            f"✅ Loaded "
            f"{len(state.approved_unstructured_nodes)} "
            f"approved unstructured_nodes: "
            f"{state.approved_unstructured_nodes}"
        )

    except FileNotFoundError:
        state.approved_unstructured_nodes = []

        state.add_system(
            f"⚠️ unstructured_node list file "
            f"'{UNSTRUCTURED_NODE_CONSTRUCTION_PATH}' not found. "
            f"Starting with empty list."
        )

    except Exception as e:
        state.approved_unstructured_nodes = []

        state.add_system(
            f"⚠️ Error reading unstructured_node "
            f"list: {e}"
        )

    return state


# ============================================================
# user_chat_node
# ============================================================

def user_chat_node(
    state: UnstructuredRelationConstructionState
) -> UnstructuredRelationConstructionState:
    activate_state(state)

    agent_message = None

    for message in reversed(
        state.messages
    ):
        if message.startswith("agent: "):
            agent_message = (
                message
                .removeprefix("agent: ")
                .strip()
            )
            break

    print("\n" + "=" * 50)

    if agent_message:
        print(
            "🤖 Agent:",
            agent_message
        )
    else:
        print(
            "🤖 Agent: (no message)"
        )

    print("=" * 50)

    user_input = "approve" if AUTO_APPROVE else input("👤 Your reply: ").strip()

    if user_input:
        state.add_user(user_input)
    else:
        state.add_user(
            "(empty response)"
        )

    return state


# ============================================================
# save_node
# ============================================================

def save_node(
    state: UnstructuredRelationConstructionState
) -> UnstructuredRelationConstructionState:
    activate_state(state)

    try:
        output_path = Path(
            UNSTRUCTURED_RELATION_CONSTRUCTION_PATH
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "subject",
                "predicate",
                "object",
            ])

            for unstructured_relation in state.proposed_unstructured_relations:
                writer.writerow([
                    unstructured_relation.subj,
                    unstructured_relation.predicate,
                    unstructured_relation.obj,
                ])

        state.add_system(
            f"✅ unstructured_relations saved to "
            f"'{UNSTRUCTURED_RELATION_CONSTRUCTION_PATH}'."
        )

    except Exception as e:
        state.add_system(
            f"⚠️ Error saving unstructured_relations: {e}"
        )

    state.isApproved = False

    return state


# ============================================================
# 工具定义
# ============================================================

@tool
def get_approved_user_goal() -> str:
    """
    Return the approved user goal.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return (
            "Error: No state instance available."
        )

    return state.approved_user_goal


@tool
def get_approved_files() -> list[str]:
    """
    Return approved TXT, Markdown, and PDF file names.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return [
            "Error: No state instance available."
        ]

    return state.approved_files


@tool
def get_approved_unstructured_node() -> list[str]:
    """
    Return approved unstructured_node types.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return [
            "Error: No state instance available."
        ]

    return state.approved_unstructured_nodes


@tool
def sample_file(
    file_names: list[str]
) -> dict[str, str]:
    """
    Preview approved TXT, Markdown, and PDF documents.

    PDF files may contain native text, scanned images, or both.
    Scanned pages are processed with Tesseract OCR.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return {
            "error": (
                "No state instance available."
            )
        }

    if not file_names:
        return {
            "error": (
                "No file names were provided."
            )
        }

    result: dict[str, str] = {}

    unique_file_names = (
        unique_preserving_order(
            file_names
        )
    )

    for file_name in unique_file_names:
        try:
            file_path = (
                resolve_sample_file_path(
                    file_name=file_name,
                    state=state
                )
            )

            result[file_name] = (
                read_file_preview(
                    file_path
                )
            )

        except FileNotFoundError as e:
            result[file_name] = (
                f"Error: {e}"
            )

        except Exception as e:
            result[file_name] = (
                f"Error reading file: {e}"
            )

    return result


@tool
def add_proposed_unstructured_relation(
    subj: str,
    predicate: str,
    obj: str
) -> str:
    """
    Add a proposed unstructured_relation type.

    Subject and object must be approved unstructured_node types.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return (
            "Error: No state instance available."
        )

    if not isinstance(subj, str):
        return (
            "Error: Subject must be a string."
        )

    if not isinstance(predicate, str):
        return (
            "Error: Predicate must be a string."
        )

    if not isinstance(obj, str):
        return (
            "Error: Object must be a string."
        )

    subj = subj.strip()
    predicate = predicate.strip()
    obj = obj.strip()

    if not subj:
        return (
            "Error: Subject cannot be empty."
        )

    if not predicate:
        return (
            "Error: Predicate cannot be empty."
        )

    if not obj:
        return (
            "Error: Object cannot be empty."
        )

    if subj not in state.approved_unstructured_nodes:
        return (
            f"Error: Subject '{subj}' is not "
            f"in the approved unstructured_nodes list."
        )

    if obj not in state.approved_unstructured_nodes:
        return (
            f"Error: Object '{obj}' is not "
            f"in the approved unstructured_nodes list."
        )

    duplicate_exists = any(
        unstructured_relation.subj == subj
        and unstructured_relation.predicate == predicate
        and unstructured_relation.obj == obj
        for unstructured_relation in state.proposed_unstructured_relations
    )

    if duplicate_exists:
        return (
            f"unstructured_relation already exists: "
            f"({subj}, {predicate}, {obj})"
        )

    new_unstructured_relation = state.UnstructuredRelation(
        subj=subj,
        predicate=predicate,
        obj=obj
    )

    state.proposed_unstructured_relations.append(
        new_unstructured_relation
    )

    # 修改 unstructured_relation 后之前的批准失效。
    state.isApproved = False

    return (
        f"Added unstructured_relation: "
        f"({subj}, {predicate}, {obj})"
    )


@tool
def remove_proposed_unstructured_relation(
    subj: str,
    predicate: str,
    obj: str
) -> str:
    """
    Remove a proposed unstructured_relation type.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return (
            "Error: No state instance available."
        )

    subj = str(subj).strip()
    predicate = str(predicate).strip()
    obj = str(obj).strip()

    initial_length = len(
        state.proposed_unstructured_relations
    )

    state.proposed_unstructured_relations = [
        unstructured_relation
        for unstructured_relation in state.proposed_unstructured_relations
        if not (
            unstructured_relation.subj == subj
            and unstructured_relation.predicate == predicate
            and unstructured_relation.obj == obj
        )
    ]

    if (
        len(state.proposed_unstructured_relations)
        == initial_length
    ):
        return (
            f"Error: unstructured_relation "
            f"({subj}, {predicate}, {obj}) "
            f"not found."
        )

    state.isApproved = False

    return (
        f"Removed unstructured_relation: "
        f"({subj}, {predicate}, {obj})"
    )


@tool
def get_proposed_unstructured_relation() -> list[str]:
    """
    Return the current proposed unstructured_relation types.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return [
            "Error: No state instance available."
        ]

    return [
        str(unstructured_relation)
        for unstructured_relation in state.proposed_unstructured_relations
    ]


@tool
def approve_proposed_unstructured_relation() -> str:
    """
    Approve the current proposed unstructured_relation types.
    """
    state = get_current_unstructured_relation_state()

    if state is None:
        return (
            "Error: No state instance available."
        )

    if not state.proposed_unstructured_relations:
        return (
            "Error: No unstructured_relations have been "
            "proposed. Please add unstructured_relations first."
        )

    state.isApproved = True

    return (
        f"Proposed unstructured_relations approved: "
        f"{[str(unstructured_relation) for unstructured_relation in state.proposed_unstructured_relations]}"
    )


tools = [
    get_approved_user_goal,
    get_approved_files,
    get_approved_unstructured_node,
    sample_file,
    add_proposed_unstructured_relation,
    remove_proposed_unstructured_relation,
    get_proposed_unstructured_relation,
    approve_proposed_unstructured_relation,
]

tool_map = {
    selected_tool.name: selected_tool
    for selected_tool in tools
}


# ============================================================
# 模型
# ============================================================

model = LLM_CHAT_COMPLETION


# ============================================================
# 历史消息转换
# ============================================================

def history_to_chat_messages(
    state: UnstructuredRelationConstructionState
) -> list[tuple[str, str]]:
    chat_messages = []

    for message in state.messages:
        if message.startswith("user: "):
            chat_messages.append(
                (
                    "user",
                    message.removeprefix(
                        "user: "
                    )
                )
            )

        elif message.startswith("agent: "):
            chat_messages.append(
                (
                    "assistant",
                    message.removeprefix(
                        "agent: "
                    )
                )
            )

        elif message.startswith("system: "):
            chat_messages.append(
                (
                    "system",
                    message.removeprefix(
                        "system: "
                    )
                )
            )

    return chat_messages


# ============================================================
# agent_node
# ============================================================

def agent_node(
    state: UnstructuredRelationConstructionState
) -> UnstructuredRelationConstructionState:
    activate_state(state)

    def parse_agent_output(
        content: Any
    ) -> str:
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
                    item_text = (
                        item.get("text")
                        or item.get("content")
                        or ""
                    )

                    if isinstance(
                        item_text,
                        str
                    ):
                        parts.append(
                            item_text
                        )

            text = "\n".join(parts)

        else:
            text = str(content)

        return (
            text
            .replace("```text", "")
            .replace("```python", "")
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    system_prompt = f"""
You are an expert knowledge graph schema designer.

Your task is to analyze approved unstructured documents and propose
general unstructured_relation types that are relevant to the user's goal.

Do not propose individual unstructured_relations such as:
"Alice teaches Database Systems."

Instead propose reusable unstructured_relation patterns such as:
(Teacher, teaches_course, Course)

Every proposed unstructured_relation pattern has this form:
(subject unstructured_node type, predicate, object unstructured_node type)

Design rules:
- Subject and object must be selected from the approved unstructured_node types.
- Never invent a new subject or object type.
- The predicate must describe a meaningful relationship.
- The relationship must be supported by the approved source documents.
- Do not guess relationships that are not supported by the text.
- Use consistent predicate naming.
- Prefer lowercase snake_case predicates, for example:
  teaches_course
  belongs_to_department
  participates_in_project
- Do not use properties, quantities, identifiers, dates, scores,
  or measurements as object unstructured_node types.
- Do not add duplicate unstructured_relation patterns.
- Relationship direction matters:
  (Teacher, teaches_course, Course)
  is different from
  (Course, taught_by, Teacher)

Always call these tools first, one per response:
1. get_approved_user_goal()
2. get_approved_files()
3. get_approved_unstructured_node()

After receiving those results:
4. Use sample_file to inspect the approved documents.
5. Identify relationships supported by the documents.
6. Call add_proposed_unstructured_relation once for each proposed unstructured_relation pattern.
7. Call get_proposed_unstructured_relation to review the complete proposal.
8. Present the proposal to the user and ask for approval.
9. Only after explicit approval, call approve_proposed_unstructured_relation.
10. If the user requests changes, use remove_proposed_unstructured_relation and
    add_proposed_unstructured_relation as needed.

Supported document formats:
- TXT
- Markdown (.md and .markdown)
- PDF

PDF files may contain native text, scanned pages, or both.
sample_file automatically uses OCR when native PDF text is insufficient.

sample_file limits each preview to:
- {SAMPLE_MAX_LINES} lines
- {SAMPLE_MAX_CHARACTERS} characters
- {SAMPLE_MAX_PDF_PAGES} PDF pages

Tools:
- get_approved_user_goal()
- get_approved_files()
- get_approved_unstructured_node()
- sample_file(file_names: list[str])
- add_proposed_unstructured_relation(subj: str, predicate: str, obj: str)
- remove_proposed_unstructured_relation(subj: str, predicate: str, obj: str)
- get_proposed_unstructured_relation()
- approve_proposed_unstructured_relation()

Critical tool-call rules:
- Call only ONE tool in each response.
- If multiple tools are needed, call them in separate responses.
- A tool-call response must consist only of the [tool_call] marker
  followed by the exact tool call.
- Do not add explanations before or after a tool call.
- Use natural language only when reviewing proposals, requesting
  approval, or responding to user feedback.

Valid calls:
[tool_call] get_approved_user_goal()
[tool_call] get_approved_files()
[tool_call] get_approved_unstructured_node()
[tool_call] sample_file(['file1.txt', 'file2.md'])
[tool_call] sample_file(['report.pdf'])
[tool_call] add_proposed_unstructured_relation(
    'Teacher',
    'teaches_course',
    'Course'
)
[tool_call] remove_proposed_unstructured_relation(
    'Teacher',
    'teaches_course',
    'Course'
)
[tool_call] get_proposed_unstructured_relation()
[tool_call] approve_proposed_unstructured_relation()

Never approve unstructured_relations without explicit user approval.
Never sample a file that was not returned by get_approved_files.
Never use an unstructured_node type that was not returned by get_approved_unstructured_node.
""".strip()

    messages = [
        SystemMessage(
            content=system_prompt
        ),
        *history_to_chat_messages(state),
    ]

    response = model.invoke(
        messages
    )

    agent_text = parse_agent_output(
        response.content
    )

    state.add_agent(
        agent_text
    )

    return state


# ============================================================
# tool_node
# ============================================================

def tool_node(
    state: UnstructuredRelationConstructionState
) -> UnstructuredRelationConstructionState:
    activate_state(state)

    def parse_tool_call(
        text: str
    ) -> tuple[
        str | None,
        list,
        dict
    ]:
        if TOOL_CALL_MARKER not in text:
            return None, [], {}

        raw_text = (
            text
            .removeprefix("agent: ")
            .strip()
        )

        call_start = raw_text.find(
            TOOL_CALL_MARKER
        )

        if call_start < 0:
            return None, [], {}

        call_text = raw_text[
            call_start:
        ].strip()

        call_text = (
            call_text
            .removeprefix(
                TOOL_CALL_MARKER
            )
            .strip()
        )

        try:
            parsed = ast.parse(
                call_text,
                mode="eval"
            )

        except SyntaxError:
            return None, [], {}

        if not isinstance(
            parsed.body,
            ast.Call
        ):
            return None, [], {}

        call = parsed.body

        if not isinstance(
            call.func,
            ast.Name
        ):
            return None, [], {}

        tool_name = call.func.id

        try:
            positional_args = [
                ast.literal_eval(argument)
                for argument in call.args
            ]

            keyword_args = {
                keyword.arg: ast.literal_eval(
                    keyword.value
                )
                for keyword in call.keywords
                if keyword.arg is not None
            }

        except (
            ValueError,
            TypeError,
            SyntaxError
        ):
            return None, [], {}

        return (
            tool_name,
            positional_args,
            keyword_args
        )

    def build_tool_input(
        tool_name: str,
        positional_args: list,
        keyword_args: dict
    ) -> dict:
        selected_tool = tool_map[
            tool_name
        ]

        tool_arg_names = list(
            selected_tool.args.keys()
        )

        tool_input = {
            argument_name: argument_value
            for argument_name, argument_value
            in zip(
                tool_arg_names,
                positional_args
            )
        }

        tool_input.update(
            keyword_args
        )

        return tool_input

    if not state.messages:
        state.add_system(
            "Invalid tool call: "
            "message history is empty."
        )
        return state

    latest_message = state.messages[-1]

    (
        tool_name,
        positional_args,
        keyword_args
    ) = parse_tool_call(
        latest_message
    )

    if tool_name is None:
        state.add_system(
            "Invalid tool call."
        )
        return state

    if tool_name not in tool_map:
        state.add_system(
            f"Unknown tool: {tool_name}"
        )
        return state

    try:
        tool_input = build_tool_input(
            tool_name=tool_name,
            positional_args=positional_args,
            keyword_args=keyword_args
        )

        result = tool_map[
            tool_name
        ].invoke(
            tool_input
        )

        state.add_system(
            f"{tool_name} returned: "
            f"{result}"
        )

    except Exception as error:
        state.add_system(
            f"{tool_name} failed: "
            f"{error}"
        )

    return state


# ============================================================
# 路由
# ============================================================

def agent_should_continue(
    state: UnstructuredRelationConstructionState
) -> Literal[
    "tool",
    "user_chat"
]:
    activate_state(state)

    if not state.messages:
        return "user_chat"

    latest_message = state.messages[-1]

    if TOOL_CALL_MARKER in latest_message:
        return "tool"

    return "user_chat"


def tool_should_continue(
    state: UnstructuredRelationConstructionState
) -> Literal[
    "agent",
    "save"
]:
    activate_state(state)

    if state.isApproved:
        return "save"

    return "agent"


# ============================================================
# LangGraph 图
# ============================================================

unstructuredRelationConstructionGraph = build_agent_graph(
    globals(), "unstructuredRelationConstruction", UnstructuredRelationConstructionState,
    complete=lambda state: state.isApproved,
    outputs=lambda state: {UNSTRUCTURED_RELATION_CONSTRUCTION_PATH: csv_text(["subject", "predicate", "object"], [(unstructured_relation.subj, unstructured_relation.predicate, unstructured_relation.obj) for unstructured_relation in state.proposed_unstructured_relations])},
    coordinator=False,
)


if __name__ == "__main__":
    initial_state = (
        UnstructuredRelationConstructionState()
    )

    final_state = (
        unstructuredRelationConstructionGraph
        .invoke(initial_state)
    )
