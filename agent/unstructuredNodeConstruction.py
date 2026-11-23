from __future__ import annotations
from dependencies import *

import pymupdf as fitz
import pytesseract
from PIL import Image, ImageOps


# ============================================================
# sample_file 配置
# ============================================================

SUPPORTED_SAMPLE_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
}

# 如果 dependencies 中没有 MAX_READ，默认读取 100 行。
SAMPLE_MAX_LINES = int(
    globals().get("MAX_READ", 100)
)

# 单个文件最多返回的字符数，避免工具输出占用过多上下文。
SAMPLE_MAX_CHARACTERS = 12000

# 最多预览多少个 PDF 页面。
SAMPLE_MAX_PDF_PAGES = 5

# PDF 页面原生文本少于该数量时执行 OCR。
MIN_PDF_NATIVE_CHARACTERS = 40

# OCR 设置。
SAMPLE_OCR_LANGUAGES = "chi_sim+eng"
SAMPLE_OCR_DPI = 250


# ============================================================
# 当前状态引用
# ============================================================

# 不要把该变量放入 dataclass。
# 否则 LangGraph 可能将它识别为状态字段。
_CURRENT_UNSTRUCTURED_NODE_STATE: Any = None


# ============================================================
# 状态定义
# ============================================================

@dataclass
class UnstructuredNodeConstructionState(BudgetState):
    messages: list[str] = field(
        default_factory=list
    )

    approved_user_goal: str = ""

    approved_files: list[str] = field(
        default_factory=list
    )

    well_known_types: list[str] = field(
        default_factory=list
    )

    unstructured_nodes: list[str] = field(
        default_factory=list
    )

    isApproved: bool = False

    def __post_init__(self):
        global _CURRENT_UNSTRUCTURED_NODE_STATE
        _CURRENT_UNSTRUCTURED_NODE_STATE = self

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
    state: UnstructuredNodeConstructionState
) -> None:
    """
    将当前 LangGraph 状态设置为工具可访问的状态。
    """
    global _CURRENT_UNSTRUCTURED_NODE_STATE
    _CURRENT_UNSTRUCTURED_NODE_STATE = state


def get_current_unstructured_node_state(
) -> UnstructuredNodeConstructionState | None:
    """
    返回当前工具可访问的状态。
    """
    return _CURRENT_UNSTRUCTURED_NODE_STATE


# ============================================================
# 通用辅助函数
# ============================================================

def unique_preserving_order(
    values: list[str]
) -> list[str]:
    """
    去重并保持原始顺序。
    """
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
    """
    清理文本、PDF 和 OCR 结果中的控制字符与多余空行。
    """
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 删除行尾多余空白。
    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    # 三个以上连续空行压缩为两个。
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def useful_character_count(
    text: str
) -> int:
    """
    统计中文、字母和数字等有效字符。
    """
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
    """
    判断 PDF 页面自带的文本层是否足够可靠。
    """
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
    """
    同时按最大行数和最大字符数截断预览。
    """
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
# 文件路径验证
# ============================================================

def resolve_sample_file_path(
    file_name: str,
    state: UnstructuredNodeConstructionState
) -> Path:
    """
    安全解析 sample_file 请求的文件。

    仅允许读取：
    1. state.approved_files 中的文件；
    2. IMPORT_UNSTRC_DIR 目录中的文件。

    防止使用 ../ 或绝对路径越界读取其他文件。
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

    # 即使 approved_files 中意外出现绝对路径，
    # 也不允许跳出导入目录。
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
    """
    读取 TXT、MD 和 Markdown 文件。

    编码优先级：
    1. UTF-8 with BOM
    2. UTF-8
    3. GB18030
    """
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

            preview = truncate_preview(
                text
            )

            return (
                f"[File type: "
                f"{file_path.suffix.lower()}]\n"
                f"[Encoding: {encoding}]\n\n"
                f"{preview}"
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
    """
    将 PDF 页面渲染为指定 DPI 的 Pillow 图片。
    """
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
    """
    使用 Tesseract 对扫描 PDF 页面执行 OCR。
    """
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
    """
    预览普通 PDF、扫描 PDF 和混合 PDF。

    每页先读取原生文本；
    原生文本不足时自动执行 Tesseract OCR。
    """
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
    """
    根据文件扩展名选择对应的预览方法。
    """
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
    state: UnstructuredNodeConstructionState
) -> UnstructuredNodeConstructionState:
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
                f"⚠️ Ignored unsupported approved "
                f"files: {unsupported_files}"
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
    # 从节点构建计划读取已知节点标签
    # --------------------------------------------------------

    try:
        with open(
            STRUCTURED_NODE_CONSTRUCTION_PATH,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError(
                    "Node construction file "
                    "has no header."
                )

            if "label" not in reader.fieldnames:
                raise ValueError(
                    "Node construction file is "
                    "missing required 'label' column."
                )

            labels = []

            for row in reader:
                label = (
                    row.get("label") or ""
                ).strip()

                if label:
                    labels.append(label)

        state.well_known_types = (
            unique_preserving_order(
                labels
            )
        )

        state.add_system(
            f"✅ Loaded "
            f"{len(state.well_known_types)} "
            f"well-known types from node file: "
            f"{state.well_known_types}"
        )

    except FileNotFoundError:
        state.well_known_types = []

        state.add_system(
            f"⚠️ Node file "
            f"'{STRUCTURED_NODE_CONSTRUCTION_PATH}' not found. "
            f"Starting with empty "
            f"well-known types."
        )

    except Exception as e:
        state.well_known_types = []

        state.add_system(
            f"⚠️ Error reading node file: {e}"
        )

    return state


# ============================================================
# save_node
# ============================================================

def save_node(
    state: UnstructuredNodeConstructionState
) -> UnstructuredNodeConstructionState:
    activate_state(state)

    try:
        output_path = Path(
            UNSTRUCTURED_NODE_CONSTRUCTION_PATH
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as f:
            for unstructured_node in state.unstructured_nodes:
                f.write(unstructured_node + "\n")

        state.add_system(
            f"✅ unstructured_node list saved to "
            f"'{UNSTRUCTURED_NODE_CONSTRUCTION_PATH}'."
        )

    except Exception as e:
        state.add_system(
            f"⚠️ Error saving unstructured_node list: {e}"
        )

    state.isApproved = False

    return state


# ============================================================
# user_chat_node
# ============================================================

def user_chat_node(
    state: UnstructuredNodeConstructionState
) -> UnstructuredNodeConstructionState:
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
# 工具定义
# ============================================================

@tool
def get_approved_user_goal() -> str:
    """
    Return the approved user goal from the current state.
    """
    state = get_current_unstructured_node_state()

    if state is None:
        return (
            "Error: No state instance available."
        )

    return state.approved_user_goal


@tool
def get_approved_files() -> list[str]:
    """
    Return the approved TXT, Markdown, and PDF file names.
    """
    state = get_current_unstructured_node_state()

    if state is None:
        return [
            "Error: No state instance available."
        ]

    return state.approved_files


@tool
def get_well_known_types() -> list[str]:
    """
    Return the well-known unstructured_node types from the graph schema.
    """
    state = get_current_unstructured_node_state()

    if state is None:
        return [
            "Error: No state instance available."
        ]

    return state.well_known_types


@tool
def sample_file(
    file_names: list[str]
) -> dict[str, str]:
    """
    Preview approved TXT, Markdown, and PDF documents.

    PDF files may contain native text, scanned images, or both.
    Scanned PDF pages are processed using Tesseract OCR.

    Preview size, line count, and PDF page count are limited.
    """
    state = get_current_unstructured_node_state()

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
def set_proposed_unstructured_node(
    unstructured_nodes: list[str]
) -> str:
    """
    Set the proposed unstructured_node type list.

    unstructured_node names are stripped, empty values are removed,
    and duplicate values are removed while preserving order.
    """
    state = get_current_unstructured_node_state()

    if state is None:
        return (
            "Error: No state instance available."
        )

    if not unstructured_nodes:
        return (
            "Error: unstructured_node list cannot be empty."
        )

    cleaned_unstructured_nodes = []

    for unstructured_node in unstructured_nodes:
        if not isinstance(unstructured_node, str):
            return (
                "Error: Every unstructured_node type "
                "must be a string."
            )

        cleaned_unstructured_node = unstructured_node.strip()

        if cleaned_unstructured_node:
            cleaned_unstructured_nodes.append(
                cleaned_unstructured_node
            )

    cleaned_unstructured_nodes = (
        unique_preserving_order(
            cleaned_unstructured_nodes
        )
    )

    if not cleaned_unstructured_nodes:
        return (
            "Error: unstructured_node list cannot contain "
            "only empty values."
        )

    state.unstructured_nodes = cleaned_unstructured_nodes

    # 修改方案后，之前的批准失效。
    state.isApproved = False

    return (
        f"Proposed unstructured_nodes set to: "
        f"{cleaned_unstructured_nodes}"
    )


@tool
def approve_proposed_unstructured_node() -> str:
    """
    Approve the current proposed unstructured_node list.
    """
    state = get_current_unstructured_node_state()

    if state is None:
        return (
            "Error: No state instance available."
        )

    if not state.unstructured_nodes:
        return (
            "Error: No unstructured_nodes have been "
            "proposed. Please call "
            "set_proposed_unstructured_node first."
        )

    state.isApproved = True

    return (
        f"Proposed unstructured_nodes approved: "
        f"{state.unstructured_nodes}"
    )


tools = [
    get_approved_user_goal,
    get_approved_files,
    get_well_known_types,
    sample_file,
    set_proposed_unstructured_node,
    approve_proposed_unstructured_node,
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
# 消息历史转换
# ============================================================

def history_to_chat_messages(
    state: UnstructuredNodeConstructionState
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
    state: UnstructuredNodeConstructionState
) -> UnstructuredNodeConstructionState:
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
You are a top-tier algorithm designed for analyzing unstructured
documents and proposing named unstructured_node types that are relevant to a
user's knowledge graph goal.

unstructured_nodes are people, places, organizations, things, events, concepts,
and meaningful domain objects. unstructured_nodes are not quantities or simple
measurable values.

Your task is to propose unstructured_node TYPES, not individual unstructured_node instances.

Two unstructured_node type sources are available:

1. Well-known unstructured_node types:
   These are existing approved node labels from the graph schema.

2. Discovered unstructured_node types:
   These are additional domain unstructured_node types consistently supported
   by the approved documents and relevant to the user goal.

Design rules for well-known unstructured_node types:
- Prefer existing well-known unstructured_node types whenever they adequately
  represent the concept.
- Reuse exact approved node labels whenever possible.
- Do not create a synonym when a suitable well-known type already exists.

Design rules for discovered unstructured_node types:
- A discovered type must be supported by the approved documents.
- It must be relevant to the user goal.
- It should represent a reusable category, not one specific instance.
- Do not create unstructured_node types for quantities, measurements, identifiers,
  dates, scores, percentages, or simple attributes.
- Avoid overly broad types such as Thing, Object, Data, or Information.
- Avoid near-duplicate unstructured_node types.

Always call these tools first, one at a time:
1. get_approved_user_goal()
2. get_approved_files()
3. get_well_known_types()

After receiving those results:
4. Use sample_file to inspect approved documents.
5. Identify relevant well-known unstructured_node types.
6. Identify strongly supported discovered unstructured_node types.
7. Call set_proposed_unstructured_node with the combined proposed list.
8. Present the proposal to the user and ask for approval.
9. Only after explicit user approval, call approve_proposed_unstructured_node.

Supported document formats:
- TXT
- Markdown (.md and .markdown)
- PDF

PDF documents may contain native text, scanned pages, or both.
sample_file automatically uses OCR when native page text is unavailable
or insufficient.

Tools:
- get_approved_user_goal()
- get_approved_files()
- get_well_known_types()
- sample_file(file_names: list[str])
- set_proposed_unstructured_node(unstructured_nodes: list[str])
- approve_proposed_unstructured_node()

sample_file limits each preview to:
- {SAMPLE_MAX_LINES} lines
- {SAMPLE_MAX_CHARACTERS} characters
- {SAMPLE_MAX_PDF_PAGES} PDF pages

Critical tool-call rules:
- Call only ONE tool in each response.
- If several tools are required, call them in separate responses.
- A tool-call response must contain only the [tool_call] marker
  followed by the exact tool call.
- Do not place explanations before or after a tool call.
- Use natural language only when presenting proposals, requesting
  approval, or responding to user feedback.

Valid tool calls:
[tool_call] get_approved_user_goal()
[tool_call] get_approved_files()
[tool_call] get_well_known_types()
[tool_call] sample_file(['file1.txt', 'file2.md'])
[tool_call] sample_file(['document.pdf'])
[tool_call] set_proposed_unstructured_node(
    ['Person', 'Organization', 'Location']
)
[tool_call] approve_proposed_unstructured_node()

Never approve unstructured_node types without explicit user approval.
Never sample a file that was not returned by get_approved_files.
Never propose actual unstructured_node instances as unstructured_node types.
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
    state: UnstructuredNodeConstructionState
) -> UnstructuredNodeConstructionState:
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

        # 关键字参数覆盖同名的位置参数。
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
# 路由函数
# ============================================================

def agent_should_continue(
    state: UnstructuredNodeConstructionState
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
    state: UnstructuredNodeConstructionState
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

unstructuredNodeConstructionGraph = build_agent_graph(
    globals(), "unstructuredNodeConstruction", UnstructuredNodeConstructionState,
    complete=lambda state: state.isApproved,
    outputs=lambda state: {UNSTRUCTURED_NODE_CONSTRUCTION_PATH: "".join(unstructured_node + "\n" for unstructured_node in state.unstructured_nodes)},
    coordinator=False,
)


if __name__ == "__main__":
    initial_state = UnstructuredNodeConstructionState()

    final_state = (
        unstructuredNodeConstructionGraph.invoke(
            initial_state
        )
    )
