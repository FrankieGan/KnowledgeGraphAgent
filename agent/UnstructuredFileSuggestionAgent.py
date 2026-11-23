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

# 如果 dependencies 中没有 MAX_READ，则默认读取 100 行。
SAMPLE_MAX_LINES = int(
    globals().get("MAX_READ", 100)
)

# 单个文件最多向 Agent 返回的字符数。
SAMPLE_MAX_CHARACTERS = 12000

# 最多预览多少个 PDF 页面。
SAMPLE_MAX_PDF_PAGES = 5

# PDF 页面原生文本少于该数量时执行 OCR。
MIN_PDF_NATIVE_CHARACTERS = 40

# Tesseract OCR 配置。
SAMPLE_OCR_LANGUAGES = "chi_sim+eng"
SAMPLE_OCR_DPI = 250


# ============================================================
# 当前状态引用
# ============================================================

# 不要把这个变量放入 dataclass。
# 否则 LangGraph 可能将它识别为状态字段。
_CURRENT_AGENT_STATE: Any = None


# ============================================================
# 状态定义
# ============================================================

@dataclass
class UnstructuredFileSuggestionAgentState(BudgetState):
    class UserGoal:
        def __init__(
            self,
            goal: str = "",
            approved: bool = False
        ):
            self.goal = goal
            self.approved = approved

    messages: list[str] = field(
        default_factory=list
    )

    approved_user_goal: str = ""

    import_files: list[str] = field(
        default_factory=list
    )

    suggested_files: list[str] = field(
        default_factory=list
    )

    is_approved: bool = False

    def __post_init__(self):
        global _CURRENT_AGENT_STATE
        _CURRENT_AGENT_STATE = self

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
    state: UnstructuredFileSuggestionAgentState
) -> None:
    """
    将当前 LangGraph 状态设置为工具可访问的状态。
    """
    global _CURRENT_AGENT_STATE
    _CURRENT_AGENT_STATE = state


def get_current_agent_state(
) -> UnstructuredFileSuggestionAgentState | None:
    """
    返回工具当前可以访问的 Agent 状态。
    """
    return _CURRENT_AGENT_STATE


# ============================================================
# 多格式文件预览辅助函数
# ============================================================

def normalize_preview_text(
    text: str
) -> str:
    """
    清理文本、PDF 和 OCR 结果中的控制字符和多余空行。
    """
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 删除行尾多余空格。
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
    同时按照最大行数和最大字符数限制预览内容。
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


def resolve_sample_file_path(
    file_name: str,
    state: UnstructuredFileSuggestionAgentState
) -> Path:
    """
    安全解析 Agent 请求预览的文件路径。

    仅允许读取：
    1. 已存在于 state.import_files 中的文件；
    2. IMPORT_UNSTRC_DIR 目录中的文件。

    防止使用 ../ 或绝对路径读取其他目录。
    """
    if not file_name:
        raise ValueError(
            "File name cannot be empty."
        )

    if file_name not in state.import_files:
        raise ValueError(
            f"File '{file_name}' is not in "
            f"the available import file list."
        )

    import_root = Path(
        IMPORT_UNSTRC_DIR
    ).resolve()

    resolved_path = (
        import_root / file_name
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


def sample_text_file(
    file_path: Path
) -> str:
    """
    读取 TXT、MD、Markdown 文档。

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


def render_pdf_page(
    page: fitz.Page
) -> Image.Image:
    """
    将 PDF 页面渲染为 Pillow 图片。
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
    对扫描 PDF 页面执行 Tesseract OCR。
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

    每个页面先尝试读取原生文本。
    原生文本不足时自动调用 Tesseract OCR。
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
    根据扩展名选择对应的文件预览方法。
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
# 节点定义
# ============================================================

def user_chat_node(
    state: UnstructuredFileSuggestionAgentState
) -> UnstructuredFileSuggestionAgentState:
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


def save_node(
    state: UnstructuredFileSuggestionAgentState
) -> UnstructuredFileSuggestionAgentState:
    activate_state(state)

    output_path = Path(
        SAVE_UNSTRC_SUGGESTED_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        for file_name in state.suggested_files:
            f.write(file_name + "\n")

    state.add_system(
        f"✅ Suggested files saved to "
        f"'{SAVE_UNSTRC_SUGGESTED_PATH}'."
    )

    return state


def import_node(
    state: UnstructuredFileSuggestionAgentState
) -> UnstructuredFileSuggestionAgentState:
    activate_state(state)

    state.add_user(
        "Hi, I need help selecting files "
        "for my knowledge graph."
    )

    # --------------------------------------------------------
    # 加载已批准的用户目标
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
    # 扫描支持的导入文件
    # --------------------------------------------------------

    import_root = Path(
        IMPORT_UNSTRC_DIR
    ).resolve()

    try:
        all_files = [
            path
            for path in import_root.iterdir()
            if path.is_file()
        ]

        supported_files = sorted(
            [
                path.name
                for path in all_files
                if (
                    path.suffix.lower()
                    in SUPPORTED_SAMPLE_EXTENSIONS
                )
            ],
            key=str.lower
        )

        ignored_files = sorted(
            [
                path.name
                for path in all_files
                if (
                    path.suffix.lower()
                    not in SUPPORTED_SAMPLE_EXTENSIONS
                )
            ],
            key=str.lower
        )

        state.import_files = supported_files

        state.add_system(
            f"📁 Found "
            f"{len(supported_files)} "
            f"supported import files: "
            f"{supported_files}"
        )

        if ignored_files:
            state.add_system(
                f"⚠️ Ignored unsupported files: "
                f"{ignored_files}"
            )

    except FileNotFoundError:
        state.import_files = []

        state.add_system(
            f"⚠️ Import directory "
            f"'{IMPORT_UNSTRC_DIR}' "
            f"does not exist."
        )

    except Exception as e:
        state.import_files = []

        state.add_system(
            f"⚠️ Error scanning import "
            f"directory: {e}"
        )

    return state


# ============================================================
# 工具定义
# ============================================================

@tool
def get_approved_user_goal() -> str:
    """
    Retrieve the approved user goal stored in the agent state.
    """
    state = get_current_agent_state()

    if state is None:
        return (
            "Error: Unable to get state instance."
        )

    return state.approved_user_goal


@tool
def list_import_files() -> list[str]:
    """
    List all available supported import documents.

    Supported formats:
    TXT, Markdown, and PDF.
    """
    state = get_current_agent_state()

    if state is None:
        return [
            "Error: Unable to get state instance."
        ]

    return state.import_files


@tool
def sample_file(
    file_names: list[str]
) -> dict[str, str]:
    """
    Preview TXT, Markdown, and PDF documents.

    PDF files may contain a native text layer or scanned images.
    Scanned pages are processed with Tesseract OCR.

    Preview size, line count, and PDF page count are limited.
    """
    state = get_current_agent_state()

    if state is None:
        return {
            "error": (
                "Unable to get state instance."
            )
        }

    if not file_names:
        return {
            "error": (
                "No file names were provided."
            )
        }

    result: dict[str, str] = {}

    # 去重并保持输入顺序。
    unique_file_names = list(
        dict.fromkeys(file_names)
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
def set_suggested_files(
    files: list[str]
) -> str:
    """
    Record the agent recommended file list.

    Every selected file must exist in the available import file list.
    """
    state = get_current_agent_state()

    if state is None:
        return (
            "Error: Unable to get state instance."
        )

    if not files:
        return (
            "Error: Suggested file list "
            "cannot be empty."
        )

    unique_files = list(
        dict.fromkeys(files)
    )

    unavailable_files = [
        file_name
        for file_name in unique_files
        if file_name not in state.import_files
    ]

    if unavailable_files:
        return (
            "Error: The following files are not "
            "available import files: "
            f"{unavailable_files}"
        )

    state.suggested_files = unique_files

    # 修改建议后，之前的批准应当失效。
    state.is_approved = False

    return (
        f"Suggested files set to: "
        f"{unique_files}"
    )


@tool
def approve_suggested_files() -> str:
    """
    Approve the current suggested file list.
    """
    state = get_current_agent_state()

    if state is None:
        return (
            "Error: Unable to get state instance."
        )

    if not state.suggested_files:
        return (
            "Error: No suggested files have "
            "been set. Please call "
            "set_suggested_files first."
        )

    state.is_approved = True

    return (
        f"Suggested files approved: "
        f"{state.suggested_files}"
    )


tools = [
    get_approved_user_goal,
    list_import_files,
    sample_file,
    set_suggested_files,
    approve_suggested_files,
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
    state: UnstructuredFileSuggestionAgentState
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
# Agent 节点
# ============================================================

def agent_node(
    state: UnstructuredFileSuggestionAgentState
) -> UnstructuredFileSuggestionAgentState:
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
You are a constructive critic AI reviewing a list of files.

Your goal is to help the user select appropriate unstructured documents
for constructing a knowledge graph.

Task:
Review the available file list for relevance to the kind of graph and
description contained in the approved user goal.

For any file that you are unsure about, use sample_file to inspect its
content before making a recommendation.

Only consider supported unstructured documents:
- TXT
- Markdown (.md and .markdown)
- PDF

PDF files may contain native text, scanned page images, or both.
sample_file automatically uses OCR when a PDF page does not contain
enough usable native text.

Tools:
- get_approved_user_goal():
  Retrieves the approved user goal.

- list_import_files():
  Returns all available supported file names.

- sample_file(file_names: list[str]):
  Previews TXT, Markdown, and PDF documents.
  Scanned PDF pages are processed with OCR.
  Preview is limited to {SAMPLE_MAX_LINES} lines,
  {SAMPLE_MAX_CHARACTERS} characters, and
  {SAMPLE_MAX_PDF_PAGES} PDF pages per file.

- set_suggested_files(files: list[str]):
  Records the recommended file list.

- approve_suggested_files():
  Finalizes the current recommendation after explicit user approval.

Critical tool-call rules:
- Call only ONE tool in each response.
- If multiple tools are needed, call them in separate responses.
- A tool-call response must contain only the tool-call marker and call.
- Do not put explanations before or after a tool call.
- Use natural language when speaking with the user.

Valid tool-call examples:
[tool_call] get_approved_user_goal()
[tool_call] list_import_files()
[tool_call] sample_file(['file1.txt', 'file2.md'])
[tool_call] sample_file(['report.pdf'])
[tool_call] set_suggested_files(['file1.txt', 'report.pdf'])
[tool_call] approve_suggested_files()

Workflow:
1. Call get_approved_user_goal to understand the requested graph.
2. Call list_import_files to retrieve available documents.
3. Evaluate file names for relevance.
4. Use sample_file for files whose relevance is uncertain.
5. Call set_suggested_files with the recommended shortlist.
6. Present the shortlist to the user and ask for approval.
7. If the user requests changes, revise the shortlist and call
   set_suggested_files again.
8. Only after the user explicitly approves the current shortlist,
   call approve_suggested_files.

Never approve a list without explicit user approval.
Never suggest a file that is not returned by list_import_files.
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
# 工具执行节点
# ============================================================

def tool_node(
    state: UnstructuredFileSuggestionAgentState
) -> UnstructuredFileSuggestionAgentState:
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
# 路由判断
# ============================================================

def should_continue(
    state: UnstructuredFileSuggestionAgentState
) -> Literal[
    "tool",
    "user_chat",
    "save"
]:
    activate_state(state)

    if not state.messages:
        return "user_chat"

    latest_message = state.messages[-1]

    if TOOL_CALL_MARKER in latest_message:
        return "tool"

    if state.is_approved:
        return "save"

    return "user_chat"


# ============================================================
# LangGraph 图
# ============================================================

UnstructuredFileSuggestionAgentGraph = build_agent_graph(
    globals(), "UnstructuredFileSuggestionAgent", UnstructuredFileSuggestionAgentState,
    complete=lambda state: state.is_approved,
    outputs=lambda state: {SAVE_UNSTRC_SUGGESTED_PATH: "".join(name + "\n" for name in state.suggested_files)},
    coordinator=False,
)


if __name__ == "__main__":
    initial_state = (
        UnstructuredFileSuggestionAgentState()
    )

    final_state = (
        UnstructuredFileSuggestionAgentGraph
        .invoke(initial_state)
    )
