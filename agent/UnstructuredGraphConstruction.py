from __future__ import annotations
from dependencies import *
from pydantic import ValidationError

# ============================================================
# 配置
# ============================================================

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
}

OCR_LANGUAGES = "chi_sim+eng"
OCR_DPI = 300
MINIMUM_NATIVE_CHARACTERS = 40

TEXT_CHUNK_SIZE = 4000
TEXT_CHUNK_OVERLAP = 300

MAX_TEXT_FILE_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_OCR_PIXELS = 25_000_000

REMOVE_KG_BUILDER_LABEL = True


# ============================================================
# 通用辅助函数
# ============================================================

def describe_exception(exc: BaseException) -> str:
    """Keep the root cause visible even when a pipeline wrapper has no message."""
    details = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValidationError):
            message = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                for error in current.errors(include_input=False, include_url=False)
            )
        else:
            message = str(current).strip()
        details.append(
            f"{type(current).__name__}: {message}"
            if message else type(current).__name__
        )
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    description = " <- ".join(details)
    for key in (OPENAI_API_KEY, ALI_API_KEY, DEEPSEEK_API_KEY):
        if key:
            description = description.replace(key, "<REDACTED>")
    return description


def unique_preserving_order(
    values: List[str],
) -> List[str]:
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def run_coroutine_safely(
    coroutine: Coroutine[Any, Any, Any],
) -> Any:
    running_loop = None
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    # Run outside the except block so failures do not inherit the expected
    # "no running event loop" probe as their exception context.
    if running_loop is None:
        return asyncio.run(coroutine)

    result_container: Dict[str, Any] = {}
    error_container: Dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result_container["result"] = asyncio.run(
                coroutine
            )
        except BaseException as exc:
            error_container["error"] = exc

    thread = threading.Thread(
        target=runner,
        daemon=False,
    )

    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container.get("result")


def get_state_value(
    state: Any,
    key: str,
    default: Any = None,
) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


def stringify_document_metadata(
    metadata: dict[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        string_key = str(key)

        if isinstance(value, str):
            result[string_key] = value

        elif isinstance(value, bool):
            result[string_key] = (
                "true" if value else "false"
            )

        elif isinstance(
            value,
            (list, tuple, dict, set),
        ):
            serializable_value = (
                list(value)
                if isinstance(value, set)
                else value
            )

            result[string_key] = json.dumps(
                serializable_value,
                ensure_ascii=False,
            )

        else:
            result[string_key] = str(value)

    return result


# ============================================================
# Neo4j 客户端
# ============================================================

class Neo4jClient:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self.close()

    def open(self) -> None:
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(
                    self.user,
                    self.password,
                ),
            )

            self.driver.verify_connectivity()

            print(
                f"✅ Connected to {self.uri}"
            )

        except Exception as exc:
            raise ConnectionError(
                f"Connection failed: {exc}"
            ) from exc

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            self.driver = None

    def get_driver(self):
        if not self.driver:
            raise RuntimeError(
                "Driver not initialized. "
                "Call open() first."
            )

        return self.driver


# ============================================================
# Neo4j 标签清理
# ============================================================

def remove_kg_builder_label(
    driver,
) -> int:
    query = """
    MATCH (node:`__KGBuilder__`)
    REMOVE node:`__KGBuilder__`
    RETURN count(node) AS updated_nodes
    """

    with driver.session() as session:
        result = session.run(query)
        record = result.single()

        if record is None:
            return 0

        return int(
            record.get(
                "updated_nodes",
                0,
            )
        )


# ============================================================
# LangGraph 状态
# ============================================================

@dataclass
class GraphConstructionState2:
    construction_status: str = "pending"
    @dataclass
    class UnstructuredRelation:
        subj: str = ""
        predicate: str = ""
        obj: str = ""

        def __str__(self) -> str:
            return (
                f"({self.subj}, "
                f"{self.predicate}, "
                f"{self.obj})"
            )

        def to_csv_line(self) -> str:
            output = io.StringIO()

            writer = csv.writer(
                output,
                lineterminator="",
            )

            writer.writerow([
                self.subj,
                self.predicate,
                self.obj,
            ])

            return output.getvalue()

    messages: list[str] = field(
        default_factory=list
    )

    approved_files: list[str] = field(
        default_factory=list
    )

    unstructured_nodes: list[str] = field(
        default_factory=list
    )

    unstructured_relations: list[UnstructuredRelation] = field(
        default_factory=list
    )

    unstructured_relation_types: list[str] = field(
        default_factory=list
    )

    configuration_errors: list[str] = field(
        default_factory=list
    )

    configuration_warnings: list[str] = field(
        default_factory=list
    )

    processed_files: list[str] = field(
        default_factory=list
    )

    failed_files: list[dict] = field(
        default_factory=list
    )

    skipped_files: list[dict] = field(
        default_factory=list
    )

    extraction_results: list[dict] = field(
        default_factory=list
    )

    kg_builder_labels_removed: int = 0

    def add_system(
        self,
        text: str,
    ) -> None:
        message = f"system: {text}"
        self.messages.append(message)
        if DEBUG_MODE:
            print(message)


# ============================================================
# PDF 加载器（使用 RapidOCR）
# ============================================================


class HybridPDFDataLoader(DataLoader):
    """
    支持文本 PDF、扫描 PDF 和混合 PDF。

    ocr_mode:
        auto:
            原生文本不足时执行 OCR。

        always:
            每页执行 OCR。

        never:
            只读取 PDF 原生文本。
    """

    def __init__(
        self,
        ocr_mode: str = "auto",
        ocr_languages: str = OCR_LANGUAGES,
        dpi: int = OCR_DPI,
        minimum_native_characters: int = (
            MINIMUM_NATIVE_CHARACTERS
        ),
        rapidocr_config: Optional[dict] = None,
    ):
        valid_modes = {
            "auto",
            "always",
            "never",
        }

        if ocr_mode not in valid_modes:
            raise ValueError(
                f"Unsupported OCR mode '{ocr_mode}'. "
                f"Expected one of "
                f"{sorted(valid_modes)}."
            )

        if dpi < 72:
            raise ValueError(
                "OCR DPI must be at least 72."
            )

        if minimum_native_characters < 0:
            raise ValueError(
                "minimum_native_characters "
                "cannot be negative."
            )

        self.ocr_mode = ocr_mode
        self.ocr_languages = ocr_languages
        self.dpi = dpi
        self.minimum_native_characters = (
            minimum_native_characters
        )

        # 初始化 RapidOCR，可传入配置
        if rapidocr_config is None:
            rapidocr_config = {
                "use_angle_cls": False,
                "use_gpu": False,
            }

        self.ocr = RapidOCR(**rapidocr_config)

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace(
            "\x00",
            "",
        )

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+\n",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def useful_character_count(
        text: str,
    ) -> int:
        if not text:
            return 0

        return sum(
            1
            for character in text
            if character.isalnum()
        )

    def native_text_is_usable(
        self,
        text: str,
    ) -> bool:
        normalized = self.normalize_text(
            text
        )

        if not normalized:
            return False

        useful_count = (
            self.useful_character_count(
                normalized
            )
        )

        if (
            useful_count
            < self.minimum_native_characters
        ):
            return False

        printable_count = sum(
            1
            for character in normalized
            if (
                character.isprintable()
                or character in "\n\t"
            )
        )

        printable_ratio = (
            printable_count
            / len(normalized)
        )

        return printable_ratio >= 0.85

    def should_run_ocr(
        self,
        native_text: str,
    ) -> bool:
        if self.ocr_mode == "always":
            return True

        if self.ocr_mode == "never":
            return False

        return not self.native_text_is_usable(
            native_text
        )

    def render_page_as_image(
        self,
        page: fitz.Page,
    ) -> Image.Image:
        zoom = self.dpi / 72.0

        expected_width = max(
            1,
            int(page.rect.width * zoom),
        )

        expected_height = max(
            1,
            int(page.rect.height * zoom),
        )

        expected_pixels = (
            expected_width
            * expected_height
        )

        if expected_pixels > MAX_OCR_PIXELS:
            raise ValueError(
                "Rendered OCR page would contain "
                f"{expected_pixels:,} pixels, "
                "exceeding the limit of "
                f"{MAX_OCR_PIXELS:,}."
            )

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
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

    def run_ocr_sync(
        self,
        image: Image.Image,
    ) -> str:
        """
        使用 RapidOCR 执行 OCR，返回识别出的文本。
        """
        # 将 PIL Image 转换为 NumPy 数组 (RGB)
        import numpy as np
        image_array = np.array(image)

        # 调用 RapidOCR
        result, _ = self.ocr(image_array)

        if not result:
            return ""

        # result 是列表，每个元素是 [box, text, score]
        texts = [item[1] for item in result if item[1].strip()]
        full_text = "\n".join(texts)

        return self.normalize_text(full_text)

    async def extract_page(
        self,
        page: fitz.Page,
        page_number: int,
    ) -> Dict[str, Any]:
        native_text = self.normalize_text(
            page.get_text("text")
        )

        result: Dict[str, Any] = {
            "page_number": page_number,
            "text": native_text,
            "method": "native",
            "native_character_count": (
                self.useful_character_count(
                    native_text
                )
            ),
            "ocr_character_count": 0,
            "ocr_error": None,
        }

        if not self.should_run_ocr(
            native_text
        ):
            return result

        image: Optional[Image.Image] = None

        try:
            image = self.render_page_as_image(
                page
            )

            ocr_text = await asyncio.to_thread(
                self.run_ocr_sync,
                image,
            )

            result["ocr_character_count"] = (
                self.useful_character_count(
                    ocr_text
                )
            )

            if ocr_text:
                result["text"] = ocr_text
                result["method"] = "ocr"
            else:
                result["method"] = (
                    "native_ocr_empty"
                )

        except Exception as exc:
            result["ocr_error"] = str(exc)
            result["method"] = (
                "native_ocr_failed"
            )

        finally:
            if image is not None:
                image.close()

        return result

    @staticmethod
    def extract_document_title(
        document: fitz.Document,
        extracted_text: str,
        filepath: Path,
    ) -> str:
        metadata = document.metadata or {}

        metadata_title = str(
            metadata.get("title") or ""
        ).strip()

        if metadata_title:
            return metadata_title[:200]

        for line in extracted_text.splitlines():
            candidate = line.strip()

            if candidate:
                return candidate[:200]

        return filepath.stem

    async def run(
        self,
        filepath: Path,
        metadata: Optional[dict] = None,
    ) -> PdfDocument:
        filepath = Path(filepath).resolve()
        metadata = dict(metadata or {})

        if not filepath.exists():
            raise FileNotFoundError(
                f"PDF file not found: {filepath}"
            )

        if not filepath.is_file():
            raise ValueError(
                f"PDF path is not a file: "
                f"{filepath}"
            )

        if filepath.suffix.lower() != ".pdf":
            raise ValueError(
                "HybridPDFDataLoader only "
                f"supports PDF files: {filepath}"
            )

        try:
            document = fitz.open(
                str(filepath)
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to open PDF "
                f"'{filepath}': {exc}"
            ) from exc

        try:
            if document.needs_pass:
                raise ValueError(
                    "PDF is encrypted and requires "
                    f"a password: {filepath}"
                )

            if document.page_count == 0:
                raise ValueError(
                    f"PDF has no pages: {filepath}"
                )

            if (
                document.page_count
                > MAX_PDF_PAGES
            ):
                raise ValueError(
                    f"PDF has {document.page_count} "
                    "pages, exceeding the limit of "
                    f"{MAX_PDF_PAGES}."
                )

            extracted_pages: list[dict] = []

            for page_index in range(
                document.page_count
            ):
                page = document.load_page(
                    page_index
                )

                page_result = (
                    await self.extract_page(
                        page=page,
                        page_number=(
                            page_index + 1
                        ),
                    )
                )

                extracted_pages.append(
                    page_result
                )

            recognized_character_count = sum(
                self.useful_character_count(
                    page["text"]
                )
                for page in extracted_pages
            )

            if recognized_character_count == 0:
                ocr_errors = [
                    (
                        f"page="
                        f"{page['page_number']}: "
                        f"{page['ocr_error']}"
                    )
                    for page in extracted_pages
                    if page["ocr_error"]
                ]

                raise ValueError(
                    "No usable text could be "
                    f"extracted from '{filepath}'. "
                    f"OCR errors: {ocr_errors}"
                )

            plain_text = "\n\n".join(
                page["text"]
                for page in extracted_pages
                if page["text"]
            )

            title = self.extract_document_title(
                document=document,
                extracted_text=plain_text,
                filepath=filepath,
            )

            page_blocks = []

            for page in extracted_pages:
                page_number = (
                    page["page_number"]
                )

                page_text = (
                    page["text"].strip()
                )

                if not page_text:
                    page_text = (
                        "[No text recognized "
                        "on this page]"
                    )

                page_blocks.append(
                    f"--- PAGE {page_number} ---"
                    f"\n\n{page_text}"
                )

            complete_text = self.normalize_text(
                "\n\n".join(page_blocks)
            )

            ocr_pages = [
                page["page_number"]
                for page in extracted_pages
                if page["method"] == "ocr"
            ]

            native_pages = [
                page["page_number"]
                for page in extracted_pages
                if page["method"] == "native"
            ]

            failed_ocr_pages = [
                page["page_number"]
                for page in extracted_pages
                if page["ocr_error"]
            ]

            ocr_error_messages = [
                (
                    f"page="
                    f"{page['page_number']}: "
                    f"{page['ocr_error']}"
                )
                for page in extracted_pages
                if page["ocr_error"]
            ]

            document_metadata = {
                "title": title,
                "filename": filepath.name,
                "fileType": "pdf",
                "pageCount": document.page_count,
                "ocrMode": self.ocr_mode,
                "ocrLanguages": (
                    self.ocr_languages
                ),
                "ocrPages": ocr_pages,
                "nativeTextPages": native_pages,
                "failedOcrPages": (
                    failed_ocr_pages
                ),
                "ocrErrorMessages": (
                    ocr_error_messages
                ),
                **metadata,
            }

            document_info = DocumentInfo(
                path=str(filepath),
                metadata=(
                    stringify_document_metadata(
                        document_metadata
                    )
                ),
            )

            return PdfDocument(
                text=complete_text,
                document_info=document_info,
            )

        finally:
            document.close()

# ============================================================
# TXT / Markdown / PDF 统一加载器
# ============================================================

class UniversalDocumentDataLoader(
    HybridPDFDataLoader
):
    SUPPORTED_EXTENSIONS = (
        SUPPORTED_DOCUMENT_EXTENSIONS
    )

    @staticmethod
    def extract_text_title(
        text: str,
        filepath: Path,
    ) -> str:
        markdown_title = re.search(
            r"^\s*#\s+(.+?)\s*$",
            text,
            flags=re.MULTILINE,
        )

        if markdown_title:
            return (
                markdown_title
                .group(1)
                .strip()[:200]
            )

        for line in text.splitlines():
            candidate = line.strip()

            if candidate:
                return candidate[:200]

        return filepath.stem

    @staticmethod
    def get_pdf_cache_path(
            pdf_path: Path,
    ) -> Path:
        """
        获取 PDF 对应的 TXT 缓存路径。

        所有 PDF 生成的 TXT 文件统一保存到
        IMPORT_UNSTRC_DIR_PDFTOTXT 目录。

        例如：
            PDF:
                D:/imports/report.pdf

            TXT:
                IMPORT_UNSTRC_DIR_PDFTOTXT/report.txt
        """
        cache_directory = Path(
            IMPORT_UNSTRC_DIR_PDFTOTXT
        ).resolve()

        return (
                cache_directory
                / f"{pdf_path.stem}.txt"
        )

    @staticmethod
    def write_text_file_sync(
            filepath: Path,
            text: str,
    ) -> None:
        """
        使用 UTF-8 编码安全写入文本文件。

        如果 IMPORT_UNSTRC_DIR_PDFTOTXT 不存在，
        会自动创建该目录。
        """
        # 自动创建 TXT 输出目录。
        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = filepath.with_name(
            f"{filepath.name}.tmp"
        )

        try:
            with temporary_path.open(
                    "w",
                    encoding="utf-8",
                    newline="\n",
            ) as file:
                file.write(text)

            temporary_path.replace(
                filepath
            )

        finally:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    async def save_pdf_cache(
        self,
        cache_path: Path,
        text: str,
    ) -> None:
        """
        将 PDF 解析后的文本保存到同名 TXT 文件。
        """
        normalized_text = self.normalize_text(
            text
        )

        if not normalized_text:
            raise ValueError(
                "Cannot save empty PDF text "
                f"cache: {cache_path}"
            )

        encoded_size = len(
            normalized_text.encode("utf-8")
        )

        if encoded_size > MAX_TEXT_FILE_BYTES:
            raise ValueError(
                f"PDF text cache '{cache_path}' "
                f"contains {encoded_size:,} bytes, "
                "exceeding the limit of "
                f"{MAX_TEXT_FILE_BYTES:,} bytes."
            )

        try:
            await asyncio.to_thread(
                self.write_text_file_sync,
                cache_path,
                normalized_text,
            )

        except Exception as exc:
            raise ValueError(
                "Unable to save PDF text cache "
                f"'{cache_path}': {exc}"
            ) from exc

    async def load_text_document(
        self,
        filepath: Path,
        metadata: Optional[dict] = None,
    ) -> PdfDocument:
        """
        读取 TXT 或 Markdown 文档。
        """
        filepath = Path(filepath).resolve()
        metadata = dict(metadata or {})

        try:
            file_size = filepath.stat().st_size

        except Exception as exc:
            raise ValueError(
                f"Unable to inspect text file "
                f"'{filepath}': {exc}"
            ) from exc

        if file_size > MAX_TEXT_FILE_BYTES:
            raise ValueError(
                f"Text file '{filepath}' is "
                f"{file_size:,} bytes, exceeding "
                f"the limit of "
                f"{MAX_TEXT_FILE_BYTES:,} bytes."
            )

        try:
            file_bytes = await asyncio.to_thread(
                filepath.read_bytes
            )

        except Exception as exc:
            raise ValueError(
                f"Unable to read text file "
                f"'{filepath}': {exc}"
            ) from exc

        text: Optional[str] = None
        used_encoding: Optional[str] = None
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

                used_encoding = encoding
                break

            except UnicodeDecodeError as exc:
                decode_errors.append(
                    f"{encoding}: {exc}"
                )

        if text is None:
            raise ValueError(
                f"Unable to decode text file "
                f"'{filepath}'. "
                f"Errors: {decode_errors}"
            )

        text = self.normalize_text(
            text
        )

        if not text:
            raise ValueError(
                f"Text file is empty: {filepath}"
            )

        title = self.extract_text_title(
            text=text,
            filepath=filepath,
        )

        document_metadata = {
            "title": title,
            "filename": filepath.name,
            "fileType": (
                filepath
                .suffix
                .lower()
                .lstrip(".")
            ),
            "encoding": used_encoding or "",
            "pageCount": 1,
            "ocrMode": "not_applicable",
            "ocrPages": [],
            **metadata,
        }

        document_info = DocumentInfo(
            path=str(filepath),
            metadata=(
                stringify_document_metadata(
                    document_metadata
                )
            ),
        )

        return PdfDocument(
            text=text,
            document_info=document_info,
        )

    async def load_pdf_document(
        self,
        filepath: Path,
        metadata: Optional[dict] = None,
    ) -> PdfDocument:
        """
        加载 PDF，并使用同名 TXT 作为文本缓存。

        执行顺序：

        1. 获取 PDF 对应的同名 TXT 路径。
        2. 如果 TXT 已存在，优先尝试读取 TXT。
        3. 如果 TXT 不存在或读取失败，则解析 PDF。
        4. PDF 解析成功后，将识别文字保存到 TXT。
        5. 再从保存后的 TXT 中读取内容并返回。
        """
        filepath = Path(filepath).resolve()
        metadata = dict(metadata or {})

        cache_path = self.get_pdf_cache_path(
            filepath
        )

        # ----------------------------------------------------
        # 第一步：读取 PDF 前先尝试读取同名 TXT
        # ----------------------------------------------------

        if cache_path.exists():
            if cache_path.is_file():
                try:
                    print(
                        "📄 Found PDF text cache, "
                        f"reading: {cache_path}"
                    )

                    cache_metadata = {
                        **metadata,
                        "filename": filepath.name,
                        "fileType": "pdf",
                        "sourcePdf": str(filepath),
                        "textCache": str(cache_path),
                        "loadedFromTextCache": True,
                    }

                    cached_document = (
                        await self.load_text_document(
                            filepath=cache_path,
                            metadata=cache_metadata,
                        )
                    )

                    print(
                        "✅ Loaded PDF text from "
                        f"cache: {cache_path.name}"
                    )

                    return cached_document

                except Exception as exc:
                    # TXT 存在但为空、损坏或无法解码时，
                    # 不直接终止，而是重新解析 PDF。
                    print(
                        "⚠️ Unable to read PDF "
                        f"text cache '{cache_path}': "
                        f"{exc}"
                    )

                    print(
                        "🔄 The PDF will be "
                        "processed again."
                    )

            else:
                print(
                    "⚠️ PDF text cache path "
                    "exists but is not a file: "
                    f"{cache_path}"
                )

        else:
            print(
                "ℹ️ No PDF text cache found: "
                f"{cache_path}"
            )

        # ----------------------------------------------------
        # 第二步：TXT 不可用，读取并解析 PDF
        # ----------------------------------------------------

        print(
            "🔍 Processing PDF: "
            f"{filepath}"
        )

        extracted_document = (
            await super().run(
                filepath=filepath,
                metadata=metadata,
            )
        )

        extracted_text = self.normalize_text(
            extracted_document.text
        )

        if not extracted_text:
            raise ValueError(
                "PDF processing completed but "
                "returned no usable text: "
                f"{filepath}"
            )

        # ----------------------------------------------------
        # 第三步：保存 PDF 识别结果到同名 TXT
        # ----------------------------------------------------

        await self.save_pdf_cache(
            cache_path=cache_path,
            text=extracted_text,
        )

        print(
            "💾 PDF text saved to: "
            f"{cache_path}"
        )

        # ----------------------------------------------------
        # 第四步：重新读取刚刚生成的 TXT 文件
        # ----------------------------------------------------

        generated_cache_metadata = {
            **metadata,
            "filename": filepath.name,
            "fileType": "pdf",
            "sourcePdf": str(filepath),
            "textCache": str(cache_path),
            "loadedFromTextCache": False,
            "textCacheCreated": True,
        }

        try:
            cached_document = (
                await self.load_text_document(
                    filepath=cache_path,
                    metadata=(
                        generated_cache_metadata
                    ),
                )
            )

        except Exception as exc:
            raise ValueError(
                "PDF text was saved successfully, "
                "but the generated text file "
                "could not be read back from "
                f"'{cache_path}': {exc}"
            ) from exc

        print(
            "✅ Generated PDF text cache "
            f"was read successfully: "
            f"{cache_path.name}"
        )

        return cached_document

    async def run(
        self,
        filepath: Path,
        metadata: Optional[dict] = None,
    ) -> PdfDocument:
        filepath = Path(filepath).resolve()
        suffix = filepath.suffix.lower()

        if not filepath.exists():
            raise FileNotFoundError(
                f"Document file not found: "
                f"{filepath}"
            )

        if not filepath.is_file():
            raise ValueError(
                "Document path is not a file: "
                f"{filepath}"
            )

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document type "
                f"'{suffix}': {filepath}. "
                f"Supported types: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if suffix == ".pdf":
            return await self.load_pdf_document(
                filepath=filepath,
                metadata=metadata,
            )

        return await self.load_text_document(
            filepath=filepath,
            metadata=metadata,
        )

# ============================================================
# 智能文本分块器
# ============================================================

class SmartTextSplitter(TextSplitter):
    def __init__(
        self,
        max_chunk_size: int = TEXT_CHUNK_SIZE,
        overlap: int = TEXT_CHUNK_OVERLAP,
    ):
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be positive."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= max_chunk_size:
            raise ValueError(
                "overlap must be smaller than "
                "max_chunk_size."
            )

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split_long_text(
        self,
        text: str,
    ) -> List[str]:
        text = text.strip()

        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            expected_end = min(
                start + self.max_chunk_size,
                text_length,
            )

            end = expected_end

            if expected_end < text_length:
                candidate_breaks = [
                    text.rfind(
                        "\n\n",
                        start,
                        expected_end,
                    ),
                    text.rfind(
                        "\n",
                        start,
                        expected_end,
                    ),
                ]

                candidate_breaks.extend([
                    text.rfind(
                        punctuation,
                        start,
                        expected_end,
                    )
                    for punctuation in (
                        "。",
                        "！",
                        "？",
                        ". ",
                        "! ",
                        "? ",
                    )
                ])

                best_break = max(
                    candidate_breaks
                )

                minimum_end = (
                    start
                    + self.max_chunk_size // 2
                )

                if best_break >= minimum_end:
                    end = best_break + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = end - self.overlap

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    async def run(
        self,
        text: str,
    ) -> TextChunks:
        if not text or not text.strip():
            return TextChunks(chunks=[])

        semantic_sections = re.split(
            r"(?m)(?=^--- PAGE \d+ ---\s*$)"
            r"|(?=^#{1,3}\s+.+$)",
            text,
        )

        raw_chunks = []

        for section in semantic_sections:
            raw_chunks.extend(
                self.split_long_text(section)
            )

        return TextChunks(
            chunks=[
                TextChunk(
                    text=chunk,
                    index=index,
                )
                for index, chunk in enumerate(
                    raw_chunks
                )
                if chunk.strip()
            ]
        )


# ============================================================
# 配置加载
# ============================================================

def import_node(
    state: GraphConstructionState2,
) -> GraphConstructionState2:
    state.approved_files.clear()
    state.unstructured_nodes.clear()
    state.unstructured_relations.clear()
    state.unstructured_relation_types.clear()
    state.configuration_errors.clear()
    state.configuration_warnings.clear()

    try:
        with open(
            SAVE_UNSTRC_SUGGESTED_PATH,
            "r",
            encoding="utf-8-sig",
        ) as file:
            files = [
                line.strip()
                for line in file
                if line.strip()
            ]

        state.approved_files = (
            unique_preserving_order(files)
        )

        state.add_system(
            f"Loaded "
            f"{len(state.approved_files)} "
            f"approved files from "
            f"'{SAVE_UNSTRC_SUGGESTED_PATH}'."
        )

    except FileNotFoundError:
        error = (
            f"Approved file list "
            f"'{SAVE_UNSTRC_SUGGESTED_PATH}' "
            f"does not exist."
        )

        state.configuration_errors.append(
            error
        )

        state.add_system(f"❌ {error}")

    except Exception as exc:
        error = (
            "Error reading approved "
            f"file list: {exc}"
        )

        state.configuration_errors.append(
            error
        )

        state.add_system(f"❌ {error}")

    try:
        with open(
            UNSTRUCTURED_NODE_CONSTRUCTION_PATH,
            "r",
            encoding="utf-8-sig",
        ) as file:
            unstructured_nodes = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if (
            unstructured_nodes
            and unstructured_nodes[0].lower()
            in {
                "unstructured_node",
                "unstructured_nodes",
                "label",
                "node_type",
                "node type",
            }
        ):
            unstructured_nodes = unstructured_nodes[1:]

        state.unstructured_nodes = (
            unique_preserving_order(
                unstructured_nodes
            )
        )

        if not state.unstructured_nodes:
            state.configuration_errors.append(
                "No unstructured_node types were configured."
            )

        state.add_system(
            f"Loaded {len(state.unstructured_nodes)} "
            f"unstructured_node types from "
            f"'{UNSTRUCTURED_NODE_CONSTRUCTION_PATH}'."
        )

    except FileNotFoundError:
        error = (
            f"Unstructured_node file "
            f"'{UNSTRUCTURED_NODE_CONSTRUCTION_PATH}' does not exist."
        )

        state.configuration_errors.append(
            error
        )

        state.add_system(f"❌ {error}")

    except Exception as exc:
        error = (
            f"Error reading unstructured_node file: {exc}"
        )

        state.configuration_errors.append(
            error
        )

        state.add_system(f"❌ {error}")

    try:
        with open(
            UNSTRUCTURED_RELATION_CONSTRUCTION_PATH,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            normalized_headers = {
                str(header).strip().lower()
                for header in (
                    reader.fieldnames or []
                )
                if header is not None
            }

            expected_headers = {
                "subject",
                "predicate",
                "object",
            }

            missing_headers = (
                expected_headers
                - normalized_headers
            )

            if missing_headers:
                raise ValueError(
                    "Missing required headers: "
                    f"{sorted(missing_headers)}"
                )

            unstructured_relations = []
            seen_unstructured_relations = set()

            for line_number, original_row in enumerate(
                reader,
                start=2,
            ):
                row = {
                    str(key).strip().lower(): value
                    for key, value
                    in original_row.items()
                    if key is not None
                }

                subject = (
                    row.get("subject") or ""
                ).strip()

                predicate = (
                    row.get("predicate") or ""
                ).strip()

                obj = (
                    row.get("object") or ""
                ).strip()

                if (
                    not subject
                    and not predicate
                    and not obj
                ):
                    continue

                if not all([
                    subject,
                    predicate,
                    obj,
                ]):
                    state.configuration_errors.append(
                        f"Invalid unstructured_relation row "
                        f"{line_number}: subject, "
                        "predicate and object must "
                        "all be non-empty."
                    )
                    continue

                if subject not in state.unstructured_nodes:
                    state.configuration_errors.append(
                        f"UnstructuredRelation row {line_number} "
                        f"uses undefined subject "
                        f"type '{subject}'."
                    )
                    continue

                if obj not in state.unstructured_nodes:
                    state.configuration_errors.append(
                        f"UnstructuredRelation row {line_number} "
                        f"uses undefined object "
                        f"type '{obj}'."
                    )
                    continue

                unstructured_relation_key = (
                    subject,
                    predicate,
                    obj,
                )

                if unstructured_relation_key in seen_unstructured_relations:
                    continue

                seen_unstructured_relations.add(unstructured_relation_key)

                unstructured_relations.append(
                    state.UnstructuredRelation(
                        subj=subject,
                        predicate=predicate,
                        obj=obj,
                    )
                )

        state.unstructured_relations = unstructured_relations

        state.add_system(
            f"Loaded {len(unstructured_relations)} "
            f"relationship patterns from "
            f"'{UNSTRUCTURED_RELATION_CONSTRUCTION_PATH}'."
        )

    except FileNotFoundError:
        error = (
            f"UnstructuredRelation file "
            f"'{UNSTRUCTURED_RELATION_CONSTRUCTION_PATH}' does not exist."
        )

        state.configuration_errors.append(
            error
        )

        state.add_system(f"❌ {error}")

    except Exception as exc:
        error = (
            f"Error reading unstructured_relation file: {exc}"
        )

        state.configuration_errors.append(
            error
        )

        state.add_system(f"❌ {error}")

    state.unstructured_relation_types = unique_preserving_order(
        [
            unstructured_relation.predicate
            for unstructured_relation in state.unstructured_relations
        ]
    )

    state.add_system(
        f"Extracted "
        f"{len(state.unstructured_relation_types)} "
        "unique relationship types."
    )

    used_unstructured_node_types = set()

    for unstructured_relation in state.unstructured_relations:
        used_unstructured_node_types.add(
            unstructured_relation.subj
        )

        used_unstructured_node_types.add(
            unstructured_relation.obj
        )

    unused_unstructured_node_types = [
        unstructured_node
        for unstructured_node in state.unstructured_nodes
        if unstructured_node not in used_unstructured_node_types
    ]

    if unused_unstructured_node_types:
        warning = (
            "Unstructured_node types not used by any "
            "relationship pattern: "
            f"{unused_unstructured_node_types}"
        )

        state.configuration_warnings.append(
            warning
        )

        state.add_system(
            f"⚠️ {warning}"
        )

    return state


# ============================================================
# 文档路径验证
# ============================================================

def resolve_document_path(
    file_name: str,
) -> Path:
    if not file_name or not file_name.strip():
        raise ValueError(
            "Document filename is empty."
        )

    import_root = Path(
        IMPORT_UNSTRC_DIR
    ).resolve()

    candidate = Path(
        file_name.strip()
    )

    if candidate.is_absolute():
        raise ValueError(
            "Absolute document paths are not "
            f"allowed: {file_name}"
        )

    resolved = (
        import_root / candidate
    ).resolve()

    try:
        common_path = Path(
            os.path.commonpath([
                str(import_root),
                str(resolved),
            ])
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid document path "
            f"'{file_name}': {exc}"
        ) from exc

    if common_path != import_root:
        raise ValueError(
            "Document path escapes the "
            f"import directory: {file_name}"
        )

    if not resolved.exists():
        raise FileNotFoundError(
            f"Document does not exist: "
            f"{resolved}"
        )

    if not resolved.is_file():
        raise ValueError(
            f"Document is not a file: "
            f"{resolved}"
        )

    if (
        resolved.suffix.lower()
        not in SUPPORTED_DOCUMENT_EXTENSIONS
    ):
        raise ValueError(
            "Unsupported document extension: "
            f"{resolved.suffix}"
        )

    return resolved


# ============================================================
# LLM 提示词
# ============================================================

def build_unstructured_extraction_prompt() -> str:
    return """
You are a knowledge graph unstructured_node and relationship extraction system.

Extract unstructured_nodes and relationships only from the supplied input text.

Use only the node labels, relationship types, relationship directions,
and source/target patterns defined in the schema.

Schema:
{schema}

Return exactly one valid JSON object with this structure:
{{
  "nodes": [
    {{
      "id": "0",
      "label": "Teacher",
      "properties": {{
        "name": "Alice",
        "displayName": "Alice"
      }}
    }}
  ],
  "relationships": [
    {{
      "type": "teaches_course",
      "start_node_id": "0",
      "end_node_id": "1"
    }}
  ]
}}

Rules:
- Return JSON only.
- Do not use Markdown code fences.
- Do not add explanations.
- Use only node labels defined in the schema.
- Use only relationship types defined in the schema.
- Respect source type, target type and relationship direction.
- Every node must contain a non-empty "name" property.
- Every node must contain a non-empty "displayName" property.
- Reuse the same node ID for the same unstructured_node in this text.
- Every relationship endpoint must reference an ID in "nodes".
- Do not invent unsupported information.

Input text:
{text}
""".strip()


# ============================================================
# Schema 和 Pipeline
# ============================================================

def build_unstructured_schema(
    state: GraphConstructionState2,
) -> Dict[str, Any]:
    return {
        "node_types": list(
            state.unstructured_nodes
        ),
        "relationship_types": list(
            state.unstructured_relation_types
        ),
        "patterns": [
            (
                unstructured_relation.subj,
                unstructured_relation.predicate,
                unstructured_relation.obj,
            )
            for unstructured_relation in state.unstructured_relations
        ],
        "additional_node_types": False,
    }


def make_kg_builder(
    driver,
    llm,
    embedder,
    unstructured_schema: Dict[str, Any],
) -> SimpleKGPipeline:
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        from_file=True,
        file_loader=UniversalDocumentDataLoader(
            ocr_mode="auto",
            ocr_languages=OCR_LANGUAGES,
            dpi=OCR_DPI,
            minimum_native_characters=(
                MINIMUM_NATIVE_CHARACTERS
            ),
        ),
        text_splitter=SmartTextSplitter(
            max_chunk_size=TEXT_CHUNK_SIZE,
            overlap=TEXT_CHUNK_OVERLAP,
        ),
        schema=unstructured_schema,
        prompt_template=(
            build_unstructured_extraction_prompt()
        ),
    )


# ============================================================
# 图谱构建
# ============================================================

def run_node(
    state: GraphConstructionState2,
) -> GraphConstructionState2:
    if state.configuration_errors:
        state.add_system(
            "❌ KG construction skipped "
            "because configuration contains errors."
        )
        state.construction_status = "failed"
        return state

    if not state.approved_files:
        state.add_system(
            "⚠️ No approved files to process."
        )
        state.construction_status = "skipped"
        return state

    if not state.unstructured_nodes:
        state.add_system(
            "❌ No unstructured_node types configured."
        )
        state.construction_status = "skipped"
        return state

    if not state.unstructured_relations:
        state.add_system(
            "⚠️ No relationship patterns "
            "are configured."
        )

    unstructured_schema = build_unstructured_schema(
        state
    )

    llm = LLM_CHAT_COMPLETION
    embedder = EMBEDDING_LLM

    state.processed_files.clear()
    state.failed_files.clear()
    state.skipped_files.clear()
    state.extraction_results.clear()
    state.kg_builder_labels_removed = 0

    async def process_files() -> None:
        with Neo4jClient() as database:
            driver = database.get_driver()

            for file_name in state.approved_files:
                try:
                    file_path = (
                        resolve_document_path(
                            file_name
                        )
                    )

                except Exception as exc:
                    state.skipped_files.append({
                        "file": file_name,
                        "stage": "validation",
                        "error": str(exc),
                    })

                    state.add_system(
                        f"⚠️ Skipped "
                        f"'{file_name}': {exc}"
                    )
                    continue

                state.add_system(
                    f"Processing document: "
                    f"{file_path.name}"
                )

                try:
                    kg_builder = make_kg_builder(
                        driver=driver,
                        llm=llm,
                        embedder=embedder,
                        unstructured_schema=unstructured_schema,
                    )

                    pipeline_result = (
                        await kg_builder.run_async(
                            file_path=str(
                                file_path
                            )
                        )
                    )

                    result_value = getattr(
                        pipeline_result,
                        "result",
                        pipeline_result,
                    )

                    state.extraction_results.append({
                        "file": file_path.name,
                        "status": "success",
                        "result": str(
                            result_value
                        ),
                    })

                    state.processed_files.append(
                        file_path.name
                    )

                    state.add_system(
                        f"✅ Processed "
                        f"'{file_path.name}'."
                    )

                except Exception as exc:
                    error = describe_exception(exc)
                    state.failed_files.append({
                        "file": file_path.name,
                        "stage": "extraction",
                        "error": error,
                    })

                    state.add_system(
                        f"❌ Failed to process "
                        f"'{file_path.name}': {error}"
                    )

            if REMOVE_KG_BUILDER_LABEL:
                try:
                    updated_nodes = (
                        remove_kg_builder_label(
                            driver
                        )
                    )

                    state.kg_builder_labels_removed = (
                        updated_nodes
                    )

                    state.add_system(
                        "🎨 Removed '__KGBuilder__' "
                        f"from {updated_nodes} nodes. "
                        "'__Entity__' and business "
                        "labels were preserved."
                    )

                except Exception as exc:
                    warning = (
                        "Unable to remove "
                        f"'__KGBuilder__': {exc}"
                    )

                    state.configuration_warnings.append(
                        warning
                    )

                    state.add_system(
                        f"⚠️ {warning}"
                    )

    try:
        run_coroutine_safely(
            process_files()
        )

    except Exception as exc:
        error = describe_exception(exc)
        state.failed_files.append({
            "file": None,
            "stage": "pipeline",
            "error": error,
        })

        state.add_system(
            "❌ KG construction terminated "
            f"unexpectedly: {error}"
        )

    state.construction_status = "failed" if state.failed_files else "completed"
    state.add_system(
        "🎉 KG construction completed: "
        f"success={len(state.processed_files)}, "
        f"failed={len(state.failed_files)}, "
        f"skipped={len(state.skipped_files)}, "
        f"removedKGBuilderLabels="
        f"{state.kg_builder_labels_removed}."
    )

    return state


# ============================================================
# LangGraph
# ============================================================

graph_builder = StateGraph(
    GraphConstructionState2
)

graph_builder.add_node(
    "import",
    import_node,
)

graph_builder.add_node(
    "run",
    run_node,
)

graph_builder.set_entry_point(
    "import"
)

graph_builder.add_edge(
    "import",
    "run",
)

graph_builder.add_edge(
    "run",
    END,
)

UnstructuredGraphConstructionGraph = (
    graph_builder.compile()
)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    initial_state = (
        GraphConstructionState2()
    )

    final_state = (
        UnstructuredGraphConstructionGraph
        .invoke(initial_state)
    )

    processed_files = get_state_value(
        final_state,
        "processed_files",
        [],
    )

    failed_files = get_state_value(
        final_state,
        "failed_files",
        [],
    )

    skipped_files = get_state_value(
        final_state,
        "skipped_files",
        [],
    )

    configuration_errors = get_state_value(
        final_state,
        "configuration_errors",
        [],
    )

    configuration_warnings = get_state_value(
        final_state,
        "configuration_warnings",
        [],
    )

    kg_builder_labels_removed = get_state_value(
        final_state,
        "kg_builder_labels_removed",
        0,
    )

    print("\n" + "=" * 60)
    print("📊 Processing summary:")

    print(
        f"  Successful files: "
        f"{len(processed_files)}"
    )

    print(
        f"  Failed files: "
        f"{len(failed_files)}"
    )

    print(
        f"  Skipped files: "
        f"{len(skipped_files)}"
    )

    print(
        f"  Configuration errors: "
        f"{len(configuration_errors)}"
    )

    print(
        f"  Configuration warnings: "
        f"{len(configuration_warnings)}"
    )

    print(
        f"  __KGBuilder__ labels removed: "
        f"{kg_builder_labels_removed}"
    )

    print(
        "  __Entity__ labels removed: 0 "
        "(preserved)"
    )

    if processed_files:
        print("\n✅ Successful files:")

        for file_name in processed_files:
            print(
                f"  - {file_name}"
            )

    if failed_files:
        print("\n❌ Failed files:")

        for failure in failed_files:
            print(
                f"  - "
                f"{failure.get('file')}: "
                f"{failure.get('error')}"
            )

    if skipped_files:
        print("\n⚠️ Skipped files:")

        for skipped in skipped_files:
            print(
                f"  - "
                f"{skipped.get('file')}: "
                f"{skipped.get('error')}"
            )

    if configuration_errors:
        print(
            "\n❌ Configuration errors:"
        )

        for error in configuration_errors:
            print(f"  - {error}")

    if configuration_warnings:
        print(
            "\n⚠️ Configuration warnings:"
        )

        for warning in configuration_warnings:
            print(f"  - {warning}")

    print("=" * 60)
