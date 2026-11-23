# Knowledge Graph Agent

KnowledgGraphAgent is LLM-based software that can fully automate the process from reading source data and generating a graph structure to producing a concrete knowledge graph. It uses the LangGraph architecture, supports various structured and unstructured source files, and ultimately imports the results into a Neo4j database.

Original project code is licensed under the [MIT License](LICENSE). The PDF workflow uses PyMuPDF/MuPDF, which are separately licensed under AGPLv3 or an Artifex commercial license. The MIT grant does not waive the AGPL conditions applicable to covered combinations. See [NOTICE](NOTICE) and the [AGPLv3 text](licenses/AGPL-3.0.txt) before distributing or deploying the complete application.


- I. Python File Descriptions
- II. File Structure
- III. Usage
- IV. Debugging Notes

## I. Python File Descriptions

### 1.1 Main Workflow and Execution Order

These are the core programs of the software. Their order is also the execution order for the initial construction.

| No. | Module | Purpose and Main Output | Model Calls |
| --- | --- | --- | --- |
| 1 | `UserIntentAgent.py` | Talks with the user to determine the graph's topic and goal; saves `approved_goal.txt`. | Chat Completion |
| 2 | `StructuredFileSuggestionAgent.py` | Previews and recommends structured source files based on the goal; saves `strc_suggested_files.csv`. | Chat Completion |
| 3 | `UnstructuredFileSuggestionAgent.py` | Previews TXT, Markdown, and PDF files and recommends unstructured source files; saves `unstrc_suggested_files.csv`. | Chat Completion |
| 4 | `StructuredProposalCoordinator.py` | Coordinates the Proposal–Critic loop, reads review feedback, requests a human decision when necessary, and saves user feedback. | Chat Completion |
| 5 | `unstructuredNodeConstruction.py` | Proposes unstructured node types; reads existing labels from the structured node plan for the model to reuse; saves `unstructured_node_construction.csv`. | Chat Completion |
| 6 | `unstructuredRelationConstruction.py` | Proposes “start node type–relationship type–end node type” based on approved node types; saves `unstructured_relation_construction.csv`. | Chat Completion |
| 7 | `StructuredGraphConstruction.py` | Reads CSV/JSON according to the structured node and relationship plans, creates uniqueness constraints, and writes to Neo4j. | No model calls |
| 8 | `UnstructuredGraphConstruction.py` | Loads documents, reads PDF caches or performs OCR, splits text, extracts nodes and relationships according to the type definitions, and writes the graph. | Chat Completion + Embedding |
| 9 | `MergeGraph.py` | Translates property names in the two graphs, compares properties and values, and creates `SAME_AS` relationships between matching nodes with the same business label. | Chat Completion, for translating property names |
| 10 | `MergeSameAsNodes.py` | Uses APOC to physically merge nodes connected by `SAME_AS`, transfer relationships, and handle duplicate relationships. | No model calls |
| 11 | `MergeRepeatRelationships.py` | Calculates semantic similarity among multiple relationships on the same unordered pair of nodes and merges relationships that meet the threshold. | Embedding; no chat model calls |
| 12 | `ClearNodes.py` | Deletes nodes labeled `Chunk` or `Document` and their relationships; label matching is case insensitive. | No model calls |

`ClearNodes.py` runs last; structured graph construction precedes unstructured graph construction. Stages 5 and 6 generate **type definitions**; stages 7 and 8 write actual data to the database.

### 1.2 Other Programs

| Module | Purpose |
| --- | --- |
| `dependencies.py` | Centralizes dependency imports and all configuration settings (paths, model clients, database connections, Agent round counts, automatic approval, and debug mode). |
| `main.py` | Main workflow launcher. After startup, it displays the 12 main programs and their order. Enter the number of the program where execution should start; the remaining programs run automatically in order. Enter `1` for first-time use. |
| `DeleteAllData.py` | Standalone database clearing script for clearing the database so users can rebuild the graph. It is not in the main workflow menu and must be run manually. |
| `structuredProposalAgent.py` | A subprogram of `StructuredProposalCoordinator.py`. Proposes node and relationship construction plans based on the selected structured files, goal, and feedback; saves `structured_node_construction.csv` and `structured_relation_construction.csv`. Its main tools include `propose_node_construction`, `propose_relationship_construction`, and `construction_complete`. |
| `structuredCriticAgent.py` | A subprogram of `StructuredProposalCoordinator.py`. Reviews whether the construction plans meet the goal and the requirements for source-file column names, node properties, and relationship connections; saves `structured_feedback.txt`. |
| `chat_completion.py` | Unifies Chat Completion calls; adapts LangChain messages, role tuples, and GraphRAG string prompts; provides synchronous and asynchronous interfaces. Tool calls still use the text-based `[tool_call]` protocol. |
| `environment.py` | Reads process environment variables; on Windows, it can fall back to user-level and system-level variables, then to the untracked `agent/config.local.json` file. |
| `agent_runtime.py` | Wraps interactive subgraphs and manages autonomous rounds, resets after user replies, saving results to disk, and main workflow stage status. Runtime state is held in memory. |


## II. File Structure

### 2.1 Directory Purposes

```text
knowledgeGraph/
├─ README.md                         Usage and maintenance guide
├─ agent/                            Python source code
├─ assets/
│  ├─ structuredData/                Structured source data
│  └─ unstructuredData/              Unstructured source documents
│     └─ pdf2txt/                    PDF text extraction cache, generated automatically; no need to add files manually
├─ file/                             Automatically generated goals, file lists, construction plans, and feedback
└─ .venv/                            Local Python virtual environment
```


Structured reading supports `.csv` and `.json`; UTF-8 CSV is recommended. JSON can contain one object or a list of objects. Unstructured reading supports `.txt`, `.md`, `.markdown`, and `.pdf`. The current `assets` directory contains school-themed examples with fictional departments, teachers, courses, and students.

### 2.2 Official Outputs in `file/`

Files in `file/` are generated automatically by the program.

| File | Content Format | Producer → Consumer |
| --- | --- | --- |
| `approved_goal.txt` | User goal text | UserIntentAgent → file recommendation, plan generation, and other stages |
| `strc_suggested_files.csv` | One structured filename per line, no header | Structured file recommendation → Proposal, Critic |
| `unstrc_suggested_files.csv` | One unstructured filename per line, no header | Unstructured file recommendation → node/relationship type Agents, unstructured graph construction |
| `structured_node_construction.csv` | Four-column node plan with a header | Proposal → Critic, structured graph construction, unstructured node type Agent |
| `structured_relation_construction.csv` | Seven-column relationship plan with a header | Proposal → Critic, structured graph construction |
| `structured_feedback.txt` | Review comments or approval text | Critic → Proposal, Coordinator |
| `structured_user_feedback.txt` | Optional text of a manual or automatic reply | Coordinator human-interaction node → Proposal |
| `unstructured_node_construction.csv` | One node type per line, no header | Unstructured node type Agent → relationship type Agent, unstructured graph construction |
| `unstructured_relation_construction.csv` | Three-column table: `subject,predicate,object` | Unstructured relationship type Agent → unstructured graph construction |

Not every file with a `.csv` extension is a conventional table. File lists and node type lists should contain one item per line, without an additional header, line numbers, or comma-separated columns.


## III. Usage


### 3.1 First-Time Use

1. Install Python and the third-party packages imported by `agent/dependencies.py`. This project currently uses Python 3.14. Run the following commands from the project root; skip the first line if `.venv` already exists:

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install langchain-core langgraph neo4j neo4j-graphrag openai prompt-toolkit rapidfuzz numpy PyMuPDF Pillow sentence-transformers pypdf pdf2image pytesseract rapidocr-onnxruntime pydantic
   .\.venv\Scripts\python.exe -m pip check
   ```

2. Install and start Neo4j, and install a matching version of the APOC plugin.
3. Install Tesseract OCR and the `chi_sim` and `eng` language packs, and add it to PATH.
4. The project path `HOME_PATH` is detected automatically. Configure the LLM, Embedding model, and Neo4j address and username in `agent/dependencies.py`. Supply `OPENAI_API_KEY`, `ALI_API_KEY`, `DEEPSEEK_API_KEY`, `ALI_BASE_URL`, and `NEO4J_PASSWORD` through environment variables or a private local configuration file. To use the file, run `Copy-Item .\agent\config.example.json .\agent\config.local.json` and fill in the values you need; `config.local.json` is excluded by `.gitignore` and must never be committed. Process environment variables take priority, followed by persisted Windows variables and then the local file. `ALI_BASE_URL` defaults to the public Beijing endpoint; set it to your own workspace endpoint if needed. Note that when `AUTO_APPROVE=True`, human interactions other than UserIntentAgent automatically receive the reply `approve`; set it to `False` for manual review. `DEBUG_MODE` controls the echoing of system/user history.
5. Put structured files in `assets/structuredData/` and unstructured files in `assets/unstructuredData/`. UTF-8 encoding is recommended.
6. Start the main program `main.py`, enter `1`, and press Enter to build the graph from the beginning:

   ```powershell
   .\.venv\Scripts\python.exe .\agent\main.py
   ```

### 3.2 Regenerating Construction Plans

When node types, relationship types, or source data table structures change, regenerate the corresponding plans. **If source filenames or the number of files change, first rerun the corresponding file recommendation Agent or edit the file list in `file/`.**

**Structured plans:**

- To redesign from scratch, first back up and delete `file/structured_node_construction.csv` and `file/structured_relation_construction.csv`. If kept, the Agent will refer to the existing plans.
- If `file/structured_feedback.txt` is empty or contains approval of an old plan, delete that feedback to trigger a new review round. You can also enter revision suggestions in `file/structured_user_feedback.txt`.
- Run `agent/StructuredProposalCoordinator.py` separately and follow the prompts to confirm or continue revising. Turn off automatic approval first if you want a manual review.

**Unstructured plans:**

- Remove the `file/unstructured_node_construction.csv` and `file/unstructured_relation_construction.csv` files that need to be regenerated.
- Run `agent/unstructuredNodeConstruction.py` and `agent/unstructuredRelationConstruction.py` separately in that order. After node types change, relationship types should also be checked again; the node type stage refers to existing structured labels.

When running a file separately, use the same interpreter as the main program. For example:

```powershell
.\.venv\Scripts\python.exe .\agent\StructuredProposalCoordinator.py
```

When the process finishes normally or reaches the round limit, the current plan is saved to `file/`; forcibly closing the process does not guarantee that changes still in memory will be saved.

### 3.3 Updating or Regenerating the Graph

- **Structured data only adds records or changes properties, while the primary keys and plans stay the same:** You can run `agent/StructuredGraphConstruction.py` separately to match and update by the existing primary keys. Records, relationships, or cleared properties removed from the source files are not automatically removed from the database; clear the database and rebuild if full synchronization is needed.
- **Unstructured documents are added or modified:** First update the file list and confirm that the node and relationship types still apply, then run `agent/UnstructuredGraphConstruction.py`. To completely replace the old extraction results, clear the database and rebuild. After changing a PDF, also remove or update the corresponding TXT cache in `assets/unstructuredData/pdf2txt/`.
- **Node or relationship types change:** First update the plans as described in the previous section, then rebuild the graph.

To clear the database, run this separately:

```powershell
.\.venv\Scripts\python.exe .\agent\DeleteAllData.py
```

This script directly deletes all nodes and relationships while retaining indexes, constraints, and local files. **Both graph types must be rebuilt after clearing the database:** Once the plans and file lists are ready, run `main.py` and enter `7` to execute all subsequent stages starting with structured graph construction.

If you ran only the graph construction scripts separately, you can then enter `9` in the main program to run `MergeGraph → MergeSameAsNodes → MergeRepeatRelationships → ClearNodes` in order. The main menu continues from the selected number to the end; to run only one module, execute that Python file directly.


# 知识图谱 Agent

KnowledgGraphAgent是一个基于LLM的可全自动从读取源数据，生成图结构，到生成具体知识图谱的软件。该软件使用LangGraph架构，支持多种结构化和非结构化源文件，并最终导入到Neo4j数据库。

项目原创代码采用 [MIT 许可证](LICENSE)。PDF 流程使用的 PyMuPDF/MuPDF 单独采用 AGPLv3 或 Artifex 商业授权；MIT 授权不免除受 AGPL 约束的组合程序所需履行的义务。完整应用的分发和部署须同时考虑这些条件，详见 [NOTICE](NOTICE) 和 [AGPLv3 全文](licenses/AGPL-3.0.txt)。

分发受 AGPL 约束的组合程序时，应保留适用的版权和许可声明，并按许可证要求提供对应版本的源码等材料。部署支持远程网络交互的修改版受约束程序时，应按照 AGPL 第 13 条向远程用户显著提供免费获取对应源码的方式；源码应覆盖实际部署版本及许可证要求的构建、安装材料，不应包含私有密钥或密码。仅将源码保存在用户无权访问的私有仓库中，不能替代适用的源码提供义务。需要不受这些 AGPL 条件约束的分发或部署方式时，应取得合适的商业授权或替换相关依赖。


- 一、各 Python 文件功能介绍
- 二、文件结构
- 三、使用
- 四、调试注意事项

## 一、各 Python 文件功能介绍

### 1.1 主流程及执行顺序

该部分是改软件的核心程序。也是首次构建时的运行顺序。

| 编号 | 模块 | 作用及主要产物 | 模型调用 |
| --- | --- | --- | --- |
| 1 | `UserIntentAgent.py` | 与用户对话，确定图谱主题和目标；保存 `approved_goal.txt`。 | Chat Completion |
| 2 | `StructuredFileSuggestionAgent.py` | 根据目标预览和推荐结构化源文件；保存 `strc_suggested_files.csv`。 | Chat Completion |
| 3 | `UnstructuredFileSuggestionAgent.py` | 预览 TXT、Markdown、PDF，推荐非结构化源文件；保存 `unstrc_suggested_files.csv`。 | Chat Completion |
| 4 | `StructuredProposalCoordinator.py` | 调度 Proposal–Critic 循环；读取评审反馈，必要时请求人工决策；保存用户反馈。 | Chat Completion |
| 5 | `unstructuredNodeConstruction.py` | 提议非结构化节点类型；读取结构化节点计划中的既有标签供模型复用；保存 `unstructured_node_construction.csv`。 | Chat Completion |
| 6 | `unstructuredRelationConstruction.py` | 基于已批准的节点类型，提议“起点类型—关系类型—终点类型”；保存 `unstructured_relation_construction.csv`。 | Chat Completion |
| 7 | `StructuredGraphConstruction.py` | 按结构化节点、关系计划读取 CSV/JSON，创建唯一约束并写入 Neo4j。 | 无模型调用 |
| 8 | `UnstructuredGraphConstruction.py` | 加载文档、读取 PDF 缓存或执行 OCR、切分文本，按类型定义提取节点与关系并写图。 | Chat Completion + Embedding |
| 9 | `MergeGraph.py` | 翻译两类图谱的属性名称，比较属性和值，为相同业务标签的匹配节点建立 `SAME_AS`。 | Chat Completion，用于属性名翻译 |
| 10 | `MergeSameAsNodes.py` | 通过 APOC 物理合并 `SAME_AS` 指向的节点，迁移关系并处理重复关系。 | 无模型调用 |
| 11 | `MergeRepeatRelationships.py` | 对同一无序节点对上的多条关系计算语义相似度，合并满足阈值的关系。 | Embedding；不调用聊天模型 |
| 12 | `ClearNodes.py` | 删除带有 `Chunk` 或 `Document` 标签的节点及其关系，标签匹配不区分大小写。 | 无模型调用 |

`ClearNodes.py` 位于最后；结构化图构建先于非结构化图构建。这里的第 5、6 阶段生成的是**类型定义**，第 7、8 阶段才把具体数据写入数据库。

### 1.2 其他程序

| 模块 | 作用 |
| --- | --- |
| `dependencies.py` | 集中导入依赖，以及所有配置设置（配置路径、模型客户端、数据库连接、Agent 轮次、自动批准和调试模式。） |
| `main.py` | 主流程运行窗口。运行后会显示12个主程序以及他们的顺序。输入需要开始运行程序的编号并按顺序自动运行剩下的程序。如果是首次使用，则输入1 |
| `DeleteAllData.py` | 独立数据库清空脚本，用于清空数据库方便用户重新生成图谱。不在主流程菜单中，需要手动运行 |
| `structuredProposalAgent.py` | `StructuredProposalCoordinator.py`中的子程序。根据已选结构化文件、目标和反馈提出节点及关系构建计划；保存 `structured_node_construction.csv`、`structured_relation_construction.csv`。主要工具包括 `propose_node_construction`、`propose_relationship_construction`、`construction_complete`。 |
| `structuredCriticAgent.py` | `StructuredProposalCoordinator.py`中的子程序。评审构建计划是否符合目标、源文件列名、节点属性和关系连接要求；保存 `structured_feedback.txt`。 |
| `chat_completion.py` | 统一 Chat Completion 调用；适配 LangChain 消息、角色元组和 GraphRAG 字符串提示词，提供同步/异步接口。工具调用仍采用文本 `[tool_call]` 协议。 |
| `environment.py` | 读取进程环境变量；Windows 下可回退读取用户级和系统级变量，最后读取不纳入版本控制的 `agent/config.local.json`。|
| `agent_runtime.py` | 封装交互子图，管理自主轮次、用户回复后的重置、结果落盘和主流程阶段状态。运行状态保存在内存中。 |


## 二、文件结构

### 2.1 目录用途

```text
knowledgeGraph/
├─ README.md                         使用与维护说明
├─ agent/                            Python 源代码
├─ assets/
│  ├─ structuredData/                结构化源数据
│  └─ unstructuredData/              非结构化源文档
│     └─ pdf2txt/                    PDF 提取文本缓存，自动生成，不需要手动放入文件
├─ file/                             自动生成的目标、文件清单、构建计划和反馈
└─ .venv/                            本机 Python 虚拟环境
```


结构化读取支持 `.csv` 和 `.json`，推荐 UTF-8 CSV。JSON 可以是一个对象或对象列表。非结构化读取支持 `.txt`、`.md`、`.markdown`、`.pdf`。当前 `assets` 中已有学校主题样例，包括虚构的部门、教师、课程和学生；

### 2.2 `file/` 的正式产物

File中的文件会由程序自动。

| 文件 | 内容格式 | 产生方 → 使用方 |
| --- | --- | --- |
| `approved_goal.txt` | 用户目标文本 | UserIntentAgent → 文件推荐、计划生成等阶段 |
| `strc_suggested_files.csv` | 每行一个结构化文件名，无表头 | 结构化文件推荐 → Proposal、Critic |
| `unstrc_suggested_files.csv` | 每行一个非结构化文件名，无表头 | 非结构化文件推荐 → 节点/关系类型 Agent、非结构化图构建 |
| `structured_node_construction.csv` | 带表头的四列节点计划 | Proposal → Critic、结构化图构建、非结构化节点类型 Agent |
| `structured_relation_construction.csv` | 带表头的七列关系计划 | Proposal → Critic、结构化图构建 |
| `structured_feedback.txt` | 评审意见或批准文本 | Critic → Proposal、Coordinator |
| `structured_user_feedback.txt` | 人工或自动回复的文本，可选 | Coordinator 人工交互节点 → Proposal |
| `unstructured_node_construction.csv` | 每行一个节点类型，无表头 | 非结构化节点类型 Agent → 关系类型 Agent、非结构化图构建 |
| `unstructured_relation_construction.csv` | `subject,predicate,object` 三列表格 | 非结构化关系类型 Agent → 非结构化图构建 |

不是所有 `.csv` 后缀文件都是常规表格。文件清单和节点类型清单应保持一行一项，不要额外添加表头、行号或逗号分列。


## 三、使用


### 3.1 首次使用

1. 安装 Python，并安装 `agent/dependencies.py` 中导入的第三方包。本项目当前使用 Python 3.14；在项目根目录执行以下命令，已有 `.venv` 时跳过第一行：

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install langchain-core langgraph neo4j neo4j-graphrag openai prompt-toolkit rapidfuzz numpy PyMuPDF Pillow sentence-transformers pypdf pdf2image pytesseract rapidocr-onnxruntime pydantic
   .\.venv\Scripts\python.exe -m pip check
   ```

2. 安装并启动 Neo4j，并安装版本匹配的 APOC 插件。
3. 安装 Tesseract OCR 及 `chi_sim`、`eng` 语言包，并将其加入 PATH。
4. 项目路径 `HOME_PATH` 会自动定位。在 `agent/dependencies.py` 中配置大模型、Embedding 模型，以及 Neo4j 的地址和用户名。通过环境变量或本地私有配置文件提供 `OPENAI_API_KEY`、`ALI_API_KEY`、`DEEPSEEK_API_KEY`、`ALI_BASE_URL` 和 `NEO4J_PASSWORD`。如使用文件，执行 `Copy-Item .\agent\config.example.json .\agent\config.local.json` 后填写所需值；`config.local.json` 已被 `.gitignore` 排除，切勿提交。读取优先级为进程环境变量、Windows 持久化环境变量、本地配置文件。`ALI_BASE_URL` 默认使用北京地域公共端点，需要专属端点时自行配置。注意，`AUTO_APPROVE=True` 时，除 UserIntentAgent 外的人工交互会自动回复 `approve`；需要手动审核时设为 `False`。`DEBUG_MODE` 控制 system/user 的 history 回显。
5. 把结构化文件放入 `assets/structuredData/`，非结构化文件放入 `assets/unstructuredData/`，推荐使用 UTF-8 编码。
6. 启动主程序`main.py`，输入 `1` 并回车，从头构建图谱：

   ```powershell
   .\.venv\Scripts\python.exe .\agent\main.py
   ```

### 3.2 重新生成构建计划

当节点类型、关系类型或源数据表结构改变时，应重新生成相应计划。**若源文件名或数量发生变化，先重新运行对应的文件推荐 Agent，或修改 `file/` 中的文件清单。**

**结构化计划：**

- 如需从头设计，可先备份并删除 `file/structured_node_construction.csv` 和 `file/structured_relation_construction.csv`；保留时，Agent 会参考已有计划。
- 若 `file/structured_feedback.txt` 为空或已批准旧计划，可删除该反馈以触发新一轮评审。也可在 `file/structured_user_feedback.txt` 中填写修改建议。
- 单独运行 `agent/StructuredProposalCoordinator.py`，按提示确认或继续修改。希望人工审核时先关闭自动批准。

**非结构化计划：**

- 移除需要重新生成的 `file/unstructured_node_construction.csv` 和 `file/unstructured_relation_construction.csv`。
- 依次单独运行 `agent/unstructuredNodeConstruction.py`、`agent/unstructuredRelationConstruction.py`。节点类型改变后，关系类型也应重新检查；节点类型阶段会参考已有结构化标签。

单独运行文件时使用与主程序相同的解释器，例如：

```powershell
.\.venv\Scripts\python.exe .\agent\StructuredProposalCoordinator.py
```

正常结束或达到轮次上限时，当前方案会保存到 `file/`；强制关闭进程不保证保存尚在内存中的修改。

### 3.3 更新或重新生成图谱

- **结构化数据只增加记录或修改属性，且主键和计划未变：**可单独运行 `agent/StructuredGraphConstruction.py`，按现有主键匹配并更新。源文件中已删除的记录、关系或清空的属性不会自动从数据库中删除；需要完整同步时可清库重建。
- **非结构化文档新增或修改：**先更新文件清单，并确认节点、关系类型仍适用，再运行 `agent/UnstructuredGraphConstruction.py`。若要完整替换旧提取结果，可清库重建。修改 PDF 后，也要移除或更新 `assets/unstructuredData/pdf2txt/` 中对应的 TXT 缓存。
- **节点或关系类型发生变化：**先完成上一节的计划更新，再重新构建图谱。

需要清空数据库时，单独运行：

```powershell
.\.venv\Scripts\python.exe .\agent\DeleteAllData.py
```

该脚本会直接删除全部节点和关系，保留索引、约束及本地文件。**清库后两类图谱都要重新构建**：计划与文件清单准备好后，运行 `main.py` 并输入 `7`，会从结构化图构建开始执行全部后续阶段。

如果只单独运行了图构建脚本，随后可在主程序输入 `9`，依次完成 `MergeGraph → MergeSameAsNodes → MergeRepeatRelationships → ClearNodes`。主菜单从所选编号继续执行到最后；若只想运行一个模块，应直接运行该 Python 文件。


