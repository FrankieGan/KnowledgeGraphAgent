from __future__ import annotations
import os
import glob
import ast
from dataclasses import dataclass, field
from typing import Literal
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import csv
import json
from typing import Any, Dict, List, Optional, Literal, Tuple, Union, Final, TypedDict, Iterator, Coroutine
from neo4j import GraphDatabase, Driver, ManagedTransaction
from chat_completion import ChatCompletionLLM
from environment import get_environment_variable
from agent_runtime import BudgetState, build_agent_graph, csv_text
from neo4j_graphrag.embeddings import OpenAIEmbeddings
import re
import asyncio
from pathlib import Path
from rapidfuzz import fuzz
import argparse
import numpy as np

import io
import threading
import pymupdf as fitz
from PIL import Image, ImageOps

from sentence_transformers import SentenceTransformer


from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import TextChunk, TextChunks
from neo4j_graphrag.components.data_loader import DataLoader
from neo4j_graphrag.components.types import LoadedDocument as PdfDocument, DocumentInfo
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

import asyncio
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

import unicodedata
from rapidocr_onnxruntime import RapidOCR

TOOL_CALL_MARKER = "[tool_call]"
SAME_AS_RELATIONSHIP = "SAME_AS"
# Neo4j GraphRAG owns this technical label; keep its value unchanged.
UNSTRUCTURED_NODE_LABEL = "__Entity__"


OPENAI_API_KEY = get_environment_variable("OPENAI_API_KEY")
ALI_API_KEY = get_environment_variable("ALI_API_KEY")
DEEPSEEK_API_KEY = get_environment_variable("DEEPSEEK_API_KEY")
ALI_BASE_URL = get_environment_variable("ALI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

HOME_PATH = Path(__file__).resolve().parent.parent.as_posix()

GOAL_SAVE_PATH = HOME_PATH + "/file/approved_goal.txt"
SAVE_STRC_SUGGESTED_PATH = HOME_PATH + "/file/strc_suggested_files.csv"
SAVE_UNSTRC_SUGGESTED_PATH = HOME_PATH + "/file/unstrc_suggested_files.csv"
IMPORT_STRC_DIR = HOME_PATH + "/assets/structuredData"
IMPORT_UNSTRC_DIR = HOME_PATH + "/assets/unstructuredData"
IMPORT_UNSTRC_DIR_PDFTOTXT = HOME_PATH + "/assets/unstructuredData/pdf2txt"
STRUCTURED_NODE_CONSTRUCTION_PATH = HOME_PATH + "/file/structured_node_construction.csv"
STRUCTURED_RELATION_CONSTRUCTION_PATH = HOME_PATH + "/file/structured_relation_construction.csv"
STRUCTURED_FEEDBACK_PATH = HOME_PATH + "/file/structured_feedback.txt"
STRUCTURED_USER_FEEDBACK_PATH = HOME_PATH + "/file/structured_user_feedback.txt"
UNSTRUCTURED_NODE_CONSTRUCTION_PATH = HOME_PATH + "/file/unstructured_node_construction.csv"
UNSTRUCTURED_RELATION_CONSTRUCTION_PATH = HOME_PATH + "/file/unstructured_relation_construction.csv"


OPENAI_CHAT_COMPLETION = ChatCompletionLLM(
    model_name="openai/gpt-5.3-codex",
    model_params={"temperature": 0.5},
    api_key=OPENAI_API_KEY,
    base_url="https://api.routerhub.ai/v1"
)

ALI_CHAT_COMPLETION = ChatCompletionLLM(
    model_name="qwen3.7-max-preview",
    api_key = ALI_API_KEY,
    base_url = ALI_BASE_URL
)

# 可选提供方：未设置密钥时不初始化，避免回退使用 OPENAI_API_KEY。
DEEPSEEK_CHAT_COMPLETION = (
    ChatCompletionLLM(
        model_name="deepseek-flash",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        supports_structured_output=False,  # DeepSeek does not accept response_format=json_schema.
    )
    if DEEPSEEK_API_KEY else None
)

ALI_EMBEDDING_LLM = OpenAIEmbeddings(
    model="qwen3.7-text-embedding-flash",
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL
)


# ============ LLM 配置 ============
LLM_CHAT_COMPLETION = DEEPSEEK_CHAT_COMPLETION
EMBEDDING_LLM = ALI_EMBEDDING_LLM

# 每次 Agent 子图调用的自主循环预算；一轮 = 执行一次 agent_node。
# 收到人工回复后，仅该 Agent 计数归零；工具返回和自动提示不重置。
# 每次重新调用子图均重新计数；Coordinator 新一轮调用 Proposal、Critic 时各自归零。
# 子图返回不会重置 Coordinator 自己的计数；达到上限保存当前结果后继续后续阶段。
# 工具、用户回复及 Embedding 不计数。两个图构建阶段和 MergeGraph 不设轮次上限。
# 以下配置必须是大于 0 的整数。
AGENT_MAX_ROUNDS = {
    "UserIntentAgent": 30,                  # 用户目标确认
    "StructuredFileSuggestionAgent": 30,    # 结构化文件选择
    "UnstructuredFileSuggestionAgent": 30,  # 非结构化文件选择
    "structuredProposalAgent": 30,              # 结构化构建方案
    "structuredCriticAgent": 30,                # 结构化方案评审
    "StructuredProposalCoordinator": 30,        # 协调器自己的决策，不包含子图轮次
    "unstructuredNodeConstruction": 30,                   # 非结构化实体类型
    "unstructuredRelationConstruction": 30,          # 非结构化关系类型
}

# ============ Neo4j 配置 ============
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = get_environment_variable("NEO4J_PASSWORD")

# ============ 其他 ============
MAX_ITERATION = 2 #StructuredProposalAgent 最大循环数
MAX_READ = 10 #Sample File最大读取行数
MAX_SEARCH = 10 # Search File最大读取数

AUTO_APPROVE = True  # 自动回复 approve；UserIntentAgent 始终保留人工输入
DEBUG_MODE = True  # 调试模式：控制写入 history 后的 system/user 打印
