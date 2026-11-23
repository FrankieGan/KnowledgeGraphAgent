"""Shared Chat Completion client for interactive agents and GraphRAG."""

from typing import Any, Iterable

from langchain_core.messages import BaseMessage
from neo4j_graphrag.llm import OpenAILLM


class ChatCompletionLLM(OpenAILLM):
    """Accept agent message formats while retaining GraphRAG's LLM interface.

    Both inherited invoke/ainvoke paths use chat.completions. String prompts
    retain GraphRAG's system_instruction/message_history handling; message lists
    pass through this shared conversion for synchronous and asynchronous calls.
    Agent tools remain text in messages, rather than native API tool calls.
    """

    def __init__(self, *args: Any, supports_structured_output: bool = True, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # OpenAI-compatible endpoints do not necessarily support JSON Schema.
        # GraphRAG uses this flag to select its prompt-based JSON extraction path.
        self.supports_structured_output = supports_structured_output

    def get_messages_v2(self, messages: list[Any]) -> Iterable[dict[str, Any]]:
        normalized = []
        for message in messages:
            if isinstance(message, BaseMessage):
                role, content = message.type, message.content
            elif isinstance(message, tuple) and len(message) == 2:
                role, content = message
            elif isinstance(message, dict):
                role, content = message["role"], message["content"]
            else:
                raise TypeError(f"Unsupported chat message type: {type(message).__name__}")

            role = {"human": "user", "ai": "assistant"}.get(role, role)
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"Unsupported chat message role: {role!r}")
            normalized.append({"role": role, "content": content})

        return super().get_messages_v2(normalized)
