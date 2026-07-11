"""HiAgent-specific one-to-one memory rewrite operation."""

import json
import re
from typing import Any

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import Message
from loguru import logger


@C.register_op()
class HiAgentOneToOneRewriteMemoryOp(BaseAsyncOp):
    """Rewrite retrieved memories while preserving item count and format."""

    file_path: str = __file__

    async def async_execute(self):
        memory_list: list[Any] = self.context.response.metadata["memory_list"]
        query: str = self.context.query
        messages = [Message(**x) if isinstance(x, dict) else x for x in self.context.get("messages", [])]

        if not memory_list:
            self.context.response.answer = ""
            return

        original_context = self._format_memories_for_context(memory_list)
        current_context = self._extract_context(messages)
        prompt = self.prompt_format(
            prompt_name="hiagent_one_to_one_memory_rewrite_prompt",
            current_query=query,
            current_context=current_context,
            original_context=original_context,
        )

        try:
            response = await self.llm.achat([Message(role=Role.USER, content=prompt)])
            rewritten_context = self._parse_json_response(response.content, "rewritten_context").strip()
            if rewritten_context:
                self.context.response.answer = rewritten_context
                logger.info("HiAgent one-to-one context successfully rewritten")
                return
        except Exception as exc:
            logger.error(f"Error in HiAgent one-to-one context rewriting: {exc}")

        self.context.response.answer = original_context

    @staticmethod
    def _format_memories_for_context(memories: list[Any]) -> str:
        formatted_memories = []
        for index, memory in enumerate(memories, 1):
            formatted_memories.append(
                f"Memory {index}:\n"
                f" When to use: {memory.when_to_use}\n"
                f" Content: {memory.content}\n"
            )
        return "\n".join(formatted_memories)

    @staticmethod
    def _extract_context(messages: list[Message]) -> str:
        if not messages:
            return ""

        context_parts = []
        for message in messages[-3:]:
            role = getattr(message.role, "value", message.role)
            content = message.content[:300] + "..." if len(message.content) > 300 else message.content
            context_parts.append(f"- {role}: {content}")
        return "\n".join(context_parts)

    @staticmethod
    def _parse_json_response(response: str, key: str) -> str:
        try:
            json_blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", response)
            candidates = json_blocks or [response]
            for candidate in candidates:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and isinstance(parsed.get(key), str):
                    return parsed[key]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for key '{key}'")
        return ""
