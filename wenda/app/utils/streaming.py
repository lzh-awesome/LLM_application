"""流式响应工具模块 - 统一 SSE 格式生成"""

import json
import time
import uuid
from typing import AsyncGenerator, Dict, Optional

from app.config import settings


class StreamingHelper:
    """统一的流式响应生成器"""

    # 思考标签 - markdown 代码块结束符 + 两个换行符
    THINK_TAG = "```" + "\n\n"

    @staticmethod
    def generate_chat_id() -> str:
        """生成聊天 ID"""
        return f"chatcmpl-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def generate_sse_chunk(
        content: str,
        chunk_id: Optional[str] = None,
        model: Optional[str] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        """
        生成 SSE 格式的数据块

        Args:
            content: 内容
            chunk_id: 数据块 ID
            model: 模型名称
            finish_reason: 结束原因

        Returns:
            SSE 格式的字符串
        """
        chunk_id = chunk_id or StreamingHelper.generate_chat_id()
        model = model or settings.llm.model

        chunk_data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "delta": {"content": content},
                    "index": 0,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "service_tier": None,
            "system_fingerprint": None,
        }
        return f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

    @staticmethod
    def generate_end_chunk(
        chunk_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        生成结束标记的数据块

        Args:
            chunk_id: 数据块 ID
            model: 模型名称

        Returns:
            SSE 格式的字符串
        """
        chunk_id = chunk_id or StreamingHelper.generate_chat_id()
        model = model or settings.llm.model

        end_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "service_tier": None,
            "system_fingerprint": None,
        }
        return f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def generate_full_response(full_response: str) -> str:
        """
        生成包含完整响应的数据块

        Args:
            full_response: 完整响应内容

        Returns:
            SSE 格式的字符串
        """
        full_data = {"full_response": full_response}
        return f"data: {json.dumps(full_data, ensure_ascii=False)}\n\n"

    @staticmethod
    def generate_source_chunk(source_text: str = "大模型") -> str:
        """
        生成资料来源数据块

        Args:
            source_text: 资料来源文本

        Returns:
            SSE 格式的字符串
        """
        source_chunk = {
            "id": "source_chunk",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek-chat",
            "choices": [
                {
                    "delta": {
                        "content": f"\n\n\n###### 资料来源  \n\n  {source_text}"
                    },
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "service_tier": None,
            "system_fingerprint": None,
        }
        return f"data: {json.dumps(source_chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def generate_hop_info_chunk(depth: int) -> str:
        """
        生成跳数信息数据块

        Args:
            depth: 当前跳数

        Returns:
            SSE 格式的字符串
        """
        further_search = {
            "id": "llm_thinking_process",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek-chat",
            "choices": [
                {
                    "delta": {"content": f"\n\n--- 🔄 开始第 {depth + 2} 跳查询 ---\n"},
                    "index": 0,
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "service_tier": None,
            "system_fingerprint": None,
        }
        return f"data: {json.dumps(further_search, ensure_ascii=False)}\n\n"

    @staticmethod
    def generate_error_chunk(error_message: str) -> str:
        """
        生成错误信息数据块

        Args:
            error_message: 错误信息

        Returns:
            SSE 格式的字符串
        """
        error_chunk = {
            "id": f"chatcmpl-error-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": settings.llm.model,
            "choices": [
                {
                    "delta": {"content": f"错误: {error_message}"},
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "service_tier": None,
            "system_fingerprint": None,
        }
        return f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def generate_entity_info_chunk(entity_id: str, entity_name: str) -> str:
        """
        生成实体信息数据块

        Args:
            entity_id: 实体 ID
            entity_name: 实体名称

        Returns:
            SSE 格式的字符串列表（两个 SSE 消息）
        """
        return (
            f"data: {json.dumps({'entity_id': entity_id}, ensure_ascii=False)}\n\n"
            f"data: {json.dumps({'entity_name': entity_name}, ensure_ascii=False)}\n\n"
        )

    @staticmethod
    def extract_llm_answer(response: str, tag: str = THINK_TAG) -> str:
        """
        从 LLM 响应中提取答案内容

        Args:
            response: LLM 响应
            tag: 分隔标签

        Returns:
            提取后的答案
        """
        if tag in response:
            llm_output = response.split(tag, 1)[1]
        else:
            llm_output = response
        return llm_output.strip()


async def process_openai_stream(
    response: AsyncGenerator,
    chat_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    处理 OpenAI 流式响应，生成 SSE 格式

    Args:
        response: OpenAI 流式响应
        chat_id: 聊天 ID

    Yields:
        SSE 格式的字符串
    """
    chat_id = chat_id or StreamingHelper.generate_chat_id()
    full_response = ""

    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            yield StreamingHelper.generate_sse_chunk(content, chat_id)

    # 发送结束标记
    yield StreamingHelper.generate_end_chunk(chat_id)

    # 发送完整响应
    yield StreamingHelper.generate_full_response(full_response)


def generate_streaming_response(message: str) -> str:
    """
    为非流式消息生成流式格式（同步版本）

    Args:
        message: 消息内容

    Returns:
        SSE 格式的字符串生成器
    """
    chat_id = StreamingHelper.generate_chat_id()
    model = settings.llm.model

    for char in message:
        yield StreamingHelper.generate_sse_chunk(char, chat_id, model)

    # 发送结束标记
    yield StreamingHelper.generate_end_chunk(chat_id, model)

    # 发送完整响应
    yield StreamingHelper.generate_full_response(message)