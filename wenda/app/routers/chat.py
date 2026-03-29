"""聊天路由 - /brief_or_profound"""

import asyncio
import json
import logging
import time
from typing import List, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, Message
from app.services import llm_service, THINK_TAG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["chat"])

# 提示词模板
CONCISE_PROMPT_TEMPLATE = """
你是一个专业的问答助手，专注于提供简洁明了的回答。请根据以下对话历史，用最简短的语言给出准确的答案，避免任何不必要的解释或背景信息。

回答格式要求：
- 直接回答问题的核心内容，答案要是中文的。
- 如果问题涉及复杂背景，仅提供关键结论。
- 如果问"你是谁？"，你需要回答你是**小智**，你的智能AI助手。
- 如果你的答案中包含人物、机构、装备、基地、设施类的实体，请把这些实体的内容用**加粗，比如**基思·亚历山大**、**美国海军部**、**B-2幽灵式战略轰炸机**、**诺福克海军基地**、**朴茨茅斯海军造船厂**。
- 如果问"你使用的大语言模型推理框架是什么?",你需要回答我采用的是LangChain-Chatchat推理框架，可以支持多种大语言模型接入。
"""

DETAILED_PROMPT_TEMPLATE = """
你是一个专业的问答助手，专注于提供深入、详细的回答。请根据以下对话历史，尽可能全面地解答，包括背景知识、相关细节和扩展信息。如果问题涉及多个方面，请分点说明。

回答格式要求：
- 提供完整的背景信息和上下文。
- 使用清晰的逻辑结构，必要时分点说明。
- 包括相关的扩展内容或建议。
- 回答应至少包含三段内容，答案要是中文的。
- 如果问"你是谁？"，你需要回答你是**小智**，你的智能AI助手。
- 如果你的答案中包含人物、机构、装备、基地、设施类的实体，请把这些实体的内容用**加粗，**基思·亚历山大**、**美国海军部**、**B-2幽灵式战略轰炸机**、**诺福克海军基地**、**朴茨茅斯海军造船厂**。
- 如果问"你使用的大语言模型推理框架是什么?",你需要回答我采用的是LangChain-Chatchat推理框架，可以支持多种大语言模型接入。
"""


@router.post("/brief_or_profound")
async def brief_or_profound(request: Request):
    """
    简洁/深入模式问答接口

    Args:
        request: 包含 messages, is_concise, reply_direct 的请求

    Returns:
        StreamingResponse: 流式响应
    """
    try:
        data = await request.json()
        messages_from_request = data.get("messages", [])
        is_concise = data.get("is_concise", True)
        reply_direct = data.get("reply_direct", False)

        # 验证消息格式
        if not isinstance(messages_from_request, list) or not all(
            isinstance(msg, dict) and "content" in msg and "role" in msg
            for msg in messages_from_request
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid messages format. Expected list of {role, content} dicts."
            )

        logger.info(f"Received {len(messages_from_request)} messages")
        logger.debug(f"is_concise: {is_concise}, reply_direct: {reply_direct}")

        if not messages_from_request:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

        # 验证最后一条消息是用户消息
        latest_user_message = messages_from_request[-1]
        if latest_user_message["role"] != "user":
            raise HTTPException(status_code=400, detail="Last message must be from a user.")

        # 保留最近 5 轮对话 (11条消息)
        messages_for_llm = messages_from_request[-11:]

        # 注入系统消息
        system_prompt_content = CONCISE_PROMPT_TEMPLATE if is_concise else DETAILED_PROMPT_TEMPLATE

        if messages_for_llm and messages_for_llm[0]["role"] == "system":
            messages_for_llm[0]["content"] = system_prompt_content
        else:
            messages_for_llm.insert(0, {"role": "system", "content": system_prompt_content})

        # 处理 /no_think 标记
        if reply_direct and messages_for_llm and messages_for_llm[-1]["role"] == "user":
            messages_for_llm[-1]["content"] = messages_for_llm[-1]["content"].replace("/no_think", "").strip()
            messages_for_llm[-1]["content"] += " /no_think"

        async def stream_response():
            try:
                full_response = ""
                async for new_text in llm_service.stream_chat(messages_for_llm):
                    if isinstance(new_text, dict) and "error" in new_text:
                        yield f"data: {json.dumps(new_text, ensure_ascii=False)}\n\n"
                        break

                    full_response += new_text
                    chunk = {
                        "id": "llm_thinking_process",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "deepseek-chat",
                        "choices": [
                            {
                                "delta": {"content": new_text},
                                "index": 0,
                                "finish_reason": None,
                                "logprobs": None,
                            }
                        ],
                        "service_tier": None,
                        "system_fingerprint": None
                    }
                    json_chunk = json.dumps(chunk, ensure_ascii=False)
                    yield f"data: {json_chunk}\n\n"
                    await asyncio.sleep(0.001)

                # 发送资料来源
                source_chunk = {
                    "id": "source_chunk",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "deepseek-chat",
                    "choices": [
                        {
                            "delta": {"content": "\n\n\n###### 资料来源  \n\n  大模型"},
                            "index": 0,
                            "finish_reason": None,
                            "logprobs": None,
                        }
                    ],
                    "service_tier": None,
                    "system_fingerprint": None
                }
                yield f"data: {json.dumps(source_chunk, ensure_ascii=False)}\n\n"

                # 发送完整响应
                full_response_chunk = {"full_response": full_response}
                yield f"data: {json.dumps(full_response_chunk, ensure_ascii=False)}\n\n"
                logger.debug(f"Full response length: {len(full_response)}")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"General error in brief_or_profound: {e}")
        raise HTTPException(status_code=500, detail=str(e))