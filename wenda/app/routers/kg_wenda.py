"""知识图谱问答路由模块"""

import asyncio
import json
import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.services import (
    embedding_service,
    weaviate_service,
    ner_service,
    kg_qa_service,
)
from app.utils.streaming import StreamingHelper
from app.utils.prompts import PromptBuilder

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge-graph"])


# ==================== 端点 1: /deepseek_kg_wenda_test ====================

@router.post("/deepseek_kg_wenda_test")
async def kg_wenda_test(request: Request):
    """
    知识图谱问答测试接口

    请求体:
        messages: 消息列表
        is_concise: 是否简洁模式
        reply_direct: 是否直接回复

    返回:
        流式响应或实体选择列表
    """
    try:
        data = await request.json()
        messages_from_request = data.get("messages", [])
        is_concise = data.get("is_concise", False)
        reply_direct = data.get("reply_direct", False)

        _validate_messages(messages_from_request)

        logger.info(f"Received {len(messages_from_request)} messages")

        question = messages_from_request[-1]["content"]
        latest_user_message = messages_from_request[-1]

        # 工具调用判断
        tool_call_response = await kg_qa_service.tools_calling([latest_user_message])

        if tool_call_response and tool_call_response.tool_calls:
            return await _handle_tool_call(
                tool_call_response,
                messages_from_request,
                reply_direct
            )

        # NER 实体抽取
        llm_ner_entities = await ner_service.extract_entities(question)
        logger.info(f"NER 实体: {llm_ner_entities}")

        # 实体校验
        if ner_service.is_no_entity(llm_ner_entities) or ner_service.is_multiple_entities(llm_ner_entities):
            logger.info("走通用问答")
            return {"general_wenda": "通用问答"}

        # 实体向量查询
        entity = ner_service.get_first_entity(llm_ner_entities)
        query_vector = embedding_service.encode([entity])[0]

        entities = weaviate_service.query_entity_info("kg_wenda_large", query_vector, k=3)
        sorted_entities = sorted(entities, key=lambda x: x['certainty'], reverse=True)

        max_score = sorted_entities[0]['certainty'] if sorted_entities else 0
        logger.info(f"实体匹配最高分: {max_score}")

        # 根据分数分支处理
        if max_score >= 0.9:
            # 高置信度：直接进行多跳推理
            entity_id = sorted_entities[0]['id_']
            converted_entity_id = f"entity/{entity_id.split('/')[1]}"
            entity_name = sorted_entities[0]['name']

            return await _stream_multi_hop_response(
                question, query_vector, "kg_wenda_large", "natural_triples_new",
                converted_entity_id, entity_name
            )

        elif 0.5 <= max_score < 0.9:
            # 中置信度：返回实体选择列表
            entity_name_list = [e['name'] for e in sorted_entities if 'name' in e]

            return {
                "reply": "以下是我在知识库中检索到的实体，是否有您需要查询的实体？",
                "entity_name_list": entity_name_list,
                "url": f"http://10.117.1.238:8105/deepseek_kg_wenda_correct",
                "question": question,
                "replace_entity": entity,
            }

        else:
            # 低置信度：问答对查询
            return await _stream_qa_response(question, reply_direct)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


# ==================== 端点 2: /deepseek_kg_wenda_correct ====================

@router.post("/deepseek_kg_wenda_correct")
async def kg_wenda_correct(request: Request):
    """
    知识图谱问答纠错接口

    请求体:
        messages: 消息列表
        user_selected_entity: 用户选择的实体
        is_concise: 是否简洁模式
        reply_direct: 是否直接回复
        application_name: 应用名称
        link_name: 关系集合名称
    """
    try:
        data = await request.json()
        messages_from_request = data.get("messages", [])
        user_selected_entity = data.get("user_selected_entity", None)
        is_concise = data.get("is_concise", True)
        reply_direct = data.get("reply_direct", False)
        application_name = data.get("application_name", "kg_wenda")
        link_name = data.get("link_name", "kg_triples")

        _validate_messages(messages_from_request)

        question = messages_from_request[-1]["content"]

        # 问题向量
        query_vector = embedding_service.encode([question])[0]

        # 问答对查询
        query_results = weaviate_service.query_qa_collection("TextQA", query_vector, 3)
        sorted_qa = sorted(query_results, key=lambda x: x['certainty'], reverse=True)
        qa_max_score = sorted_qa[0]['certainty'] if sorted_qa else 0

        # 实体向量查询
        entity_vector = embedding_service.encode([user_selected_entity])[0]
        entities = weaviate_service.query_entity_info(application_name, entity_vector, k=3)
        sorted_entities = sorted(entities, key=lambda x: x['certainty'], reverse=True)
        max_score = sorted_entities[0]['certainty'] if sorted_entities else 0

        logger.info(f"qa_max_score: {qa_max_score}, entity_max_score: {max_score}")

        if qa_max_score <= max_score:
            # 走多跳推理
            entity_id = sorted_entities[0]['id_']
            converted_entity_id = f"entity/{entity_id.split('/')[1]}"
            entity_name = sorted_entities[0]['name']

            return await _stream_multi_hop_response(
                question, query_vector, application_name, link_name,
                converted_entity_id, entity_name
            )
        else:
            # 走问答对
            return await _stream_qa_response(question, reply_direct)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


# ==================== 端点 3: /kg_wenda_weaviate ====================

@router.post("/kg_wenda_weaviate")
async def kg_wenda_weaviate(request: Request):
    """
    知识图谱问答接口（Weaviate 版本）

    请求体:
        messages: 消息列表
        is_concise: 是否简洁模式
        reply_direct: 是否直接回复
        application_name: 应用名称
        link_name: 关系集合名称
    """
    try:
        data = await request.json()
        messages_from_request = data.get("messages", [])
        is_concise = data.get("is_concise", True)
        reply_direct = data.get("reply_direct", False)
        application_name = data.get("application_name", "kg_wenda")
        link_name = data.get("link_name", "kg_triples")

        _validate_messages(messages_from_request)

        question = messages_from_request[-1]["content"]
        latest_user_message = messages_from_request[-1]

        # 工具调用判断
        tool_call_response = await kg_qa_service.tools_calling([latest_user_message])

        if tool_call_response and tool_call_response.tool_calls:
            return await _handle_tool_call(
                tool_call_response,
                messages_from_request,
                reply_direct
            )

        # NER 实体抽取
        llm_ner_entities = await ner_service.extract_entities(question)
        logger.info(f"NER 实体: {llm_ner_entities}")

        # 实体校验
        if ner_service.is_no_entity(llm_ner_entities) or ner_service.is_multiple_entities(llm_ner_entities):
            logger.info("走通用问答")
            return {"general_wenda": "通用问答"}

        # 实体向量查询
        entity = ner_service.get_first_entity(llm_ner_entities)
        query_vector = embedding_service.encode([entity])[0]

        entities = weaviate_service.query_entity_info(application_name, query_vector, k=5)
        sorted_entities = sorted(entities, key=lambda x: x['certainty'], reverse=True)

        max_score = sorted_entities[0]['certainty'] if sorted_entities else 0
        logger.info(f"实体匹配最高分: {max_score}")

        if max_score >= 0.84:
            entity_id = sorted_entities[0]['id_']
            converted_entity_id = f"entity/{entity_id.split('/')[-1]}"
            entity_name = sorted_entities[0]['name']

            return await _stream_multi_hop_response(
                question, query_vector, application_name, link_name,
                converted_entity_id, entity_name
            )

        elif 0.5 <= max_score < 0.84:
            entity_name_list = [e['name'] for e in sorted_entities if 'name' in e]

            return {
                "reply": "以下是我在知识库中检索到的实体，是否有您需要查询的实体？",
                "entity_name_list": entity_name_list,
                "url": f"http://10.117.1.238:8105/deepseek_kg_wenda_correct",
                "question": question,
                "replace_entity": entity,
            }

        else:
            return await _stream_qa_response(question, reply_direct)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


# ==================== 辅助函数 ====================

def _validate_messages(messages: List[Dict]) -> None:
    """验证消息格式"""
    if not isinstance(messages, list) or not all(
        isinstance(msg, dict) and "content" in msg and "role" in msg
        for msg in messages
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid messages format. Expected list of {role, content} dicts."
        )

    if not messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from a user.")


async def _handle_tool_call(
    tool_call_response,
    messages: List[Dict],
    reply_direct: bool
):
    """处理工具调用"""
    tool = tool_call_response.tool_calls[0]
    tool_name = tool.function.name

    logger.info(f"工具调用: {tool_name}")

    if tool_name == "research_report":
        return await kg_qa_service.research_report(messages, reply_direct)
    elif tool_name == "generate_image":
        return {"image_generation": "图片生成"}
    elif tool_name == "get_picture":
        return {"image_search": "图片搜索"}
    elif tool_name == "get_video":
        return {"video_search": "视频搜索"}
    elif tool_name == "get_audio":
        return {"get_audio": "音频声纹检索"}
    elif tool_name == "audio_desc":
        return {"audio_desc": "音频描述"}
    elif tool_name == "map_and_chart":
        return {"general_wenda": "通用问答"}
    else:
        return {"general_wenda": "通用问答"}


async def _stream_multi_hop_response(
    question: str,
    query_vector,
    application_name: str,
    link_name: str,
    entity_id: str,
    entity_name: str,
) -> StreamingResponse:
    """生成多跳推理流式响应"""

    async def stream_response():
        try:
            # 发送实体信息
            yield StreamingHelper.generate_entity_info_chunk(entity_id, entity_name)

            # 执行多跳推理
            result_generator = kg_qa_service.multi_hop_reasoning(
                question, query_vector, application_name, link_name, entity_id
            )

            async for chunk in result_generator:
                yield chunk

        except Exception as e:
            logger.error(f"多跳推理流式错误: {e}")
            yield StreamingHelper.generate_error_chunk(str(e))

    return StreamingResponse(stream_response(), media_type="text/event-stream")


async def _stream_qa_response(question: str, reply_direct: bool) -> StreamingResponse:
    """生成问答对流式响应"""
    from app.services import llm_service

    # 问答对查询
    q_vec = embedding_service.encode([question])[0].tolist()
    query_results = weaviate_service.query_qa_collection("TextQA", q_vec, 3)

    filtered_results = [r for r in query_results if r.get('certainty', 0) >= 0.8]
    sorted_results = sorted(filtered_results, key=lambda x: x['certainty'], reverse=True)

    qa_text = ""
    if sorted_results:
        for qa in sorted_results:
            qa_text += f"问：{qa['query']}\n答：{qa['answer']}\n"
    else:
        qa_text = "\n\n问答对资料为空，请根据自身知识进行回答。\n"

    prompt = PromptBuilder.build_qa_prompt(qa_text, question)

    if reply_direct:
        prompt += " /no_think"

    messages_for_llm = [{"role": "system", "content": prompt}]

    async def stream_response():
        try:
            full_response = ""
            async for new_text in llm_service.stream_chat(messages_for_llm):
                if isinstance(new_text, dict) and "error" in new_text:
                    yield f"data: {json.dumps(new_text, ensure_ascii=False)}\n\n"
                    break

                full_response += new_text
                yield StreamingHelper.generate_sse_chunk(new_text)
                await asyncio.sleep(0.001)

            # 发送完整响应
            yield StreamingHelper.generate_full_response(full_response)

        except Exception as e:
            logger.error(f"问答对流式错误: {e}")
            yield StreamingHelper.generate_error_chunk(str(e))

    return StreamingResponse(stream_response(), media_type="text/event-stream")