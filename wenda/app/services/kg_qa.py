"""知识图谱问答服务模块 - 核心问答逻辑"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.services import embedding_service, weaviate_service, reranker_service, ner_service
from app.utils.streaming import StreamingHelper
from app.utils.prompts import PromptTemplates, PromptBuilder
from app.utils.triple_utils import TripleUtils, TripleFormatter

logger = logging.getLogger(__name__)


# 工具定义
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research_report",
            "description": "当用户请求中明确包含'研究报告'或'作战'时调用此函数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "包含关键词如'研究报告'、'作战规划'的查询语句。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "当用户请求中明确包含'生成'、'画一张'、'创作'、'设计'等关键词，并且请求是关于生成图像时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "用于图像生成的详细文本描述。"
                    },
                    "resolution": {
                        "type": "string",
                        "description": "可选参数，指定生成图像的分辨率。默认值为 '1024x1024'。",
                        "default": "1024x1024"
                    },
                    "style": {
                        "type": "string",
                        "description": "可选参数，指定图像的艺术风格。默认为 'realistic'。",
                        "default": "realistic"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_picture",
            "description": "当用户请求中明确包含'图片'、'照片'、'影像'、'图像'等关键词，并且请求是关于查找或获取某个事物的图片时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要查找图片的主体对象。只提取名词短语。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_video",
            "description": "当用户请求中明确包含'视频'、'录像'、'影片'等关键词，并且请求是关于查找动态视觉内容时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要查找视频的主体对象。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_audio",
            "description": "当需要根据音频内容或声纹特征检索相似音频时调用此函数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "描述音频检索需求。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "audio_desc",
            "description": "当用户请求描述音频内容时调用此函数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于描述音频内容的自然语言请求。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "map_and_chart",
            "description": "当用户请求描述为标记XXX位置和获取具体位置信息时或者用户需要统计相关数据信息时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于查询地理位置以及统计数据信息的自然语言请求。"
                    }
                },
                "required": ["query"]
            }
        }
    },
]


class KGQAService:
    """知识图谱问答服务"""

    def __init__(self):
        self._llm_client: Optional[AsyncOpenAI] = None

    @property
    def llm_client(self) -> AsyncOpenAI:
        """懒加载 LLM 客户端"""
        if self._llm_client is None:
            self._llm_client = AsyncOpenAI(
                base_url=settings.llm.base_url,
                api_key=settings.llm.api_key,
            )
        return self._llm_client

    async def tools_calling(self, messages: List[Dict[str, str]]) -> Optional[Dict]:
        """
        根据消息列表判断是否需要进行工具调用

        Args:
            messages: 消息列表

        Returns:
            工具调用消息或 None
        """
        usr_messages = messages[-1]
        new_content = usr_messages["content"].rstrip() + " /no_think"

        new_message = {
            "role": usr_messages["role"],
            "content": new_content
        }

        tool_messages = [
            {"role": "system", "content": PromptTemplates.TOOL_CALLING_SYSTEM_PROMPT},
            new_message
        ]

        start_time = time.perf_counter()

        try:
            response = await self.llm_client.chat.completions.create(
                model=settings.llm.model,
                messages=tool_messages,
                tools=TOOLS,
                temperature=0.01,
            )

            end_time = time.perf_counter()
            duration = end_time - start_time

            message = response.choices[0].message
            if message.tool_calls:
                logger.info(f"[Tool Calling] 工具调用耗时: {duration:.3f} 秒")
                return message
            else:
                logger.info(f"[Tool Calling] 无工具调用，判断耗时: {duration:.3f} 秒")
                return None

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.error(f"[Tool Calling] 调用失败，耗时: {duration:.3f} 秒, 错误: {e}")
            return None

    async def multi_hop_reasoning(
        self,
        question: str,
        query_vector,
        application_name: str,
        link_name: str,
        converted_entity_id: str,
        depth: int = 0,
        max_depth: int = 3,
        accumulated_triples: Optional[List] = None,
    ) -> AsyncGenerator[str, None]:
        """
        多跳推理函数 - 流式返回版本

        Args:
            question: 用户问题
            query_vector: 问题向量
            application_name: 应用名称（Weaviate 集合名）
            link_name: 关系集合名称
            converted_entity_id: 实体 ID (格式: entity/xxx)
            depth: 当前深度
            max_depth: 最大深度
            accumulated_triples: 累积的三元组信息

        Yields:
            SSE 格式的流式响应
        """
        if accumulated_triples is None:
            accumulated_triples = []

        if depth > max_depth:
            error_msg = "无法回答该问题，超过最大跳数限制。"
            for chunk in StreamingHelper.generate_sse_chunk(error_msg):
                yield chunk
            return

        # 根据实体 ID 查询相关三元组
        entity_id = converted_entity_id.split('/')[-1]
        related_triples = weaviate_service.query_triples_by_entity_id(link_name, entity_id, k=10)

        logger.info(f"related_triples 数量: {len(related_triples)}")

        # 如果是孤立的单实体，走通用问答
        if not related_triples:
            entities = weaviate_service.query_entity_info(application_name, query_vector, k=1)
            entity_info = entities[0] if entities else None
            entity_desc = entity_info.get("desc", "") if entity_info else ""

            accumulated_info = ""
            if accumulated_triples:
                accumulated_info = f"\n\n之前查询过程中获得的相关三元组信息：\n{TripleUtils.format_accumulated_triples(accumulated_triples)}\n"

            prompt = PromptBuilder.build_entity_isolated_prompt(
                question=question,
                entity_desc=entity_desc,
                accumulated_info=accumulated_info,
            )

            logger.debug(f"孤立实体 Prompt: {prompt[:200]}...")

            response = await self.llm_client.chat.completions.create(
                model=settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            async for chunk in self._process_streaming_response(response):
                yield chunk
            return

        # 问答对查询
        q_vec = embedding_service.encode([question])[0].tolist()
        query_results = weaviate_service.query_qa_collection("TextQA", q_vec, 3)

        # 过滤问答对结果
        filtered_qa_results = [r for r in query_results if r.get('certainty', 0) >= 0.8]
        sorted_qa_results = sorted(filtered_qa_results, key=lambda x: x['certainty'], reverse=True)

        qa_text = TripleFormatter.format_qa_results(sorted_qa_results) if sorted_qa_results else "\n\n问答对资料为空，请根据其他资料进行回答。\n"

        # 第一跳：向量查询三元组
        if depth == 0:
            query_vector_for_triples = embedding_service.encode([question])[0]
            triples = weaviate_service.query_triples(link_name, query_vector_for_triples, k=20)

            # Rerank
            reranked = reranker_service.rerank(question, triples)
            high_score_triples = [item for item in reranked if item.get("rerank_score", 0) > 0.5]

            if not high_score_triples and reranked:
                high_score_triples = [reranked[0]]

            if high_score_triples:
                best_triple = [high_score_triples[0]]
                if len(high_score_triples) < 4:
                    other_three_triples = [high_score_triples[0]] * 3
                else:
                    other_three_triples = high_score_triples[1:4]
            else:
                best_triple = [reranked[0]] if reranked else []
                other_three_triples = [reranked[0]] * 3 if reranked else []

        else:
            # 后续跳：Rerank 相关三元组
            reranked = reranker_service.rerank(question, related_triples)
            best_triple = [reranked[0]] if reranked else []
            other_three_triples = reranked[1:4] if len(reranked) >= 4 else reranked[1:] if len(reranked) > 1 else []

        if not best_triple:
            error_msg = "未能从候选三元组中识别出相关的三元组。"
            for chunk in StreamingHelper.generate_sse_chunk(error_msg):
                yield chunk
            return

        # 累积三元组信息
        current_hop_info = {
            "depth": depth,
            "best_triple": best_triple,
            "other_three_triples": other_three_triples
        }
        accumulated_triples.append(current_hop_info)

        # 获取实体信息
        link_ids = [triple['id_'] for triple in best_triple]

        for link_id in link_ids:
            result = await self._get_entity_info_by_link_id(link_id, application_name, link_name)

            if not result:
                continue

            from_entity_desc = result["from_entity_desc"]
            to_entity_desc = result["to_entity_desc"]
            to_entity_id = result["to_entity_id"]

            accumulated_info = TripleUtils.format_accumulated_triples(accumulated_triples)

            prompt = PromptBuilder.build_multi_hop_judgment_prompt(
                question=question,
                best_triple=best_triple,
                other_three_triples=other_three_triples,
                from_entity_desc=from_entity_desc,
                to_entity_desc=to_entity_desc,
                qa_text=qa_text,
                accumulated_info=accumulated_info,
            )

            logger.debug(f"多跳判断 Prompt: {prompt[:200]}...")

            judgment_response = await self.llm_client.chat.completions.create(
                model=settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=5096,
            )

            full_data = None

            async for chunk in self._process_streaming_response(judgment_response):
                yield chunk
                if chunk.startswith('data: {"full_response":'):
                    full_data = json.loads(chunk[len("data: "):])

            if full_data:
                full_response = full_data["full_response"]
                llm_answer = StreamingHelper.extract_llm_answer(full_response)

                logger.debug(f"LLM 回答: {llm_answer[:100]}...")

                tail_entity_id = TripleUtils.convert_description_id_to_entity_id(to_entity_id)

                if "不足" in llm_answer or "无法" in llm_answer:
                    logger.info("当前信息无法回答该问题，需要进一步查询")

                    # 发送跳数信息
                    yield StreamingHelper.generate_hop_info_chunk(depth)

                    async for chunk in self.multi_hop_reasoning(
                        question, query_vector, application_name,
                        link_name, tail_entity_id, depth + 1, max_depth,
                        accumulated_triples
                    ):
                        yield chunk
                else:
                    logger.info("当前信息可以回答该问题")
                return

    async def _process_streaming_response(self, response) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        chat_id = StreamingHelper.generate_chat_id()
        full_response = ""

        async for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield StreamingHelper.generate_sse_chunk(content, chat_id)

        # 发送结束标记
        yield StreamingHelper.generate_end_chunk(chat_id)

        # 发送完整响应
        yield StreamingHelper.generate_full_response(full_response)

    async def _get_entity_info_by_link_id(
        self, link_id: str, application_name: str, link_name: str
    ) -> Optional[Dict]:
        """
        根据 link_id 获取实体信息

        Args:
            link_id: 链接 ID
            application_name: 应用名称
            link_name: 关系集合名称

        Returns:
            包含实体信息的字典
        """
        triple_name = weaviate_service.get_triple_name_by_link_id(link_name, link_id)
        if not triple_name:
            return {
                "from_entity_desc": None,
                "to_entity_desc": None,
                "to_entity_id": None,
            }

        head_entity, tail_entity = TripleUtils.extract_entities_from_triple(triple_name)
        if not head_entity or not tail_entity:
            return {
                "from_entity_desc": None,
                "to_entity_desc": None,
                "to_entity_id": None,
            }

        # 查询头实体信息
        head_entity_vector = embedding_service.encode([head_entity])[0]
        head_entities = weaviate_service.query_entity_info(application_name, head_entity_vector, k=1)
        from_entity_desc = head_entities[0]["desc"] if head_entities else None

        # 查询尾实体信息
        tail_entity_vector = embedding_service.encode([tail_entity])[0]
        tail_entities = weaviate_service.query_entity_info(application_name, tail_entity_vector, k=1)
        to_entity_desc = tail_entities[0]["desc"] if tail_entities else None
        to_entity_id = tail_entities[0]["id_"] if tail_entities else None

        return {
            "from_entity_desc": from_entity_desc,
            "to_entity_desc": to_entity_desc,
            "to_entity_id": to_entity_id,
        }

    async def research_report(
        self,
        messages: List[Dict[str, str]],
        reply_direct: bool = False
    ) -> StreamingResponse:
        """
        研究报告处理

        Args:
            messages: 消息列表
            reply_direct: 是否直接回复

        Returns:
            流式响应
        """
        from datetime import datetime

        content = messages[-1]["content"]
        current_date = datetime.now().date().isoformat()

        if '研究报告' in content:
            prompt = PromptBuilder.build_research_report_prompt(content, current_date)
        elif '作战' in content:
            prompt = PromptBuilder.build_operational_plan_prompt(content, current_date)
        else:
            prompt = content

        if reply_direct:
            prompt += " /no_think"

        logger.info(f"研究报告 Prompt 长度: {len(prompt)}")

        messages_for_llm = [{"role": "system", "content": prompt}]

        async def stream_response():
            try:
                full_response = ""
                response = await self.llm_client.chat.completions.create(
                    model=settings.llm.model,
                    messages=messages_for_llm,
                    stream=True,
                    max_tokens=4096,
                )

                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield StreamingHelper.generate_sse_chunk(content)
                        await asyncio.sleep(0.001)

                # 发送资料来源
                yield StreamingHelper.generate_source_chunk("大模型")

                # 发送完整响应
                yield StreamingHelper.generate_full_response(full_response)

            except Exception as e:
                logger.error(f"研究报告流式错误: {e}")
                yield StreamingHelper.generate_error_chunk(str(e))

        return StreamingResponse(stream_response(), media_type="text/event-stream")


# 全局服务实例
kg_qa_service = KGQAService()