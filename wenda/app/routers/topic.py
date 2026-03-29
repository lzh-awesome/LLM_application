"""话题总结路由 - /summarize_topic"""

import json
import logging
import re

from fastapi import APIRouter, HTTPException, Request

from app.services import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["topic"])

# 摘要提示词模板
SUMMARIZATION_PROMPT = """
你是一名专业的标题总结大师,请根据我的text内容将我的内容进行总结成一句话标题.请结合用户和助手问答,20字以内。
总结内容精准且言简意赅，无需多余的回复和解释,一句话中不要添加任何的标点。
我的text文本为：
%s
"""


@router.post("/summarize_topic")
async def summarize_topic(request: Request):
    """
    会话主题总结接口

    Args:
        request: 包含 text 字段的请求

    Returns:
        dict: 包含 topic 字段的响应
    """
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            raise HTTPException(status_code=400, detail="text 参数不能为空")

        logger.debug(f"Summarizing topic for text length: {len(text)}")

        # 构建提示词
        prompt = SUMMARIZATION_PROMPT % text
        messages = [{"role": "user", "content": prompt}]

        # 调用 LLM
        full_title_response = await llm_service.chat(messages, max_tokens=100)

        # 提取标题
        match = re.search(r'<title>\s*(.*?)\s*</title>', full_title_response, re.DOTALL)

        if not match:
            raise HTTPException(status_code=500, detail="无法提取会话主题")

        topic = match.group(1).strip()
        logger.info(f"Generated topic: {topic}")

        return {"topic": topic}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarize topic error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")