"""问题推荐路由 - /recommend_questions"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request

from app.services import llm_service, THINK_TAG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["recommend"])

# 问题推荐提示词模板
RECOMMEND_PROMPT_TEMPLATE = """
用户的问题：%s

请根据用户的问题，生成3个相关的推荐问题，帮助用户深入了解该主题。
要求：
1. 推荐问题应该与原问题相关但角度不同
2. 每个问题用换行分隔
3. 问题要简洁明了
4. 只输出问题列表，不要其他解释

推荐问题：
"""


@router.post("/recommend_questions")
async def recommend_questions(request: Request):
    """
    问题推荐接口

    Args:
        request: 包含 text 字段的请求

    Returns:
        dict: 包含 questions 列表的响应
    """
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            raise HTTPException(status_code=400, detail="text 参数不能为空")

        logger.debug(f"Generating recommendations for: {text[:50]}...")

        # 构建提示词
        prompt = RECOMMEND_PROMPT_TEMPLATE % text
        messages = [{"role": "user", "content": prompt}]

        # 调用 LLM
        response = await llm_service.chat(messages, max_tokens=200)

        # 提取思考标签后的内容
        if THINK_TAG in response:
            response = response.split(THINK_TAG, 1)[1]

        # 解析推荐问题
        questions = [
            q.strip()
            for q in response.strip().split("\n")
            if q.strip() and not q.strip().startswith(("#", "*", "-"))
        ]

        logger.info(f"Generated {len(questions)} recommended questions")

        return {"questions": questions[:3]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommend questions error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")