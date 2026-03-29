"""翻译路由 - /translate-to-en"""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.services import llm_service, THINK_TAG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["translate"])

# 翻译提示词模板
TRANSLATE_PROMPT_TEMPLATE = """
请将以下中文文本翻译成英文，只输出翻译结果，不要其他解释：

%s
"""


@router.post("/translate-to-en")
async def translate_to_english(request: Request):
    """
    中文翻译英文接口

    Args:
        request: 包含 text 字段的请求

    Returns:
        dict: 包含 translation 字段的响应
    """
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            raise HTTPException(status_code=400, detail="text 参数不能为空")

        logger.debug(f"Translating text length: {len(text)}")

        # 构建提示词
        prompt = TRANSLATE_PROMPT_TEMPLATE % text
        messages = [{"role": "user", "content": prompt}]

        # 调用 LLM
        translation = await llm_service.chat(messages, max_tokens=1000)

        # 提取思考标签后的内容
        if THINK_TAG in translation:
            translation = translation.split(THINK_TAG, 1)[1]

        logger.info(f"Translation completed, length: {len(translation)}")

        return {"translation": translation}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")