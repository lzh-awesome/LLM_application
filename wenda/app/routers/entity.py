"""实体标签匹配路由 - /match-tags"""

import logging
from typing import List, Set, Union

from fastapi import APIRouter, HTTPException, Request

from app.services import llm_service, THINK_TAG
from app.utils import process_entity_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["entity"])

# 标签匹配提示词模板
MATCH_TAGS_PROMPT_TEMPLATE = """
你是一个专业的实体标签匹配助手。请根据以下实体信息，从给定的标签列表中选择最匹配的标签。

实体信息：
- 标签: %s
- 描述: %s

可选标签列表: %s

请仔细分析实体的特征，从上面的标签列表中选择所有匹配的标签。
只输出匹配的标签，用逗号分隔，不要其他解释。
如果没有任何标签匹配，输出"无匹配"。
"""


async def match_tags_with_traits(
    entity_data: dict,
    tags: List[str],
) -> Union[List[str], Set[str]]:
    """
    根据实体特征匹配标签

    Args:
        entity_data: 处理后的实体数据
        tags: 可选标签列表

    Returns:
        匹配的标签集合
    """
    labels = entity_data.get("labels", "")
    descs = entity_data.get("descs", "")

    prompt = MATCH_TAGS_PROMPT_TEMPLATE % (labels, descs, ", ".join(tags))
    messages = [{"role": "user", "content": prompt}]

    response = await llm_service.chat(messages, max_tokens=100)

    # 提取思考标签后的内容
    if THINK_TAG in response:
        response = response.split(THINK_TAG, 1)[1]

    # 提取匹配的标签
    matched_tags = [
        tag.strip()
        for tag in response.split(",")
        if tag.strip() in tags
    ]

    if not matched_tags:
        return {"matched_tags": []}

    return set(matched_tags)


@router.post("/match-tags/")
async def match_tags(request: Request):
    """
    实体标签匹配接口

    Args:
        request: 包含 entity 和 tags 字段的请求

    Returns:
        dict: 包含 matched_tags 的响应
    """
    try:
        entity = await request.json()
        tags = entity.get("tags", [])

        logger.debug(f"Entity data: {entity}")
        logger.debug(f"Tags: {tags}")

        if not entity or not tags:
            return {"matched_tags": []}

        # 处理实体数据
        processed_data = process_entity_data(entity)
        logger.debug(f"Processed data: {processed_data}")

        # 匹配标签
        matched_tags = await match_tags_with_traits(processed_data, tags)

        return {"matched_tags": matched_tags}

    except Exception as e:
        logger.error(f"Match tags error: {e}")
        raise HTTPException(status_code=500, detail=f"Tag matching failed: {str(e)}")