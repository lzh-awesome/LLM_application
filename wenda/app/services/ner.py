"""NER 服务模块 - 实体抽取服务"""

import logging
import time
from typing import List, Optional

import aiohttp
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)


class NERService:
    """实体抽取服务"""

    async def extract_entities(self, text: str) -> List[str]:
        """
        调用外部 NER 服务抽取实体

        Args:
            text: 输入文本

        Returns:
            抽取的实体列表

        Raises:
            HTTPException: NER 服务调用失败
        """
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.llm.ner_url,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=500,
                            detail=f"NER 服务调用失败，状态码: {response.status}"
                        )

                    data = await response.json()
                    entities = data.get("result", [])

                    end_time = time.time()
                    logger.info(f"NER 请求耗时 {end_time - start_time:.2f} 秒")

                    logger.debug(f"抽取的实体: {entities}")
                    return entities

        except aiohttp.ClientError as e:
            end_time = time.time()
            logger.error(f"NER 服务调用异常，耗时 {end_time - start_time:.2f} 秒: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"NER 服务调用异常: {str(e)}"
            )

    def is_no_entity(self, entities: List[str]) -> bool:
        """
        判断是否没有抽取到有效实体

        Args:
            entities: 实体列表

        Returns:
            True 如果没有有效实体
        """
        if not entities:
            return True
        if "无相关XX名" in entities:
            return True
        return False

    def is_multiple_entities(self, entities: List[str]) -> bool:
        """
        判断是否抽取到多个实体

        Args:
            entities: 实体列表

        Returns:
            True 如果有多个实体
        """
        return len(entities) > 1 if entities else False

    def get_first_entity(self, entities: List[str]) -> Optional[str]:
        """
        获取第一个实体

        Args:
            entities: 实体列表

        Returns:
            第一个实体，如果没有则返回 None
        """
        return entities[0] if entities else None


# 全局服务实例
ner_service = NERService()