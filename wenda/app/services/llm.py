"""LLM 服务层 - 封装 LLM 调用逻辑"""

import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# 思考标签 - markdown 代码块结束符 + 两个换行符
THINK_TAG = "```" + "\n\n"


class LLMService:
    """LLM 服务类"""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=settings.llm.base_url,
                api_key=settings.llm.api_key,
            )
        return self._client

    @staticmethod
    def extract_think_content(response: str, tag: str = THINK_TAG) -> str:
        """提取 tag 后的内容"""
        if tag in response:
            return response.split(tag, 1)[1]
        return response

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """使用 OpenAI 兼容客户端进行流式聊天补全"""
        model = model_name or settings.llm.model
        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM Stream Error: {e}")
            yield json.dumps({"error": str(e)})

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> str:
        """使用 OpenAI 兼容客户端进行非流式聊天补全"""
        model = model_name or settings.llm.model
        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            raise Exception(f"LLM call failed: {e}")

    async def stream_ollama(
        self,
        prompt: str,
        model_name: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """调用 Ollama API 进行流式生成"""
        model = model_name or settings.llm.model
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.llm.ollama_api_url, json=payload
                ) as response:
                    if response.status != 200:
                        error_msg = f"请求失败: 状态码 {response.status}"
                        logger.error(error_msg)
                        yield json.dumps({"error": error_msg})
                        return

                    async for line in response.content:
                        try:
                            data = line.decode("utf-8").strip()
                            chunk = json.loads(data)
                            yield chunk.get("response", "")
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Ollama Stream Error: {e}")
            yield json.dumps({"error": str(e)})

    async def ollama_chat(
        self,
        prompt: str,
        model_name: Optional[str] = None,
    ) -> str:
        """调用 Ollama API 进行非流式生成"""
        model = model_name or settings.llm.model
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 50,
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.llm.ollama_api_url, json=payload
                ) as response:
                    if response.status != 200:
                        error_msg = f"请求失败: 状态码 {response.status}"
                        raise Exception(error_msg)
                    data = await response.json()
                    return data.get("response", "")
        except Exception as e:
            raise Exception(f"请求异常: {str(e)}")


# 全局服务实例
llm_service = LLMService()
