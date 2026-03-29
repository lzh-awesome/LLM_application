"""vLLM 多模态服务层 - 封装图片/视频问答逻辑"""

import base64
import io
import logging
from typing import Optional

from openai import AsyncOpenAI
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class VLLMService:
    """vLLM 多模态服务类"""

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=settings.vllm.api_base,
                api_key=settings.vllm.api_key,
            )
        return self._client

    @property
    def model(self) -> str:
        """获取模型名称"""
        return settings.vllm.model

    def image_to_base64_data_url(self, image_bytes: bytes, image_format: str = "JPEG") -> str:
        """
        将图片字节转换为 base64 编码的 data URL

        Args:
            image_bytes: 图片字节数据
            image_format: 输出图片格式

        Returns:
            base64 data URL 字符串
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # 转换为 RGB 格式（防止 alpha 通道问题）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffered = io.BytesIO()
            img.save(buffered, format=image_format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return f"data:image/{image_format.lower()};base64,{img_str}"
        except Exception as e:
            raise ValueError(f"图片处理失败: {e}")

    async def image_qa(self, image_bytes: bytes, question: str, max_tokens: int = 4096) -> str:
        """
        图片问答

        Args:
            image_bytes: 图片字节数据
            question: 用户问题
            max_tokens: 最大生成 token 数

        Returns:
            模型回答
        """
        try:
            b64_image_url = self.image_to_base64_data_url(image_bytes)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": b64_image_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"图片问答失败: {e}")
            raise

    async def video_qa(
        self,
        video_bytes: bytes,
        mime_type: str,
        question: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """
        视频问答

        Args:
            video_bytes: 视频字节数据
            mime_type: 视频 MIME 类型
            question: 用户问题
            max_tokens: 最大生成 token 数
            temperature: 生成温度

        Returns:
            模型回答
        """
        try:
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            data_url = f"data:{mime_type};base64,{video_base64}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "video_url", "video_url": {"url": data_url}},
                    ],
                }
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"stop_token_ids": [151645, 151643]},
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"视频问答失败: {e}")
            raise


# 全局服务实例
vllm_service = VLLMService()