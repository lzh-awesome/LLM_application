"""Embedding 服务层 - 模型加载和 GPU 管理"""

import logging
from typing import Optional

import pynvml
import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


def get_lowest_memory_gpu() -> Optional[int]:
    """
    获取显存使用最低的 GPU ID

    Returns:
        GPU ID，如果没有可用 GPU 返回 None
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            logger.info("未检测到 GPU，将使用 CPU")
            return None

        min_memory = float("inf")
        best_gpu_id = 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_memory = memory_info.used

            logger.debug(f"GPU {i}: 已使用显存 {used_memory / 1024 / 1024:.2f} MB")

            if used_memory < min_memory:
                min_memory = used_memory
                best_gpu_id = i

        logger.info(f"选择 GPU {best_gpu_id} (显存使用最低)")
        return best_gpu_id

    except Exception as e:
        logger.warning(f"获取 GPU 信息时出错: {e}")
        return None


class EmbeddingService:
    """Embedding 服务类"""

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._device: Optional[str] = None

    def load_model(self) -> None:
        """加载 embedding 模型"""
        if self._model is not None:
            return

        # 选择设备
        best_gpu_id = get_lowest_memory_gpu()
        if best_gpu_id is not None and torch.cuda.is_available():
            self._device = f"cuda:{best_gpu_id}"
        else:
            self._device = "cpu"

        # 加载模型
        self._model = SentenceTransformer(
            settings.embedding_model.bge_m3_path,
            device=self._device,
        )
        logger.info(f"Embedding 模型已加载到设备: {self._device}")

    @property
    def model(self) -> SentenceTransformer:
        """获取模型实例"""
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def device(self) -> str:
        """获取当前设备"""
        if self._device is None:
            self.load_model()
        return self._device

    def encode(self, texts: list, **kwargs):
        """
        编码文本为向量

        Args:
            texts: 文本列表
            **kwargs: 传递给模型的参数

        Returns:
            向量数组
        """
        return self.model.encode(texts, **kwargs)


# 全局服务实例
embedding_service = EmbeddingService()