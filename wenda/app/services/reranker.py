"""Reranker 服务模块 - 重排序服务"""

import logging
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Reranker 服务类"""

    def __init__(self):
        self._model: Optional[CrossEncoder] = None

    def load_model(self) -> None:
        """加载 reranker 模型"""
        if self._model is not None:
            return

        try:
            self._model = CrossEncoder(
                settings.reranker.model_path,
                max_length=settings.reranker.max_length,
            )
            logger.info(f"Reranker 模型已加载: {settings.reranker.model_path}")
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            raise

    @property
    def model(self) -> CrossEncoder:
        """获取模型实例"""
        if self._model is None:
            self.load_model()
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, any]],
        threshold: float = 0.5,
    ) -> List[Dict[str, any]]:
        """
        使用 CrossEncoder 对候选结果进行重排序

        Args:
            query: 查询文本
            candidates: 候选结果列表，每个元素需包含 "triple_name" 字段
            threshold: 重排序分数阈值，低于此值的结果将被过滤

        Returns:
            重排序后的结果列表，包含 rerank_score 字段
        """
        if not candidates:
            return candidates

        # 构造 passage：拼接 triple_name
        passages = []
        for cand in candidates:
            content = (cand.get("triple_name", "") + "\n").strip()
            passages.append(content)

        # 构造 (query, passage) 对
        pairs = [(query, p) for p in passages]

        # 计算相关性分数
        scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False).tolist()

        # 确保 scores 是 list
        if isinstance(scores, float):
            scores = [scores]

        # 绑定分数
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        # 按 rerank 分数降序排序
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        logger.debug(f"Rerank 完成，最高分数: {candidates[0].get('rerank_score', 0):.3f}")

        return candidates

    def get_high_score_results(
        self,
        query: str,
        candidates: List[Dict[str, any]],
        threshold: float = 0.5,
    ) -> List[Dict[str, any]]:
        """
        获取高于阈值的重排序结果

        Args:
            query: 查询文本
            candidates: 候选结果列表
            threshold: 阈值

        Returns:
            高于阈值的结果列表，如果全部低于阈值则返回第一个结果
        """
        reranked = self.rerank(query, candidates)
        high_score = [item for item in reranked if item.get("rerank_score", 0) > threshold]

        # 如果重排结果低于阈值，返回第一个结果
        if not high_score and reranked:
            high_score = [reranked[0]]

        return high_score


# 全局服务实例
reranker_service = RerankerService()