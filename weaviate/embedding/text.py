"""
文本向量化模块
使用 BGE-M3 模型进行文本向量化
"""
from typing import List, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class TextEmbedding:
    """文本向量化类"""

    def __init__(self, model_path: str):
        """
        初始化文本向量化模型

        Args:
            model_path: BGE-M3 模型路径
        """
        self.model_path = model_path
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """延迟加载模型"""
        if self._model is None:
            print(f"正在加载文本向量化模型: {self.model_path}")
            self._model = SentenceTransformer(self.model_path)
        return self._model

    def encode(self, texts: List[str], batch_size: int = 1000,
               show_progress: bool = True) -> List[List[float]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            向量列表
        """
        embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="文本向量化", total=total_batches)

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def encode_single(self, text: str) -> List[float]:
        """
        单个文本向量化

        Args:
            text: 文本

        Returns:
            向量
        """
        return self.model.encode([text])[0].tolist()

    def encode_with_metadata(self, texts: List[str], batch_size: int = 1000) -> List[dict]:
        """
        批量文本向量化，返回包含元数据的结果

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            包含文本和向量的字典列表
        """
        vectors = self.encode(texts, batch_size)
        return [
            {"text": text, "vector": vector.tolist() if hasattr(vector, 'tolist') else vector}
            for text, vector in zip(texts, vectors)
        ]