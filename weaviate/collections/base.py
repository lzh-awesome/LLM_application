"""
集合基类模块
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.operations import VectorStoreOperations


class BaseCollection(ABC):
    """集合基类"""

    # 子类需要定义的属性
    COLLECTION_NAME: str = ""
    PROPERTIES: List[Dict] = []
    DESCRIPTION: str = ""

    def __init__(self, client, collection_name: Optional[str] = None):
        """
        初始化集合

        Args:
            client: Weaviate 客户端
            collection_name: 集合名称（可选，默认使用类属性）
        """
        self.client = client
        self.collection_name = collection_name or self.COLLECTION_NAME
        self._ops: Optional[VectorStoreOperations] = None

    @property
    def ops(self) -> VectorStoreOperations:
        """获取操作实例"""
        if self._ops is None:
            self._ops = VectorStoreOperations(self.client, self.collection_name)
        return self._ops

    def create_collection(self) -> bool:
        """创建集合"""
        if self.ops.exists():
            print(f"⚠️ 集合 '{self.collection_name}' 已存在")
            return False
        return self.ops.create(self.PROPERTIES, self.DESCRIPTION)

    def delete_collection(self) -> bool:
        """删除集合"""
        return self.ops.delete()

    def collection_exists(self) -> bool:
        """检查集合是否存在"""
        return self.ops.exists()

    @abstractmethod
    def add(self, items: List[Dict], vectors: List[List[float]], **kwargs) -> Dict[str, int]:
        """
        添加数据

        Args:
            items: 数据列表
            vectors: 向量列表

        Returns:
            {"inserted": 数量, "skipped": 数量}
        """
        pass

    @abstractmethod
    def query(self, vector: List[float], **kwargs) -> List[Dict]:
        """
        查询数据

        Args:
            vector: 查询向量

        Returns:
            查询结果列表
        """
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """
        删除数据

        Args:
            item_id: 数据ID

        Returns:
            是否删除成功
        """
        pass

    def query_by_vector(self, vector: List[float], k: int = 10) -> List[Dict]:
        """
        向量相似度查询

        Args:
            vector: 查询向量
            k: 返回数量

        Returns:
            查询结果列表
        """
        return self.ops.query_by_vector(vector, k, self._get_return_properties())

    def query_by_certainty(self, vector: List[float], min_certainty: float = 0.8) -> List[Dict]:
        """
        按置信度阈值查询

        Args:
            vector: 查询向量
            min_certainty: 最小置信度

        Returns:
            查询结果列表
        """
        return self.ops.query_by_certainty(vector, min_certainty, self._get_return_properties())

    def _get_return_properties(self) -> List[str]:
        """获取返回的属性列表"""
        return [p["name"] for p in self.PROPERTIES]