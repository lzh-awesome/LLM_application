"""
实体集合模块
"""
from typing import List, Dict, Optional
from .base import BaseCollection


class EntityCollection(BaseCollection):
    """实体集合类"""

    COLLECTION_NAME = "kg_wenda"
    DESCRIPTION = "实体集合 - 存储知识图谱实体信息"

    PROPERTIES = [
        {
            "name": "id_",
            "dataType": ["text"],
            "description": "实体ID",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "name",
            "dataType": ["text"],
            "description": "实体名称",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "desc",
            "dataType": ["text"],
            "description": "实体描述",
            "indexFilterable": True,
            "indexSearchable": True
        }
    ]

    def __init__(self, client, collection_name: Optional[str] = None):
        super().__init__(client, collection_name or self.COLLECTION_NAME)

    def add(self, entities: List[Dict], vectors: List[List[float]],
            dedup: bool = True) -> Dict[str, int]:
        """
        添加实体

        Args:
            entities: 实体列表，每个实体应包含 _id, name, desc
            vectors: 向量列表
            dedup: 是否去重

        Returns:
            {"inserted": 数量, "skipped": 数量}
        """
        documents = []
        for entity in entities:
            doc = {
                "id_": str(entity.get("_id", "")),
                "name": str(entity.get("name", "")),
                "desc": str(entity.get("desc", ""))
            }
            documents.append(doc)

        dedup_field = "id_" if dedup else None
        return self.ops.batch_insert(documents, vectors, dedup_field=dedup_field)

    def query(self, vector: List[float], k: int = 10) -> List[Dict]:
        """
        查询实体

        Args:
            vector: 查询向量
            k: 返回数量

        Returns:
            实体列表
        """
        return self.query_by_vector(vector, k)

    def query_by_certainty(self, vector: List[float], min_certainty: float = 0.8) -> List[Dict]:
        """
        按置信度查询实体

        Args:
            vector: 查询向量
            min_certainty: 最小置信度

        Returns:
            实体列表
        """
        return self.ops.query_by_certainty(vector, min_certainty, ["id_", "name", "desc"])

    def delete(self, entity_id: str) -> bool:
        """
        删除实体

        Args:
            entity_id: 实体ID

        Returns:
            是否删除成功
        """
        return self.ops.delete_by_id(entity_id, "id_")

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        根据ID获取实体

        Args:
            entity_id: 实体ID

        Returns:
            实体信息或None
        """
        results = self.ops.query_by_filter("id_", entity_id, ["id_", "name", "desc"], limit=1)
        return results[0] if results else None