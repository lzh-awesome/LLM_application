"""
关系（三元组）集合模块
"""
from typing import List, Dict, Optional
from .base import BaseCollection


class LinkCollection(BaseCollection):
    """关系（三元组）集合类"""

    COLLECTION_NAME = "natural_triples"
    DESCRIPTION = "关系集合 - 存储知识图谱三元组信息"

    PROPERTIES = [
        {
            "name": "id_",
            "dataType": ["text"],
            "description": "关系ID",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "triple_name",
            "dataType": ["text"],
            "description": "三元组自然语言描述",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "_from",
            "dataType": ["text"],
            "description": "头实体ID",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "_to",
            "dataType": ["text"],
            "description": "尾实体ID",
            "indexFilterable": True,
            "indexSearchable": True
        }
    ]

    def __init__(self, client, collection_name: Optional[str] = None):
        super().__init__(client, collection_name or self.COLLECTION_NAME)

    def add(self, triples: List[Dict], vectors: List[List[float]],
            dedup: bool = True) -> Dict[str, int]:
        """
        添加三元组

        Args:
            triples: 三元组列表，每个三元组应包含 _id, triple_name, _from, _to
            vectors: 向量列表
            dedup: 是否去重

        Returns:
            {"inserted": 数量, "skipped": 数量}
        """
        documents = []
        for triple in triples:
            doc = {
                "id_": str(triple.get("_id", "")),
                "triple_name": str(triple.get("triple_name", "")),
                "_from": str(triple.get("_from", "")),
                "_to": str(triple.get("_to", ""))
            }
            documents.append(doc)

        dedup_field = "id_" if dedup else None
        return self.ops.batch_insert(documents, vectors, dedup_field=dedup_field)

    def query(self, vector: List[float], k: int = 10) -> List[Dict]:
        """
        查询三元组

        Args:
            vector: 查询向量
            k: 返回数量

        Returns:
            三元组列表
        """
        return self.query_by_vector(vector, k)

    def delete(self, triple_id: str) -> bool:
        """
        删除三元组

        Args:
            triple_id: 三元组ID

        Returns:
            是否删除成功
        """
        return self.ops.delete_by_id(triple_id, "id_")

    def get_triples_by_entity(self, entity_id: str) -> List[Dict]:
        """
        根据实体ID获取相关的三元组

        Args:
            entity_id: 实体ID

        Returns:
            三元组列表
        """
        # 查询头实体或尾实体为指定实体的三元组
        results = []
        from_results = self.ops.query_by_filter("_from", entity_id, ["id_", "triple_name", "_from", "_to"])
        to_results = self.ops.query_by_filter("_to", entity_id, ["id_", "triple_name", "_from", "_to"])
        results.extend(from_results)
        results.extend(to_results)
        return results

    @staticmethod
    def format_triple(head: str, relation: str, tail: str) -> str:
        """
        格式化三元组为自然语言

        Args:
            head: 头实体名称
            relation: 关系名称
            tail: 尾实体名称

        Returns:
            自然语言描述
        """
        return f"{head}的{relation}是{tail}"