"""
图片集合模块
"""
from typing import List, Dict, Optional
from .base import BaseCollection


class ImageCollection(BaseCollection):
    """图片集合类"""

    COLLECTION_NAME = "image_retrieval"
    DESCRIPTION = "图片集合 - 存储图片向量检索信息"

    PROPERTIES = [
        {
            "name": "image_name",
            "dataType": ["text"],
            "description": "图片名称/类别",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "image_id",
            "dataType": ["text"],
            "description": "图片ID",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "image_base64",
            "dataType": ["text"],
            "description": "图片Base64编码",
            "indexFilterable": False,
            "indexSearchable": False
        },
        {
            "name": "image_md5",
            "dataType": ["text"],
            "description": "图片MD5哈希值",
            "indexFilterable": True,
            "indexSearchable": False
        }
    ]

    def __init__(self, client, collection_name: Optional[str] = None):
        super().__init__(client, collection_name or self.COLLECTION_NAME)

    def add(self, images: List[Dict], vectors: List[List[float]],
            dedup: bool = True) -> Dict[str, int]:
        """
        添加图片

        Args:
            images: 图片信息列表，每个应包含 image_name, image_id, image_base64, image_md5
            vectors: 向量列表
            dedup: 是否去重（基于MD5）

        Returns:
            {"inserted": 数量, "skipped": 数量}
        """
        documents = []
        for image in images:
            doc = {
                "image_name": str(image.get("image_name", "")),
                "image_id": str(image.get("image_id", "")),
                "image_base64": str(image.get("image_base64", "")),
                "image_md5": str(image.get("image_md5", ""))
            }
            documents.append(doc)

        dedup_field = "image_md5" if dedup else None
        return self.ops.batch_insert(documents, vectors, dedup_field=dedup_field)

    def query(self, vector: List[float], k: int = 10) -> List[Dict]:
        """
        查询图片

        Args:
            vector: 查询向量
            k: 返回数量

        Returns:
            图片列表
        """
        return self.query_by_vector(vector, k)

    def delete(self, image_id: str) -> bool:
        """
        删除图片

        Args:
            image_id: 图片ID

        Returns:
            是否删除成功
        """
        return self.ops.delete_by_id(image_id, "image_id")

    def delete_by_md5(self, md5: str) -> bool:
        """
        根据MD5删除图片

        Args:
            md5: 图片MD5值

        Returns:
            是否删除成功
        """
        return self.ops.delete_by_id(md5, "image_md5")

    def get_image_by_id(self, image_id: str) -> Optional[Dict]:
        """
        根据ID获取图片

        Args:
            image_id: 图片ID

        Returns:
            图片信息或None
        """
        results = self.ops.query_by_filter("image_id", image_id, limit=1)
        return results[0] if results else None

    def exists_by_md5(self, md5: str) -> bool:
        """
        检查MD5是否已存在

        Args:
            md5: 图片MD5值

        Returns:
            是否存在
        """
        return self.ops._exists_by_field("image_md5", md5)


class ImageNameCollection(BaseCollection):
    """图片名称集合类（用于文搜图）"""

    COLLECTION_NAME = "image_name_retrieval"
    DESCRIPTION = "图片名称集合 - 用于文本搜索图片"

    PROPERTIES = [
        {
            "name": "image_name",
            "dataType": ["text"],
            "description": "图片名称",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "image_base64",
            "dataType": ["text"],
            "description": "图片Base64编码",
            "indexFilterable": False,
            "indexSearchable": False
        },
        {
            "name": "image_md5",
            "dataType": ["text"],
            "description": "图片MD5哈希值",
            "indexFilterable": True,
            "indexSearchable": False
        }
    ]

    def __init__(self, client, collection_name: Optional[str] = None):
        super().__init__(client, collection_name or self.COLLECTION_NAME)

    def add(self, images: List[Dict], vectors: List[List[float]],
            dedup: bool = True) -> Dict[str, int]:
        """添加图片（用于文搜图）"""
        documents = []
        for image in images:
            doc = {
                "image_name": str(image.get("image_name", "")),
                "image_base64": str(image.get("image_base64", "")),
                "image_md5": str(image.get("image_md5", ""))
            }
            documents.append(doc)

        dedup_field = "image_md5" if dedup else None
        return self.ops.batch_insert(documents, vectors, dedup_field=dedup_field)

    def query(self, vector: List[float], k: int = 10) -> List[Dict]:
        """查询图片（文搜图）"""
        return self.query_by_vector(vector, k)

    def delete(self, md5: str) -> bool:
        """根据MD5删除"""
        return self.ops.delete_by_id(md5, "image_md5")