"""
视频集合模块
"""
from typing import List, Dict, Optional
from .base import BaseCollection


class VideoCollection(BaseCollection):
    """视频集合类"""

    COLLECTION_NAME = "video_retrieval"
    DESCRIPTION = "视频集合 - 存储视频向量检索信息"

    PROPERTIES = [
        {
            "name": "video_name",
            "dataType": ["text"],
            "description": "视频名称/类别",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "video_base64",
            "dataType": ["text"],
            "description": "视频Base64编码",
            "indexFilterable": False,
            "indexSearchable": False
        },
        {
            "name": "video_md5",
            "dataType": ["text"],
            "description": "视频MD5哈希值",
            "indexFilterable": True,
            "indexSearchable": False
        }
    ]

    def __init__(self, client, collection_name: Optional[str] = None):
        super().__init__(client, collection_name or self.COLLECTION_NAME)

    def add(self, videos: List[Dict], vectors: List[List[float]],
            dedup: bool = True) -> Dict[str, int]:
        """
        添加视频

        Args:
            videos: 视频信息列表，每个应包含 video_name, video_base64, video_md5
            vectors: 向量列表
            dedup: 是否去重（基于MD5）

        Returns:
            {"inserted": 数量, "skipped": 数量}
        """
        documents = []
        for video in videos:
            doc = {
                "video_name": str(video.get("video_name", "")),
                "video_base64": str(video.get("video_base64", "")),
                "video_md5": str(video.get("video_md5", ""))
            }
            documents.append(doc)

        dedup_field = "video_md5" if dedup else None
        return self.ops.batch_insert(documents, vectors, dedup_field=dedup_field)

    def query(self, vector: List[float], k: int = 10) -> List[Dict]:
        """
        查询视频

        Args:
            vector: 查询向量
            k: 返回数量

        Returns:
            视频列表
        """
        return self.query_by_vector(vector, k)

    def delete(self, md5: str) -> bool:
        """
        根据MD5删除视频

        Args:
            md5: 视频MD5值

        Returns:
            是否删除成功
        """
        return self.ops.delete_by_id(md5, "video_md5")

    def exists_by_md5(self, md5: str) -> bool:
        """
        检查MD5是否已存在

        Args:
            md5: 视频MD5值

        Returns:
            是否存在
        """
        return self.ops._exists_by_field("video_md5", md5)


class VideoNameCollection(BaseCollection):
    """视频名称集合类（用于文搜视频）"""

    COLLECTION_NAME = "video_name_retrieval"
    DESCRIPTION = "视频名称集合 - 用于文本搜索视频"

    PROPERTIES = [
        {
            "name": "video_name",
            "dataType": ["text"],
            "description": "视频名称",
            "indexFilterable": True,
            "indexSearchable": True
        },
        {
            "name": "video_base64",
            "dataType": ["text"],
            "description": "视频Base64编码",
            "indexFilterable": False,
            "indexSearchable": False
        },
        {
            "name": "video_md5",
            "dataType": ["text"],
            "description": "视频MD5哈希值",
            "indexFilterable": True,
            "indexSearchable": False
        }
    ]

    def __init__(self, client, collection_name: Optional[str] = None):
        super().__init__(client, collection_name or self.COLLECTION_NAME)

    def add(self, videos: List[Dict], vectors: List[List[float]],
            dedup: bool = True) -> Dict[str, int]:
        """添加视频（用于文搜视频）"""
        documents = []
        for video in videos:
            doc = {
                "video_name": str(video.get("video_name", "")),
                "video_base64": str(video.get("video_base64", "")),
                "video_md5": str(video.get("video_md5", ""))
            }
            documents.append(doc)

        dedup_field = "video_md5" if dedup else None
        return self.ops.batch_insert(documents, vectors, dedup_field=dedup_field)

    def query(self, vector: List[float], k: int = 10) -> List[Dict]:
        """查询视频（文搜视频）"""
        return self.query_by_vector(vector, k)

    def delete(self, md5: str) -> bool:
        """根据MD5删除"""
        return self.ops.delete_by_id(md5, "video_md5")