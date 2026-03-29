"""数据库服务层 - ArangoDB 和 Weaviate 操作封装"""

import logging
from typing import Any, Dict, List, Optional

import weaviate
import weaviate.classes as wvc
from arango import ArangoClient
from weaviate.auth import AuthApiKey

from app.config import settings

logger = logging.getLogger(__name__)


class ArangoService:
    """ArangoDB 服务类"""

    def __init__(self):
        self._client: Optional[ArangoClient] = None
        self._db = None

    def connect(self) -> None:
        """建立连接"""
        if self._client is None:
            self._client = ArangoClient(hosts=settings.arango.host)
            self._db = self._client.db(
                settings.arango.database_name,
                username=settings.arango.username,
                password=settings.arango.password,
            )
            logger.info(f"ArangoDB 连接成功: {settings.arango.host}")

    def disconnect(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("ArangoDB 连接已关闭")

    @property# 装饰器，用于将方法转换为只读属性
    def db(self):
        """获取数据库实例"""
        if self._db is None:
            self.connect()
        return self._db

    def search_entity_by_zh_cn_label(self, entity: str) -> Optional[Dict]:
        """
        根据中文名称搜索实体

        Args:
            entity: 实体名称

        Returns:
            实体数据字典，未找到返回 None
        """
        try:
            if not self.db.has_collection(settings.arango.entity_collection):
                raise ValueError(
                    f"集合 '{settings.arango.entity_collection}' 不存在于数据库"
                )

            aql_query = """
            FOR doc IN @@collection_name
                LET label_zh_cn = doc.labels['zh-cn']
                LET label_zh = doc.labels['zh']
                LET label_en = doc.labels['en']

                FILTER label_zh_cn == @entity
                    OR (label_zh_cn == null AND label_zh == @entity)
                    OR (label_zh_cn == null AND label_zh == null AND label_en == @entity)
                RETURN doc
            """
            bind_vars = {
                "@collection_name": settings.arango.entity_collection,
                "entity": entity,
            }

            cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
            results = list(cursor)

            return results[0] if results else None

        except Exception as e:
            logger.error(f"ArangoDB 查询失败: {e}")
            raise

    def get_label_value_from_arango(self, item_label: str) -> Optional[str]:
        """
        从 ArangoDB 获取标签值

        Args:
            item_label: 标签名称

        Returns:
            标签值，未找到返回 None
        """
        try:
            query = """
            FOR doc IN @@collection
                FILTER doc.label == @item_label
                RETURN doc.label
            """
            bind_vars = {
                "@collection": settings.arango.entity_collection,
                "item_label": item_label,
            }
            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            results = list(cursor)
            return results[0] if results else None
        except Exception as e:
            logger.error(f"ArangoDB 标签查询失败: {e}")
            return None


class WeaviateService:
    """Weaviate 服务类"""

    def __init__(self):
        self._client = None

    def connect(self) -> None:
        """建立连接"""
        if self._client is None:
            self._client = weaviate.connect_to_custom(
                http_host=settings.weaviate.http_host,
                http_port=settings.weaviate.http_port,
                http_secure=False,
                grpc_host=settings.weaviate.grpc_host,
                grpc_port=settings.weaviate.grpc_port,
                grpc_secure=False,
                auth_credentials=AuthApiKey(settings.weaviate.api_key),
            )
            logger.info("Weaviate 连接成功")

    def disconnect(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Weaviate 连接已关闭")

    @property
    def client(self):
        """获取客户端实例"""
        if self._client is None:
            self.connect()
        return self._client

    def query_triples(self, collection_name: str, vector, k: int = 20) -> List[Dict]:
        """
        向量查询三元组

        Args:
            collection_name: 集合名称
            vector: 查询向量
            k: 返回数量

        Returns:
            三元组结果列表
        """
        collection = self.client.collections.get(collection_name)

        try:
            response = collection.query.near_vector(
                near_vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
                limit=k,
                return_properties=["triple_name", "id_"],
                return_metadata=["distance", "certainty"],
            )

            results = []
            for obj in response.objects:
                results.append({
                    "triple_name": obj.properties.get("triple_name"),
                    "id_": obj.properties.get("id_"),
                    "certainty": obj.metadata.certainty,
                })

            return results
        except Exception as e:
            logger.error(f"查询三元组异常: {e}")
            return []

    def query_qa_collection(self, collection_name: str, vector, k: int = 3) -> List[Dict]:
        """
        向量查询问答对

        Args:
            collection_name: 集合名称
            vector: 查询向量
            k: 返回数量

        Returns:
            问答对结果列表
        """
        collection = self.client.collections.get(collection_name)

        try:
            response = collection.query.near_vector(
                near_vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
                limit=k,
                return_properties=["query", "answer"],
                return_metadata=["distance", "certainty"],
            )

            results = []
            for obj in response.objects:
                results.append({
                    "query": obj.properties.get("query"),
                    "answer": obj.properties.get("answer"),
                    "certainty": obj.metadata.certainty,
                })

            return results
        except Exception as e:
            logger.error(f"查询问答对异常: {e}")
            return []

    def query_entity_info(self, collection_name: str, vector, k: int = 5) -> List[Dict]:
        """
        向量查询实体信息

        Args:
            collection_name: 集合名称
            vector: 查询向量
            k: 返回数量

        Returns:
            实体信息结果列表
        """
        collection = self.client.collections.get(collection_name)

        try:
            response = collection.query.near_vector(
                near_vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
                limit=k,
                return_properties=["id_", "name", "desc"],
                return_metadata=["distance", "certainty"],
            )

            results = []
            for obj in response.objects:
                results.append({
                    "id_": obj.properties.get("id_"),
                    "name": obj.properties.get("name"),
                    "desc": obj.properties.get("desc"),
                    "certainty": obj.metadata.certainty,
                })

            return results
        except Exception as e:
            logger.error(f"查询实体信息异常: {e}")
            return []

    def query_triples_by_entity_id(
        self, collection_name: str, entity_id: str, k: int = 15
    ) -> List[Dict]:
        """
        根据头实体 ID 查询三元组

        Args:
            collection_name: 集合名称
            entity_id: 头实体 ID
            k: 返回数量

        Returns:
            三元组结果列表
        """
        collection = self.client.collections.get(collection_name)

        try:
            response = collection.query.fetch_objects(
                filters=wvc.query.Filter.by_property("_from").equal(entity_id),
                limit=k,
                return_properties=["triple_name", "id_", "_from", "_to"],
                return_metadata=["distance", "certainty"],
            )

            results = []
            for obj in response.objects:
                results.append({
                    "triple_name": obj.properties.get("triple_name"),
                    "id_": obj.properties.get("id_"),
                    "_from": obj.properties.get("_from"),
                    "_to": obj.properties.get("_to"),
                    "certainty": obj.metadata.certainty,
                })

            return results
        except Exception as e:
            logger.error(f"根据实体 ID 查询三元组异常: {e}")
            return []

    def get_triple_name_by_link_id(
        self, collection_name: str, link_id: str
    ) -> Optional[str]:
        """
        根据 link_id 获取 triple_name

        Args:
            collection_name: 集合名称
            link_id: link 的 ID

        Returns:
            triple_name 或 None
        """
        collection = self.client.collections.get(collection_name)

        try:
            response = collection.query.fetch_objects(
                filters=wvc.query.Filter.by_property("id_").equal(link_id),
                return_properties=["triple_name", "id_"],
                limit=1,
            )

            if response.objects:
                obj = response.objects[0]
                return obj.properties.get("triple_name")
            else:
                logger.warning(f"未找到 ID 为 {link_id} 的记录")
                return None

        except Exception as e:
            logger.error(f"查询 triple_name 异常: {e}")
            return None


# 全局服务实例
arango_service = ArangoService()
weaviate_service = WeaviateService()