"""
客户端管理模块
提供 Weaviate 和 ArangoDB 的连接管理
"""
from typing import Optional
import weaviate
from weaviate.auth import AuthApiKey
from arango import ArangoClient

from config.settings import WeaviateConfig, ArangoConfig


class WeaviateClient:
    """Weaviate 客户端单例管理器"""
    _instance: Optional["WeaviateClient"] = None
    _client: Optional[weaviate.WeaviateClient] = None

    def __new__(cls, config: Optional[WeaviateConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[WeaviateConfig] = None):
        if self._client is None and config is not None:
            self._connect(config)

    def _connect(self, config: WeaviateConfig):
        """建立连接"""
        self._client = weaviate.connect_to_custom(
            http_host=config.http_host,
            http_port=config.http_port,
            http_secure=config.http_secure,
            grpc_host=config.grpc_host,
            grpc_port=config.grpc_port,
            grpc_secure=config.grpc_secure,
            auth_credentials=AuthApiKey(config.api_key)
        )

    @property
    def client(self) -> weaviate.WeaviateClient:
        """获取客户端实例"""
        if self._client is None:
            raise RuntimeError("Weaviate 客户端未初始化，请先调用 connect() 方法")
        return self._client

    def connect(self, config: WeaviateConfig):
        """连接到 Weaviate"""
        if self._client is not None:
            self.close()
        self._connect(config)

    def close(self):
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._client is not None

    def list_collections(self) -> list:
        """列出所有集合"""
        return list(self.client.collections.list_all().keys())

    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        return collection_name in self.client.collections.list_all()

    @classmethod
    def get_instance(cls) -> "WeaviateClient":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """重置单例（主要用于测试）"""
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None
        cls._client = None


class ArangoClientManager:
    """ArangoDB 客户端管理器"""

    @staticmethod
    def connect(config: ArangoConfig):
        """连接到 ArangoDB"""
        client = ArangoClient(hosts=config.host)
        return client.db(config.database, username=config.username, password=config.password)

    @staticmethod
    def connect_to_database(config: ArangoConfig, database: str):
        """连接到指定数据库"""
        client = ArangoClient(hosts=config.host)
        return client.db(database, username=config.username, password=config.password)