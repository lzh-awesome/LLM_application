"""
通用向量库 CRUD 操作模块
"""
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import weaviate
from weaviate.classes.query import Filter


class VectorStoreOperations:
    """向量库通用操作类"""

    def __init__(self, client: weaviate.WeaviateClient, collection_name: str):
        """
        初始化操作类

        Args:
            client: Weaviate 客户端
            collection_name: 集合名称
        """
        self.client = client
        self.collection_name = collection_name
        self._collection = None

    @property
    def collection(self):
        """获取集合实例"""
        if self._collection is None:
            self._collection = self.client.collections.get(self.collection_name)
        return self._collection

    # ==================== 集合管理 ====================

    def exists(self) -> bool:
        """检查集合是否存在"""
        return self.collection_name in self.client.collections.list_all()

    def create(self, properties: List[Dict], description: str = "",
               distance: str = "cosine", ef_construction: int = 200,
               max_connections: int = 64) -> bool:
        """
        创建集合

        Args:
            properties: 属性定义列表
            description: 集合描述
            distance: 距离度量 (cosine, l2-squared, dot)
            ef_construction: HNSW 构建参数
            max_connections: HNSW 最大连接数

        Returns:
            是否创建成功
        """
        collection_obj = {
            "class": self.collection_name,
            "description": description,
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": distance,
                "efConstruction": ef_construction,
                "maxConnections": max_connections
            },
            "properties": properties
        }

        try:
            self.client.collections.create_from_dict(collection_obj)
            print(f"✅ 集合 '{self.collection_name}' 创建成功")
            return True
        except Exception as e:
            print(f"❌ 创建集合 '{self.collection_name}' 失败: {e}")
            return False

    def delete(self) -> bool:
        """删除集合"""
        try:
            self.client.collections.delete(self.collection_name)
            self._collection = None
            print(f"🗑️ 集合 '{self.collection_name}' 删除成功")
            return True
        except Exception as e:
            print(f"❌ 删除集合 '{self.collection_name}' 失败: {e}")
            return False

    # ==================== 数据插入 ====================

    def insert(self, properties: Dict[str, Any], vector: List[float]) -> Optional[str]:
        """
        插入单个文档

        Args:
            properties: 文档属性
            vector: 向量

        Returns:
            UUID 或 None
        """
        try:
            uuid = self.collection.data.insert(properties=properties, vector=vector)
            return str(uuid)
        except Exception as e:
            print(f"❌ 插入文档失败: {e}")
            return None

    def batch_insert(self, documents: List[Dict], vectors: List[List[float]],
                     dedup_field: Optional[str] = None, show_progress: bool = True) -> Dict[str, int]:
        """
        批量插入文档

        Args:
            documents: 文档列表
            vectors: 向量列表
            dedup_field: 去重字段（如果指定，将检查该字段是否已存在）
            show_progress: 是否显示进度条

        Returns:
            {"inserted": 插入数量, "skipped": 跳过数量}
        """
        inserted, skipped = 0, 0
        iterator = zip(documents, vectors)

        if show_progress:
            iterator = tqdm(list(iterator), desc=f"插入到 {self.collection_name}")

        for doc, vec in iterator:
            if dedup_field and dedup_field in doc:
                if self._exists_by_field(dedup_field, doc[dedup_field]):
                    skipped += 1
                    continue

            uuid = self.insert(doc, vec)
            if uuid:
                inserted += 1
            else:
                skipped += 1

        print(f"✅ 插入完成: {inserted} 条, 跳过: {skipped} 条")
        return {"inserted": inserted, "skipped": skipped}

    # ==================== 数据查询 ====================

    def query_by_vector(self, vector: List[float], k: int = 10,
                        return_properties: Optional[List[str]] = None,
                        include_metadata: bool = True) -> List[Dict]:
        """
        向量相似度查询 (Top-K)

        Args:
            vector: 查询向量
            k: 返回数量
            return_properties: 返回的属性列表
            include_metadata: 是否包含元数据

        Returns:
            查询结果列表
        """
        try:
            kwargs = {
                "near_vector": vector,
                "limit": k
            }

            if return_properties:
                kwargs["return_properties"] = return_properties

            if include_metadata:
                kwargs["return_metadata"] = ["distance", "certainty"]

            response = self.collection.query.near_vector(**kwargs)
            return [self._parse_object(obj) for obj in response.objects]

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return []

    def query_by_certainty(self, vector: List[float], min_certainty: float = 0.8,
                           return_properties: Optional[List[str]] = None) -> List[Dict]:
        """
        按置信度阈值查询

        Args:
            vector: 查询向量
            min_certainty: 最小置信度阈值
            return_properties: 返回的属性列表

        Returns:
            查询结果列表
        """
        try:
            kwargs = {
                "near_vector": vector,
                "certainty": min_certainty
            }

            if return_properties:
                kwargs["return_properties"] = return_properties

            kwargs["return_metadata"] = ["distance", "certainty"]

            response = self.collection.query.near_vector(**kwargs)
            return [self._parse_object(obj) for obj in response.objects]

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return []

    def query_by_filter(self, field: str, value: Any,
                        return_properties: Optional[List[str]] = None,
                        limit: int = 100) -> List[Dict]:
        """
        按属性过滤查询

        Args:
            field: 属性名
            value: 属性值
            return_properties: 返回的属性列表
            limit: 返回数量限制

        Returns:
            查询结果列表
        """
        try:
            kwargs = {
                "filters": Filter.by_property(field).equal(value),
                "limit": limit
            }

            if return_properties:
                kwargs["return_properties"] = return_properties

            response = self.collection.query.fetch_objects(**kwargs)
            return [self._parse_object(obj) for obj in response.objects]

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return []

    # ==================== 数据删除 ====================

    def delete_by_id(self, id_value: str, id_field: str = "id_") -> bool:
        """
        根据属性ID删除文档

        Args:
            id_value: ID值
            id_field: ID字段名

        Returns:
            是否删除成功
        """
        try:
            response = self.collection.query.fetch_objects(
                filters=Filter.by_property(id_field).equal(id_value),
                limit=1
            )

            if response.objects:
                self.collection.data.delete_by_id(response.objects[0].uuid)
                print(f"✅ 删除成功: {id_field}={id_value}")
                return True
            else:
                print(f"⚠️ 未找到: {id_field}={id_value}")
                return False

        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False

    def delete_by_ids(self, id_values: List[str], id_field: str = "id_") -> Dict[str, int]:
        """
        批量删除文档

        Args:
            id_values: ID值列表
            id_field: ID字段名

        Returns:
            {"deleted": 删除数量, "not_found": 未找到数量}
        """
        deleted, not_found = 0, 0

        for id_value in tqdm(id_values, desc="删除文档"):
            if self.delete_by_id(id_value, id_field):
                deleted += 1
            else:
                not_found += 1

        print(f"✅ 删除完成: {deleted} 条, 未找到: {not_found} 条")
        return {"deleted": deleted, "not_found": not_found}

    # ==================== 辅助方法 ====================

    def _exists_by_field(self, field: str, value: Any) -> bool:
        """检查字段值是否已存在"""
        try:
            response = self.collection.query.fetch_objects(
                filters=Filter.by_property(field).equal(value),
                limit=1
            )
            return len(response.objects) > 0
        except Exception:
            return False

    def _parse_object(self, obj) -> Dict:
        """解析查询结果对象"""
        result = dict(obj.properties)
        if hasattr(obj, "metadata") and obj.metadata:
            result["certainty"] = getattr(obj.metadata, "certainty", None)
            result["distance"] = getattr(obj.metadata, "distance", None)
        if hasattr(obj, "uuid"):
            result["uuid"] = str(obj.uuid)
        return result

    def count(self) -> int:
        """获取集合中的文档数量"""
        try:
            response = self.collection.query.fetch_objects(limit=1)
            # Weaviate 不直接提供 count，这里返回 -1 表示需要聚合查询
            return -1
        except Exception:
            return -1