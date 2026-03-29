"""
添加实体脚本
从 ArangoDB 读取实体数据并存入 Weaviate 向量库
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient, ArangoClientManager
from embedding import TextEmbedding
from collections import EntityCollection


def add_entities_from_arango(
    arango_collection: str,
    weaviate_collection: str = "kg_wenda",
    create_collection: bool = False
):
    """
    从 ArangoDB 读取实体并添加到 Weaviate

    Args:
        arango_collection: ArangoDB 集合名称
        weaviate_collection: Weaviate 集合名称
        create_collection: 是否创建新集合
    """
    settings = get_settings()

    # 连接数据库
    weaviate_client = WeaviateClient(settings.weaviate)
    arango_db = ArangoClientManager.connect(settings.arango)

    # 初始化向量化模型
    text_embedding = TextEmbedding(settings.models.bge_m3_path)

    # 初始化集合
    entity_collection = EntityCollection(weaviate_client.client, weaviate_collection)

    if create_collection:
        entity_collection.create_collection()

    # 从 ArangoDB 读取实体
    print(f"正在从 ArangoDB 集合 '{arango_collection}' 读取实体...")
    collection = arango_db.collection(arango_collection)
    cursor = collection.all()

    entities = []
    for doc in cursor:
        if doc.get("name"):
            entities.append({
                "_id": doc.get("_id"),
                "name": doc.get("name"),
                "desc": doc.get("desc", "")
            })

    print(f"共读取 {len(entities)} 个实体")

    # 向量化
    names = [e["name"] for e in entities]
    vectors = text_embedding.encode(names, settings.batch_size)

    # 存入 Weaviate
    result = entity_collection.add(entities, vectors)
    print(f"添加完成: {result}")

    # 关闭连接
    weaviate_client.close()


def add_entities_from_list(entities: list, weaviate_collection: str = "kg_wenda"):
    """
    从实体列表添加到 Weaviate

    Args:
        entities: 实体列表
        weaviate_collection: Weaviate 集合名称
    """
    settings = get_settings()

    weaviate_client = WeaviateClient(settings.weaviate)
    text_embedding = TextEmbedding(settings.models.bge_m3_path)
    entity_collection = EntityCollection(weaviate_client.client, weaviate_collection)

    names = [e["name"] for e in entities]
    vectors = text_embedding.encode(names, settings.batch_size)

    result = entity_collection.add(entities, vectors)
    print(f"添加完成: {result}")

    weaviate_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="添加实体到 Weaviate")
    parser.add_argument("--arango-collection", type=str, help="ArangoDB 集合名称")
    parser.add_argument("--weaviate-collection", type=str, default="kg_wenda", help="Weaviate 集合名称")
    parser.add_argument("--create", action="store_true", help="是否创建新集合")

    args = parser.parse_args()

    if args.arango_collection:
        add_entities_from_arango(
            args.arango_collection,
            args.weaviate_collection,
            args.create
        )
    else:
        print("请指定 --arango-collection 参数")