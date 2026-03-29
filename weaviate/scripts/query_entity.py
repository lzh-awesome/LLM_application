"""
查询实体脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient
from embedding import TextEmbedding
from collections import EntityCollection


def query_entities(query: str, collection_name: str = "kg_wenda", k: int = 10):
    """
    查询实体

    Args:
        query: 查询文本
        collection_name: 集合名称
        k: 返回数量
    """
    settings = get_settings()
    weaviate_client = WeaviateClient(settings.weaviate)
    text_embedding = TextEmbedding(settings.models.bge_m3_path)

    entity_collection = EntityCollection(weaviate_client.client, collection_name)

    # 向量化查询
    vector = text_embedding.encode_single(query)

    # 查询
    results = entity_collection.query(vector, k)

    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 个结果:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('name')}")
        print(f"   ID: {result.get('id_')}")
        print(f"   描述: {result.get('desc', '')[:100]}...")
        print(f"   置信度: {result.get('certainty', 0):.4f}")
        print()

    weaviate_client.close()


def query_entities_by_certainty(query: str, collection_name: str = "kg_wenda", min_certainty: float = 0.8):
    """
    按置信度查询实体

    Args:
        query: 查询文本
        collection_name: 集合名称
        min_certainty: 最小置信度
    """
    settings = get_settings()
    weaviate_client = WeaviateClient(settings.weaviate)
    text_embedding = TextEmbedding(settings.models.bge_m3_path)

    entity_collection = EntityCollection(weaviate_client.client, collection_name)

    vector = text_embedding.encode_single(query)
    results = entity_collection.query_by_certainty(vector, min_certainty)

    print(f"\n查询: {query}")
    print(f"置信度阈值: {min_certainty}")
    print(f"找到 {len(results)} 个结果:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('name')} (置信度: {result.get('certainty', 0):.4f})")

    weaviate_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查询实体")
    parser.add_argument("query", type=str, help="查询文本")
    parser.add_argument("--collection", type=str, default="kg_wenda", help="集合名称")
    parser.add_argument("-k", type=int, default=10, help="返回数量")
    parser.add_argument("--certainty", type=float, help="最小置信度阈值")

    args = parser.parse_args()

    if args.certainty:
        query_entities_by_certainty(args.query, args.collection, args.certainty)
    else:
        query_entities(args.query, args.collection, args.k)