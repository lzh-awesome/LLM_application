"""
查询关系脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient
from embedding import TextEmbedding
from collections import LinkCollection


def query_links(query: str, collection_name: str = "natural_triples", k: int = 10):
    """
    查询关系

    Args:
        query: 查询文本
        collection_name: 集合名称
        k: 返回数量
    """
    settings = get_settings()
    weaviate_client = WeaviateClient(settings.weaviate)
    text_embedding = TextEmbedding(settings.models.bge_m3_path)

    link_collection = LinkCollection(weaviate_client.client, collection_name)

    # 向量化查询
    vector = text_embedding.encode_single(query)

    # 查询
    results = link_collection.query(vector, k)

    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 条关系:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('triple_name')}")
        print(f"   ID: {result.get('id_')}")
        print(f"   头实体: {result.get('_from')}")
        print(f"   尾实体: {result.get('_to')}")
        print(f"   置信度: {result.get('certainty', 0):.4f}")
        print()

    weaviate_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查询关系")
    parser.add_argument("query", type=str, help="查询文本")
    parser.add_argument("--collection", type=str, default="natural_triples", help="集合名称")
    parser.add_argument("-k", type=int, default=10, help="返回数量")

    args = parser.parse_args()
    query_links(args.query, args.collection, args.k)