"""
删除关系脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient
from collections import LinkCollection


def delete_link(link_id: str, collection_name: str = "natural_triples"):
    """
    删除关系

    Args:
        link_id: 关系ID
        collection_name: 集合名称
    """
    settings = get_settings()
    weaviate_client = WeaviateClient(settings.weaviate)

    link_collection = LinkCollection(weaviate_client.client, collection_name)
    success = link_collection.delete(link_id)

    if success:
        print(f"✅ 关系 {link_id} 删除成功")
    else:
        print(f"❌ 关系 {link_id} 删除失败或不存在")

    weaviate_client.close()


def delete_links(link_ids: list, collection_name: str = "natural_triples"):
    """
    批量删除关系

    Args:
        link_ids: 关系ID列表
        collection_name: 集合名称
    """
    settings = get_settings()
    weaviate_client = WeaviateClient(settings.weaviate)

    link_collection = LinkCollection(weaviate_client.client, collection_name)

    deleted = 0
    for link_id in link_ids:
        if link_collection.delete(link_id):
            deleted += 1

    print(f"删除完成: {deleted}/{len(link_ids)}")

    weaviate_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除关系")
    parser.add_argument("--id", type=str, help="关系ID")
    parser.add_argument("--ids", type=str, help="关系ID列表（逗号分隔）")
    parser.add_argument("--collection", type=str, default="natural_triples", help="集合名称")

    args = parser.parse_args()

    if args.id:
        delete_link(args.id, args.collection)
    elif args.ids:
        ids = [id.strip() for id in args.ids.split(",")]
        delete_links(ids, args.collection)
    else:
        print("请指定 --id 或 --ids 参数")