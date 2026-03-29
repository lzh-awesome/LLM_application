"""
删除实体脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient
from collections import EntityCollection


def delete_entity(entity_id: str, collection_name: str = "kg_wenda"):
    """
    删除实体

    Args:
        entity_id: 实体ID
        collection_name: 集合名称
    """
    settings = get_settings()
    weaviate_client = WeaviateClient(settings.weaviate)

    entity_collection = EntityCollection(weaviate_client.client, collection_name)
    success = entity_collection.delete(entity_id)

    if success:
        print(f"✅ 实体 {entity_id} 删除成功")
    else:
        print(f"❌ 实体 {entity_id} 删除失败或不存在")

    weaviate_client.close()


def delete_entities(entity_ids: list, collection_name: str = "kg_wenda"):
    """
    批量删除实体

    Args:
        entity_ids: 实体ID列表
        collection_name: 集合名称
    """
    settings = get_settings()
    weaviate_client = WeaviateClient(settings.weaviate)

    entity_collection = EntityCollection(weaviate_client.client, collection_name)

    deleted = 0
    for entity_id in entity_ids:
        if entity_collection.delete(entity_id):
            deleted += 1

    print(f"删除完成: {deleted}/{len(entity_ids)}")

    weaviate_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除实体")
    parser.add_argument("--id", type=str, help="实体ID")
    parser.add_argument("--ids", type=str, help="实体ID列表（逗号分隔）")
    parser.add_argument("--collection", type=str, default="kg_wenda", help="集合名称")

    args = parser.parse_args()

    if args.id:
        delete_entity(args.id, args.collection)
    elif args.ids:
        ids = [id.strip() for id in args.ids.split(",")]
        delete_entities(ids, args.collection)
    else:
        print("请指定 --id 或 --ids 参数")