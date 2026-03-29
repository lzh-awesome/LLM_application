"""
Weaviate 向量库管理工具
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from config import get_settings
from core import WeaviateClient


def list_collections():
    """列出所有集合"""
    settings = get_settings()
    client = WeaviateClient(settings.weaviate)

    collections = client.list_collections()
    print("\n现有集合:")
    for name in collections:
        print(f"  - {name}")

    client.close()


def test_connection():
    """测试连接"""
    settings = get_settings()

    print("正在测试 Weaviate 连接...")
    try:
        client = WeaviateClient(settings.weaviate)
        collections = client.list_collections()
        print(f"✅ 连接成功！共 {len(collections)} 个集合")
        client.close()
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def delete_collection(collection_name: str):
    """删除集合"""
    settings = get_settings()
    client = WeaviateClient(settings.weaviate)

    if not client.collection_exists(collection_name):
        print(f"❌ 集合 '{collection_name}' 不存在")
        client.close()
        return

    confirm = input(f"确定要删除集合 '{collection_name}' 吗？(y/n): ")
    if confirm.lower() == 'y':
        from core.operations import VectorStoreOperations
        ops = VectorStoreOperations(client.client, collection_name)
        ops.delete()
    else:
        print("已取消")

    client.close()


def main():
    parser = argparse.ArgumentParser(description="Weaviate 向量库管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # list 命令
    subparsers.add_parser("list", help="列出所有集合")

    # test 命令
    subparsers.add_parser("test", help="测试连接")

    # delete 命令
    delete_parser = subparsers.add_parser("delete", help="删除集合")
    delete_parser.add_argument("collection", type=str, help="集合名称")

    args = parser.parse_args()

    if args.command == "list":
        list_collections()
    elif args.command == "test":
        test_connection()
    elif args.command == "delete":
        delete_collection(args.collection)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()