"""
连接测试脚本
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings, Settings
from core import WeaviateClient, ArangoClientManager


def test_weaviate_connection():
    """测试 Weaviate 连接"""
    print("=" * 50)
    print("测试 Weaviate 连接")
    print("=" * 50)

    try:
        settings = get_settings()
        print(f"配置信息:")
        print(f"  HTTP: {settings.weaviate.http_host}:{settings.weaviate.http_port}")
        print(f"  gRPC: {settings.weaviate.grpc_host}:{settings.weaviate.grpc_port}")

        client = WeaviateClient(settings.weaviate)
        collections = client.list_collections()

        print(f"\n✅ 连接成功！")
        print(f"现有集合 ({len(collections)} 个):")
        for name in collections:
            print(f"  - {name}")

        client.close()
        return True

    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        return False


def test_arango_connection():
    """测试 ArangoDB 连接"""
    print("\n" + "=" * 50)
    print("测试 ArangoDB 连接")
    print("=" * 50)

    try:
        settings = get_settings()
        print(f"配置信息:")
        print(f"  Host: {settings.arango.host}")
        print(f"  Database: {settings.arango.database}")

        db = ArangoClientManager.connect(settings.arango)
        collections = db.collections()

        print(f"\n✅ 连接成功！")
        print(f"现有集合 ({len(collections)} 个):")
        for name in collections:
            print(f"  - {name}")

        return True

    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 50)
    print("测试配置加载")
    print("=" * 50)

    try:
        settings = get_settings()

        print(f"\n✅ 配置加载成功！")
        print(f"\nWeaviate 配置:")
        print(f"  HTTP Host: {settings.weaviate.http_host}")
        print(f"  HTTP Port: {settings.weaviate.http_port}")
        print(f"  API Key: {settings.weaviate.api_key[:10]}...")

        print(f"\nArangoDB 配置:")
        print(f"  Host: {settings.arango.host}")
        print(f"  Database: {settings.arango.database}")

        print(f"\n模型配置:")
        print(f"  BGE-M3: {settings.models.bge_m3_path}")
        print(f"  BGE-VL: {settings.models.bge_vl_path}")

        print(f"\n批处理大小: {settings.batch_size}")

        return True

    except Exception as e:
        print(f"\n❌ 配置加载失败: {e}")
        return False


def main():
    print("\n" + "=" * 50)
    print("Weaviate 向量库连接测试")
    print("=" * 50 + "\n")

    results = []

    # 测试配置加载
    results.append(("配置加载", test_config_loading()))

    # 测试 Weaviate 连接
    results.append(("Weaviate 连接", test_weaviate_connection()))

    # 测试 ArangoDB 连接
    results.append(("ArangoDB 连接", test_arango_connection()))

    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\n总计: {passed}/{total} 通过")


if __name__ == "__main__":
    main()