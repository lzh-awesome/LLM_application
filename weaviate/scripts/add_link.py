"""
添加关系（三元组）脚本
从 ArangoDB 读取关系数据并存入 Weaviate 向量库
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient, ArangoClientManager
from embedding import TextEmbedding
from collections import LinkCollection


def get_formatted_triples(arango_db, link_collection: str, entity_collection: str) -> list:
    """
    从 ArangoDB 获取格式化的三元组

    Args:
        arango_db: ArangoDB 数据库实例
        link_collection: 关系集合名称
        entity_collection: 实体集合名称

    Returns:
        三元组列表
    """
    # 检查集合是否存在
    if not arango_db.has_collection(link_collection):
        raise ValueError(f"集合 '{link_collection}' 不存在")

    # 查询所有关系
    aql_query = f"""
    FOR doc IN {link_collection}
        FILTER HAS(doc, 'label')
        RETURN doc
    """
    cursor = arango_db.aql.execute(aql_query)
    link_docs = list(cursor)

    # 构造三元组
    triples = []
    for link_doc in link_docs:
        from_field = link_doc.get("_from", "")
        to_field = link_doc.get("_to", "")
        from_entity_id = from_field.split("/")[-1] if from_field else None
        to_entity_id = to_field.split("/")[-1] if to_field else None

        # 查询头实体名称
        from_name = get_entity_name(arango_db, entity_collection, from_entity_id)
        # 查询尾实体名称
        to_name = get_entity_name(arango_db, entity_collection, to_entity_id)

        if from_name and to_name:
            relation = link_doc.get("label", "")
            triple_name = f"{from_name}的{relation}是{to_name}"
            triples.append({
                "_id": link_doc.get("_id"),
                "triple_name": triple_name,
                "_from": from_entity_id,
                "_to": to_entity_id
            })

    return triples


def get_entity_name(arango_db, entity_collection: str, entity_id: str) -> str:
    """获取实体名称"""
    if not entity_id:
        return None

    query = f"""
    FOR doc IN {entity_collection}
        FILTER doc._key == "{entity_id}"
        RETURN doc
    """
    cursor = arango_db.aql.execute(query)
    docs = list(cursor)

    if docs:
        labels = docs[0].get("labels", {})
        return labels.get("zh-cn") or labels.get("zh") or labels.get("en")
    return None


def add_links_from_arango(
    link_collection: str = "link",
    entity_collection: str = "entity",
    weaviate_collection: str = "natural_triples",
    create_collection: bool = False
):
    """
    从 ArangoDB 读取关系并添加到 Weaviate

    Args:
        link_collection: ArangoDB 关系集合名称
        entity_collection: ArangoDB 实体集合名称
        weaviate_collection: Weaviate 集合名称
        create_collection: 是否创建新集合
    """
    settings = get_settings()

    weaviate_client = WeaviateClient(settings.weaviate)
    arango_db = ArangoClientManager.connect(settings.arango)
    text_embedding = TextEmbedding(settings.models.bge_m3_path)

    link_coll = LinkCollection(weaviate_client.client, weaviate_collection)

    if create_collection:
        link_coll.create_collection()

    # 获取三元组
    print(f"正在从 ArangoDB 读取关系...")
    triples = get_formatted_triples(arango_db, link_collection, entity_collection)
    print(f"共读取 {len(triples)} 条关系")

    # 向量化
    triple_names = [t["triple_name"] for t in triples]
    vectors = text_embedding.encode(triple_names, settings.batch_size)

    # 存入 Weaviate
    result = link_coll.add(triples, vectors)
    print(f"添加完成: {result}")

    weaviate_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="添加关系到 Weaviate")
    parser.add_argument("--link-collection", type=str, default="link", help="ArangoDB 关系集合名称")
    parser.add_argument("--entity-collection", type=str, default="entity", help="ArangoDB 实体集合名称")
    parser.add_argument("--weaviate-collection", type=str, default="natural_triples", help="Weaviate 集合名称")
    parser.add_argument("--create", action="store_true", help="是否创建新集合")

    args = parser.parse_args()
    add_links_from_arango(
        args.link_collection,
        args.entity_collection,
        args.weaviate_collection,
        args.create
    )