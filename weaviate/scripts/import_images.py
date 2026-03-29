"""
批量导入图片脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient
from embedding import TextEmbedding, ImageEmbedding
from collections import ImageCollection, ImageNameCollection


def import_images(
    root_directory: str,
    create_image_collection: bool = False,
    create_name_collection: bool = False,
    include_base64: bool = True
):
    """
    批量导入图片

    Args:
        root_directory: 图片根目录
        create_image_collection: 是否创建图搜图集合
        create_name_collection: 是否创建文搜图集合
        include_base64: 是否包含Base64编码
    """
    settings = get_settings()

    weaviate_client = WeaviateClient(settings.weaviate)
    text_embedding = TextEmbedding(settings.models.bge_m3_path)
    image_embedding = ImageEmbedding(settings.models.bge_vl_path)

    # 获取图片信息
    print(f"正在扫描目录: {root_directory}")
    images_info = image_embedding.get_images_info(root_directory, include_base64)
    print(f"共找到 {len(images_info)} 张图片")

    if not images_info:
        print("没有找到图片")
        weaviate_client.close()
        return

    # 向量化图片名称（用于文搜图）
    print("\n正在向量化图片名称...")
    image_names = [img["image_name"] for img in images_info]
    name_vectors = text_embedding.encode(image_names, settings.batch_size)

    # 向量化图片（用于图搜图）
    print("\n正在向量化图片...")
    image_paths = [img["image_path"] for img in images_info]
    image_vectors = image_embedding.encode(image_paths)

    # 存入图搜图集合
    image_coll = ImageCollection(weaviate_client.client)
    if create_image_collection:
        image_coll.create_collection()

    print("\n正在存入图搜图集合...")
    result = image_coll.add(images_info, image_vectors)
    print(f"图搜图集合: {result}")

    # 存入文搜图集合
    name_coll = ImageNameCollection(weaviate_client.client)
    if create_name_collection:
        name_coll.create_collection()

    print("\n正在存入文搜图集合...")
    result = name_coll.add(images_info, name_vectors)
    print(f"文搜图集合: {result}")

    weaviate_client.close()
    print("\n✅ 图片导入完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量导入图片")
    parser.add_argument("directory", type=str, help="图片根目录")
    parser.add_argument("--create-image", action="store_true", help="创建图搜图集合")
    parser.add_argument("--create-name", action="store_true", help="创建文搜图集合")
    parser.add_argument("--no-base64", action="store_true", help="不存储Base64编码")

    args = parser.parse_args()
    import_images(
        args.directory,
        args.create_image,
        args.create_name,
        not args.no_base64
    )