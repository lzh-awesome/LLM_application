"""
批量导入视频脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import get_settings
from core import WeaviateClient
from embedding import TextEmbedding, VideoEmbedding
from collections import VideoCollection, VideoNameCollection


def import_videos(
    root_directory: str,
    create_video_collection: bool = False,
    create_name_collection: bool = False,
    include_base64: bool = True
):
    """
    批量导入视频

    Args:
        root_directory: 视频根目录
        create_video_collection: 是否创建视频搜视频集合
        create_name_collection: 是否创建文搜视频集合
        include_base64: 是否包含Base64编码
    """
    settings = get_settings()

    weaviate_client = WeaviateClient(settings.weaviate)
    text_embedding = TextEmbedding(settings.models.bge_m3_path)
    video_embedding = VideoEmbedding(
        settings.models.video_clip_path,
        settings.models.video_clip_weights
    )

    # 获取视频信息
    print(f"正在扫描目录: {root_directory}")
    videos_info = video_embedding.get_videos_info(root_directory, include_base64)
    print(f"共找到 {len(videos_info)} 个视频")

    if not videos_info:
        print("没有找到视频")
        weaviate_client.close()
        return

    # 向量化视频名称（用于文搜视频）
    print("\n正在向量化视频名称...")
    video_names = [v["video_name"] for v in videos_info]
    name_vectors = text_embedding.encode(video_names, settings.batch_size)

    # 向量化视频（用于视频搜视频）
    print("\n正在向量化视频...")
    video_paths = [v["video_path"] for v in videos_info]
    video_vectors = video_embedding.encode(video_paths)

    # 存入视频搜视频集合
    video_coll = VideoCollection(weaviate_client.client)
    if create_video_collection:
        video_coll.create_collection()

    print("\n正在存入视频搜视频集合...")
    result = video_coll.add(videos_info, video_vectors)
    print(f"视频搜视频集合: {result}")

    # 存入文搜视频集合
    name_coll = VideoNameCollection(weaviate_client.client)
    if create_name_collection:
        name_coll.create_collection()

    print("\n正在存入文搜视频集合...")
    result = name_coll.add(videos_info, name_vectors)
    print(f"文搜视频集合: {result}")

    weaviate_client.close()
    print("\n✅ 视频导入完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量导入视频")
    parser.add_argument("directory", type=str, help="视频根目录")
    parser.add_argument("--create-video", action="store_true", help="创建视频搜视频集合")
    parser.add_argument("--create-name", action="store_true", help="创建文搜视频集合")
    parser.add_argument("--no-base64", action="store_true", help="不存储Base64编码")

    args = parser.parse_args()
    import_videos(
        args.directory,
        args.create_video,
        args.create_name,
        not args.no_base64
    )