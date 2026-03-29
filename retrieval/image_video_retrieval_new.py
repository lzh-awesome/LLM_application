# 智能检索的图片检索接口、视频检索接口
# 端口号：7001
# 功能：
# 输入：需要检索的图片/视频
# 输出：
# 1.图片/视频的描述内容
# 2.检索到的相关图片/视频
# 3.相关实体的id

from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from arango import ArangoClient
import weaviate
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer

import torch
import requests
import time
import json
import cv2
import numpy as np
import tempfile
import uvicorn
from tqdm import tqdm
from transformers import AutoModel
from typing import List

from modeling import VideoCLIP_XL


# 配置类
class Config:
    # Weaviate连接参数
    WEAVIATE_HTTP_HOST = "weaviate"
    WEAVIATE_HTTP_PORT = 8080
    WEAVIATE_GRPC_HOST = "weaviate"
    WEAVIATE_GRPC_PORT = 50051
    WEAVIATE_API_KEY = "test-secret-key"

    # ArangoDB连接参数
    DB_HOST = "http://10.117.254.37:8529"
    DB_USERNAME = "root"
    DB_PASSWORD = "root"
    DB_NAME = "wiki"

    # 模型配置
    IMAGE_MODEL_PATH = "/workspace/codes/deepseek/multimodal/int_retrieval/image_retrieval/BGE-VL-base"
    TEXT_MODEL_PATH = "/workspace/codes/deepseek/deepseek_model/bge-m3"
    VIDEO_MODEL_PATH = "/workspace/codes/deepseek/multimodal/int_retrieval/video_retrieval/VideoCLIP-XL/VideoCLIP-XL.bin"

    # API配置
    IMAGE_DESCRIPTION_URL = "http://10.117.1.238:7012/wenda/image"
    VIDEO_DESCRIPTION_URL = "http://10.117.1.238:7012/wenda/video"

    # 向量集合名称
    IMAGE_COLLECTION_NAME = "image_retrieval_new"
    IMAGE_NAME_COLLECTION_NAME = "image_name_retrieval"
    VIDEO_COLLECTION_NAME = "video_retrieval"
    VIDEO_NAME_COLLECTION_NAME = "video_name_retrieval"

    # 候选图片目录
    CANDIDATE_IMAGE_DIR = "/workspace/codes/deepseek/multimodal/int_retrieval/fdfs2"


# 初始化 FastAPI 应用
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载图片检索模型
image_model = AutoModel.from_pretrained(
    Config.IMAGE_MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
image_model.eval()
image_model.set_processor(Config.IMAGE_MODEL_PATH)

# 加载文本模型
text_model = SentenceTransformer(Config.TEXT_MODEL_PATH)

# 加载视频检索模型
videoclip_xl = VideoCLIP_XL()
state_dict = torch.load(Config.VIDEO_MODEL_PATH, map_location="cpu")
videoclip_xl.load_state_dict(state_dict)
videoclip_xl.cuda().eval()

# 连接 Weaviate
client = weaviate.connect_to_custom(
    http_host=Config.WEAVIATE_HTTP_HOST,
    http_port=Config.WEAVIATE_HTTP_PORT,
    http_secure=False,
    grpc_host=Config.WEAVIATE_GRPC_HOST,
    grpc_port=Config.WEAVIATE_GRPC_PORT,
    grpc_secure=False,
    auth_credentials=AuthApiKey(Config.WEAVIATE_API_KEY)
)


def vectorize_image(image_path):
    """使用 BGE-VL-base 模型对图片进行向量化"""
    with torch.no_grad():
        embeddings = image_model.encode(images=[image_path])
        return embeddings.cpu().numpy()[0]


def vectorize_text(texts: List[str], batch_size: int = 1000):
    """对文本列表进行向量化"""
    embeddings = []
    total_batches = (len(texts) - 1) // batch_size + 1
    for i in tqdm(range(0, len(texts), batch_size), desc="Vectorizing texts", total=total_batches):
        batch = texts[i:i + batch_size]
        batch_embeddings = text_model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings


def query_vector_collection(collection_name: str, vector, k: int, return_properties: list) -> list:
    """通用向量查询函数"""
    collection = client.collections.get(collection_name)
    try:
        response = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            return_properties=return_properties,
            return_metadata=["distance", "certainty"],
        )
        results = []
        for obj in response.objects:
            result = {"certainty": obj.metadata.certainty}
            for prop in return_properties:
                result[prop] = obj.properties.get(prop)
            results.append(result)
        return results
    except Exception as e:
        print(f"查询异常: {e}")
        return []


def query_entity_by_name(entity_name):
    """根据实体名称查询实体信息"""
    arango_client = ArangoClient(hosts=Config.DB_HOST)
    db = arango_client.db(Config.DB_NAME, username=Config.DB_USERNAME, password=Config.DB_PASSWORD)

    if not db.has_collection("entity"):
        raise ValueError("Collection 'entity' does not exist in the database.")

    aql_query = f"""
    FOR doc IN entity
        FILTER doc.labels['zh-cn'] == "{entity_name}"
        RETURN doc
    """

    try:
        cursor = db.aql.execute(aql_query)
        results = list(cursor)
        if results:
            return results[0]
        else:
            print(f"未找到与实体名称 '{entity_name}' 匹配的记录。")
            return []
    except Exception as e:
        print(f"查询失败：{str(e)}")
        return None


def extract_entity_id(data):
    """从实体数据中提取ID"""
    if "_id" not in data:
        raise KeyError("字典中缺少 '_id' 键")
    _id_value = data["_id"]
    if not _id_value.startswith("entity/"):
        raise ValueError("_id 的值格式不正确，应以 'entity/' 开头")
    return _id_value.split("entity/")[1]


def build_retrieval_result(items: list, media_type: str, description: str = None, entity_id_list: list = None):
    """构造检索结果"""
    result = {"message": f"以下是我检索到的{media_type}：", "results": items}
    if description:
        result[f"{media_type}_description"] = description
    if entity_id_list:
        result["entity_id_list"] = entity_id_list
    return result


# ------------------------------- 图片检索接口 -------------------------------
@app.post("/int_retrieval_image")
async def image_retrieval(image_file: UploadFile = File(...)):
    try:
        print("````````````````````开始图片智能检索`````````````````````")

        temp_image_path = tempfile.mktemp(suffix=".jpg", dir=Config.CANDIDATE_IMAGE_DIR)
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await image_file.read())

        # 图片向量化并检索
        image_vector = vectorize_image(temp_image_path)
        print("图片向量化完成。")

        similar_images = query_vector_collection(
            Config.IMAGE_COLLECTION_NAME,
            image_vector,
            k=3,
            return_properties=["image_name", "image_id", "image_base64"]
        )

        print(f"找到 {len(similar_images)} 张相似图片")
        image_names = similar_images[0].get("image_name")
        print("\nimage_names:\n", image_names)

        # 查询实体ID
        entity_id_list = []
        entities_raw_info = query_entity_by_name(image_names)
        print("\nentities_raw_info:\n", entities_raw_info)

        # 获取图片描述
        final_prompt = f"已知该图片的信息与 {image_names} 相关，请根据这些信息描述该图片的内容, 在你生成的描述内容中需要包含{image_names}。"
        image_prompt = {"user_text": final_prompt}
        print("\nimage_prompt:\n", image_prompt)

        with open(temp_image_path, "rb") as f:
            files = {"media": (image_file.filename, f, image_file.content_type)}
            description_response = requests.post(Config.IMAGE_DESCRIPTION_URL, files=files, data=image_prompt)

        if description_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve image description.")

        description_data = description_response.json()
        image_description = description_data.get("description", "无法描述图片")
        print("image_description:\n", image_description)

        # 构造结果列表
        results = [
            {
                "related_image": img.get("image_name"),
                "related_image_id": img.get("image_id"),
                "image_base64": img.get("image_base64"),
                "similarity_score": img.get("certainty")
            }
            for img in similar_images
        ]

        if entities_raw_info:
            entity_id_list.append(extract_entity_id(entities_raw_info))

        return build_retrieval_result(results, "图片", image_description, entity_id_list if entity_id_list else None)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image retrieval: {str(e)}")


@app.post("/text_retrieval_image")
async def text_retrieval_image(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        print("question:\n", question)

        question_embedding = vectorize_text([question], batch_size=1)[0]
        print("问题向量化完成。")

        similar_images = query_vector_collection(
            Config.IMAGE_NAME_COLLECTION_NAME,
            question_embedding,
            k=3,
            return_properties=["image_name", "image_base64"]
        )

        image_names = similar_images[0].get("image_name")
        print("image_names:\n", image_names)

        entity_id_list = []
        entities_raw_info = query_entity_by_name(image_names)
        print("entities_raw_info:\n", entities_raw_info)

        results = [
            {
                "related_image": img.get("image_name"),
                "image_base64": img.get("image_base64"),
                "similarity_score": img.get("certainty")
            }
            for img in similar_images
        ]

        if entities_raw_info:
            entity_id_list.append(extract_entity_id(entities_raw_info))

        return build_retrieval_result(results, "图片", None, entity_id_list if entity_id_list else None)

    except Exception as e:
        print(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


# ------------------------------- 视频检索接口 -------------------------------
def _frame_from_video(video):
    """从视频文件中逐帧读取图像"""
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def video_preprocessing(video_path, fnum=8):
    """视频预处理"""
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    step = len(frames) // fnum
    frames = frames[::step][:fnum]

    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    vid_tube = []
    for fr in frames:
        fr = fr[:, :, ::-1]
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims((fr / 255.0 - v_mean) / v_std, axis=(0, 1))
        vid_tube.append(fr)

    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    return torch.from_numpy(vid_tube)


def vectorize_videos(video_paths, model, batch_size: int = 8):
    """预处理视频并提取特征向量"""
    video_embeddings = []
    for i in tqdm(range(0, len(video_paths), batch_size), desc="Processing video batches"):
        batch_video_paths = video_paths[i:i + batch_size]
        batch_video_tensors = []

        for video_path in batch_video_paths:
            try:
                video_tensor = video_preprocessing(video_path).float().cuda()
                batch_video_tensors.append(video_tensor)
            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")
                continue

        if not batch_video_tensors:
            continue

        batch_video_tensor = torch.cat(batch_video_tensors, dim=0)
        with torch.no_grad():
            batch_features = model.vision_model.get_vid_features(batch_video_tensor).float()
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

        video_embeddings.extend(batch_features.cpu().numpy())

    return video_embeddings


@app.post("/int_retrieval_video")
async def video_retrieval(media: UploadFile = File(...)):
    try:
        print("````````````````````开始视频智能检索`````````````````````")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_video_path = os.path.join(temp_dir, media.filename)
            with open(temp_video_path, "wb") as temp_file:
                temp_file.write(await media.read())

            print(f"视频已保存到临时目录: {temp_video_path}")

            video_vectors = vectorize_videos([temp_video_path], videoclip_xl, batch_size=8)
            print("视频向量化完成")

            similar_videos = query_vector_collection(
                Config.VIDEO_COLLECTION_NAME,
                video_vectors[0],
                k=3,
                return_properties=["video_name", "video_base64"]
            )
            print("相似视频检索完成")

            sorted_similar_videos = sorted(similar_videos, key=lambda x: x['certainty'], reverse=True)
            video_name = sorted_similar_videos[0].get("video_name")
            print("video_name:\n", video_name)

            entity_id_list = []
            entities_raw_info = query_entity_by_name(video_name)
            print("entities_raw_info:\n", entities_raw_info)

            final_prompt = f"已知该视频的信息与 {video_name} 相关，请根据这些信息描述该视频的内容, 在你生成的描述内容中需要包含{video_name}。"
            video_prompt = {"user_text": final_prompt}
            print("\nvideo_prompt:\n", video_prompt)

            description_response = requests.post(
                Config.VIDEO_DESCRIPTION_URL,
                files={"media": open(temp_video_path, "rb")},
                data=video_prompt
            )

            if description_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to retrieve video description.")

            description_data = description_response.json()
            video_description = description_data.get("description")
            print("\nvideo_description:\n", video_description)

            results = [
                {
                    "related_video": video.get("video_name"),
                    "video_base64": video.get("video_base64"),
                    "similarity_score": video.get("certainty")
                }
                for video in similar_videos
            ]

            if entities_raw_info:
                entity_id_list.append(extract_entity_id(entities_raw_info))

            return build_retrieval_result(results, "视频", video_description, entity_id_list if entity_id_list else None)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频检索过程中发生错误: {str(e)}")


@app.post("/text_retrieval_video")
async def text_retrieval_video(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        print("question:\n", question)

        question_embedding = vectorize_text([question], batch_size=1)[0]
        print("问题向量化完成。")

        similar_videos = query_vector_collection(
            Config.VIDEO_NAME_COLLECTION_NAME,
            question_embedding,
            k=3,
            return_properties=["video_name", "video_base64"]
        )

        sorted_similar_videos = sorted(similar_videos, key=lambda x: x['certainty'], reverse=True)
        video_names = sorted_similar_videos[0].get("video_name")
        print("video_names:\n", video_names)

        entity_id_list = []
        entities_raw_info = query_entity_by_name(video_names)
        print("entities_raw_info:\n", entities_raw_info)

        results = [
            {
                "related_video": video.get("video_name"),
                "video_base64": video.get("video_base64"),
                "similarity_score": video.get("certainty")
            }
            for video in similar_videos
        ]

        if entities_raw_info:
            entity_id_list.append(extract_entity_id(entities_raw_info))

        return build_retrieval_result(results, "视频", None, entity_id_list if entity_id_list else None)

    except Exception as e:
        print(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


if __name__ == "__main__":
    import os
    uvicorn.run(app, host="0.0.0.0", port=7001)