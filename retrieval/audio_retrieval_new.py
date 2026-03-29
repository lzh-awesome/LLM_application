# 智能检索的音频检索接口
# 端口号：7002
# 功能：
# 1、调用接口将上传的音频转成文字
# 2、然后抽取文字中的实体信息
# 3、去weaviate实体向量库中进行实体检索
# 4、将返回的entity_id、entity_name丢给前端

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from arango import ArangoClient
import weaviate
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer

import time
import json
import uuid
import uvicorn
import requests


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
    COLLECTION_NAME = "entity_description_large"

    # 模型配置
    MODEL_PATH = "/workspace/codes/deepseek/deepseek_model/bge-m3"

    # API配置
    WHISPER_URL = "http://10.117.1.238:8002/whisper2text"
    LLM_NER_URL = "http://10.117.1.238:8100/ner_extract"
    LLM_MODEL = "qwen3:32b"

    # 向量集合名称
    VECTOR_COLLECTION_NAME = "kg_wenda_large"


# 初始化 FastAPI 应用
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
model = SentenceTransformer(Config.MODEL_PATH)

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

# ArangoDB连接
db = None


def init_connections():
    """初始化数据库连接"""
    global db
    try:
        arango_client = ArangoClient(hosts=Config.DB_HOST)
        db = arango_client.db(Config.DB_NAME, username=Config.DB_USERNAME, password=Config.DB_PASSWORD)
        print("数据库初始化成功")
    except Exception as e:
        print(f"初始化连接失败: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化连接"""
    init_connections()


def query_entity_collection(collection_name: str, vector, k: int) -> list:
    """查询实体向量集合"""
    collection = client.collections.get(collection_name)
    try:
        response = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            return_properties=["id_", "name", "desc"],
            return_metadata=["distance", "certainty"],
        )
        results = []
        for obj in response.objects:
            results.append({
                "id_": obj.properties.get("id_"),
                "name": obj.properties.get("name"),
                "desc": obj.properties.get("desc"),
                "certainty": obj.metadata.certainty,
            })
        return results
    except Exception as e:
        print(f"查询异常: {e}")
        return []


def _generate_chat_chunks(message: str, entity_name_list=None, entity_id_list=None):
    """生成流式响应的chunk数据"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())

    def stream():
        # 发送实体信息
        if entity_name_list is not None:
            yield f"data: {json.dumps({'entity_name_list': entity_name_list}, ensure_ascii=False)}\n\n"
        if entity_id_list is not None:
            yield f"data: {json.dumps({'entity_id_list': entity_id_list}, ensure_ascii=False)}\n\n"

        # 逐字符发送
        for char in message:
            chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": Config.LLM_MODEL,
                "choices": [{
                    "delta": {"content": char},
                    "index": 0,
                    "finish_reason": None
                }],
                "service_tier": None,
                "system_fingerprint": None
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # 发送结束标记
        end_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": Config.LLM_MODEL,
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }],
            "service_tier": None,
            "system_fingerprint": None
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"

        # 发送完整响应
        yield f"data: {json.dumps({'full_response': message}, ensure_ascii=False)}\n\n"

    return stream()


def _generate_error_chunks(error_msg: str):
    """生成错误响应的流式格式"""
    error_chunk = {
        "id": f"chatcmpl-error-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": Config.LLM_MODEL,
        "choices": [{
            "delta": {"content": f"错误: {error_msg}"},
            "index": 0,
            "finish_reason": "stop"
        }]
    }

    def stream():
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

    return stream()


@app.post("/int_retrieval_audio")
async def int_retrieval_audio(audio: UploadFile = File(...)):
    try:
        print("````````````````````开始音频智能检索`````````````````````")

        # 1. 读取上传的音频文件并调用Whisper转文字
        audio_content = await audio.read()
        files = {"audio": (audio.filename, audio_content, audio.content_type)}
        response = requests.post(Config.WHISPER_URL, files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Whisper service failed: {response.text}")

        whisper_response = response.json()
        audio_text = whisper_response.get("音频文本", "")
        print("audio_text:", audio_text)

        # 2. 调用NER抽取实体
        payload = {"text": audio_text}
        headers = {"Content-Type": "application/json"}
        response = requests.post(Config.LLM_NER_URL, json=payload, headers=headers)
        llm_ner_entities = response.json().get("result")
        print("\nllm_ner_entities:\n", llm_ner_entities)

        # 3. 根据抽取结果处理
        if '无相关XX名' in llm_ner_entities or len(llm_ner_entities) > 1:
            # 抽不到实体或抽到多个实体，直接描述音频内容
            return StreamingResponse(
                _generate_chat_chunks(audio_text, entity_name_list=[], entity_id_list=[]),
                media_type="text/event-stream"
            )
        else:
            # 抽到实体，进行向量检索
            query_vector = model.encode([llm_ner_entities[0]])[0]
            entities = query_entity_collection(Config.VECTOR_COLLECTION_NAME, query_vector, k=5)

            sorted_entities = sorted(entities, key=lambda x: x['certainty'], reverse=True)

            sorted_name_list = []
            sorted_id_list = []
            for entity in sorted_entities:
                sorted_name_list.append(entity.get("name"))
                entity_id = entity.get('id_')
                converted_entity_id = f"entity/{entity_id.split('/')[1]}"
                sorted_id_list.append(converted_entity_id)

            print("entity_name_list:\n", sorted_name_list)
            print("entity_id_list:\n", sorted_id_list)

            return StreamingResponse(
                _generate_chat_chunks(audio_text, entity_name_list=sorted_name_list, entity_id_list=sorted_id_list),
                media_type="text/event-stream"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频检索过程中发生错误: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7002)