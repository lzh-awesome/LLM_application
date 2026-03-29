# 智能检索的文本检索接口
# 端口号：7000
# 功能：
# 1、获取与文本信息相关的id; 返回值：entity_id列表
# 2、获取文本信息语义解析的接口; 返回值：流式的答案输出

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import weaviate
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer

import time
import json
import aiohttp
import asyncio
import uvicorn


# 配置类
class Config:
    # Weaviate连接参数
    WEAVIATE_HTTP_HOST = "weaviate"
    WEAVIATE_HTTP_PORT = 8080
    WEAVIATE_GRPC_HOST = "weaviate"
    WEAVIATE_GRPC_PORT = 50051
    WEAVIATE_API_KEY = "test-secret-key"

    # 模型配置
    MODEL_PATH = "/workspace/codes/deepseek/deepseek_model/bge-m3"

    # LLM配置
    OLLAMA_API_URL = "http://10.117.1.238:7003/api/generate"
    LLM_MODEL = "qwen3:32b"

    # 向量集合名称
    COLLECTION_NAME = "kg_wenda_large"


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


# 简洁模式提示词模版
CONCISE_PROMPT_TEMPLATE = """
你是一个知识图谱辅助推理助手，需要基于提供的实体名称、实体描述信息，并结合自身知识库回答用户问题。请严格按照以下规则执行：

- 以下是与问题相关的实体信息：
  - 相关的实体信息：
    {Entity_Info}

- 回答要求：
  1. 优先使用提供的实体信息；
  2. 当信息不足时，仅说明限制条件，无需展开解释；
  3. 使用最简短的语言组织答案，不超过五句话。

- 请根据以上信息回答以下问题：
  问题：{question}
  答案：
/no_think
"""

# 深入模式提示词模版
DETAILED_PROMPT_TEMPLATE = """
你是一个知识图谱辅助推理助手，需要基于提供的实体名称、实体描述信息，并结合自身知识库回答用户问题。请严格按照以下规则执行：

- 以下是与问题相关的实体信息：
  - 相关的实体信息：
    {Entity_Info}

- 回答要求：
  1. 优先使用提供的实体信息；
  2. 当信息不足时，主动说明限制条件，并尝试从其他角度补充相关信息；
  3. 使用自然语言组织答案，确保逻辑清晰；
  4. 如果问题涉及多个方面，请分点说明；
  5. 包括相关的扩展内容或建议。

- 请根据以上信息回答以下问题：
  问题：{question}
  答案：
/no_think
"""


def query_vector_collection_by_k(collection_name: str, vector, k: int) -> list:
    """根据k值查询向量集合，返回固定数量的结果"""
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


def query_vector_collection_by_certainty(collection_name: str, vector, min_certainty: float = 0.8):
    """根据certainty阈值查询向量集合，动态返回高相似度结果"""
    collection = client.collections.get(collection_name)
    try:
        response = collection.query.near_vector(
            near_vector=vector,
            return_properties=["id_", "name", "desc"],
            return_metadata=["distance", "certainty"],
            certainty=min_certainty,
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


async def call_ollama_stream(prompt, model_name=Config.LLM_MODEL):
    """调用 Ollama API，生成文本并实现流式输出"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "stream": True
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(Config.OLLAMA_API_URL, json=payload) as response:
                if response.status != 200:
                    error_message = f"请求失败: 状态码 {response.status}, 响应内容 {await response.text()}"
                    print(error_message)
                    yield {"error": error_message}
                    return

                async for line in response.content:
                    if line:
                        try:
                            data = line.decode("utf-8").strip()
                            chunk = json.loads(data)
                            yield chunk.get("response", "")
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        error_message = f"请求异常: {str(e)}"
        print(error_message)
        yield {"error": error_message}


async def call_chat_completions(prompt, entity_id_list, entity_name_list, entity_certainty_list):
    """调用大模型并处理流式响应"""
    print("prompt:", prompt)

    async def generate_stream():
        try:
            yield f"data: {json.dumps({'entity_name': entity_name_list[0]}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'entity_id_list': entity_id_list}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'entity_certainty_list': entity_certainty_list}, ensure_ascii=False)}\n\n"

            full_response = ""
            async for new_text in call_ollama_stream(prompt):
                if isinstance(new_text, dict) and "error" in new_text:
                    yield f"data: {json.dumps(new_text, ensure_ascii=False)}\n\n"
                    break

                full_response += new_text
                chunk = {
                    "id": "llm_thinking_process",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": Config.LLM_MODEL,
                    "choices": [{
                        "delta": {"content": new_text},
                        "index": 0,
                        "finish_reason": None,
                        "logprobs": None,
                    }],
                    "service_tier": None,
                    "system_fingerprint": None
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.001)

            yield f"data: {json.dumps({'full_response': full_response}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_message = f"Streaming error: {e}"
            print(error_message)
            yield f"data: {json.dumps({'error': error_message}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/int_retrieval_text")
async def kg_wenda(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        is_concise = data.get("is_concise", True)

        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        start_time = time.time()

        query_vector = model.encode([question])[0]
        entities = query_vector_collection_by_certainty(Config.COLLECTION_NAME, query_vector, min_certainty=0.7)

        print("len(entities):", len(entities))
        sorted_entities = sorted(entities, key=lambda x: x['certainty'], reverse=True)

        end_time = time.time()
        print(f"Query took {end_time - start_time:.2f} seconds.")

        entity_info = "Entity Info:\n"
        for entity in sorted_entities[:3]:
            entity_info += f"- {entity['name']}: {entity['desc']}\n"

        print("entity_info:\n", entity_info)

        if is_concise:
            prompt = CONCISE_PROMPT_TEMPLATE.format(Entity_Info=entity_info, question=question)
        else:
            prompt = DETAILED_PROMPT_TEMPLATE.format(Entity_Info=entity_info, question=question)

        entity_id_list = []
        entity_name_list = []
        entity_certainty_list = []

        for entity in sorted_entities:
            entity_ids = entity['id_'].replace('entity_description/', '')
            entity_id_list.append(entity_ids)
            entity_name_list.append(entity['name'])
            entity_certainty_list.append(entity['certainty'])

        print("entity_id_list:", entity_id_list)
        print("entity_name_list:", entity_name_list)
        print("entity_certainty_list:", entity_certainty_list)

        return await call_chat_completions(prompt, entity_id_list, entity_name_list, entity_certainty_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve_related_entities")
async def retrieve_related_entities(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")

        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        start_time = time.time()

        query_vector = model.encode([question])[0]
        entities = query_vector_collection_by_k("kg_wenda", query_vector, k=10)

        print("len(entities):", len(entities))
        sorted_entities = sorted(entities, key=lambda x: x['certainty'], reverse=True)

        end_time = time.time()
        print(f"Query took {end_time - start_time:.2f} seconds.")

        entity_id_list = []
        entity_name_list = []

        for entity in sorted_entities:
            entity_ids = entity['id_'].replace('entity_description/', '')
            entity_id_list.append(entity_ids)
            entity_name_list.append(entity['name'])

        print("entity_id_list:", entity_id_list)
        print("entity_name_list:", entity_name_list)

        return {"entity_id_list": entity_id_list, "entity_name_list": entity_name_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)