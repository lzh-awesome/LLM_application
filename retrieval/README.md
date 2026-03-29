# 智能检索服务

基于知识图谱和向量数据库的多模态智能检索服务，支持文本、音频、图片、视频等多种检索方式。

## 服务架构

| 服务文件 | 端口 | 功能 |
|---------|------|------|
| `text_retrieval_new.py` | 7000 | 文本智能检索 |
| `image_video_retrieval_new.py` | 7001 | 图片/视频智能检索 |
| `audio_retrieval_new.py` | 7002 | 音频智能检索 |

## API 接口

### 文本检索服务 (端口 7000)

#### 1. 智能文本检索 `/int_retrieval_text`

根据用户问题检索相关实体，并生成流式回答。

**请求参数：**
```json
{
  "question": "国防大学的主要职能是什么？",
  "is_concise": true
}
```

| 参数 | 类型 | 说明 |
|-----|------|------|
| question | string | 用户问题（必填） |
| is_concise | boolean | 回答模式：true=简洁模式，false=深入模式 |

**响应（SSE 流式）：**
- `entity_name`: 最相关的实体名称
- `entity_id_list`: 实体ID列表
- `entity_certainty_list`: 置信度列表
- `llm_thinking_process`: LLM 生成的回答（流式）
- `full_response`: 完整回答内容

**示例：**
```bash
curl -X POST http://localhost:7000/int_retrieval_text \
  -H "Content-Type: application/json" \
  -d '{"question": "国防大学的主要职能是什么？", "is_concise": true}'
```

#### 2. 实体检索 `/retrieve_related_entities`

仅返回与问题相关的实体ID和名称，不生成回答。

**请求参数：**
```json
{
  "question": "国防大学"
}
```

**响应：**
```json
{
  "entity_id_list": ["Q12345", "Q67890"],
  "entity_name_list": ["国防大学", "中国人民解放军国防大学"]
}
```

---

### 图片/视频检索服务 (端口 7001)

#### 1. 图搜图 `/int_retrieval_image`

上传图片，检索相似图片并获取描述。

**请求：** multipart/form-data，字段 `image_file`

**响应：**
```json
{
  "message": "以下是我检索到的图片：",
  "results": [
    {
      "related_image": "图片名称",
      "related_image_id": "图片ID",
      "image_base64": "Base64编码",
      "similarity_score": 0.95
    }
  ],
  "image_description": "图片描述内容",
  "entity_id_list": ["Q12345"]
}
```

**示例：**
```bash
curl -X POST http://localhost:7001/int_retrieval_image \
  -F "image_file=@/path/to/your/image.jpg"
```

#### 2. 文搜图 `/text_retrieval_image`

根据自然语言描述检索相关图片。

**请求参数：**
```json
{
  "question": "一张展示军事装备的图片"
}
```

**示例：**
```bash
curl -X POST http://localhost:7001/text_retrieval_image \
  -H "Content-Type: application/json" \
  -d '{"question": "一张展示军事装备的图片"}'
```

#### 3. 视频搜视频 `/int_retrieval_video`

上传视频，检索相似视频并获取描述。

**请求：** multipart/form-data，字段 `media`

**示例：**
```bash
curl -X POST http://localhost:7001/int_retrieval_video \
  -F "media=@/path/to/your/video.mp4"
```

#### 4. 文搜视频 `/text_retrieval_video`

根据自然语言描述检索相关视频。

**请求参数：**
```json
{
  "question": "一段展示飞行训练的视频"
}
```

---

### 音频检索服务 (端口 7002)

#### 1. 音频智能检索 `/int_retrieval_audio`

上传音频文件，自动转文字、抽取实体、检索相关信息。

**请求：** multipart/form-data，字段 `audio`

**响应（SSE 流式）：**
- `entity_name_list`: 实体名称列表
- `entity_id_list`: 实体ID列表
- `chat.completion.chunk`: LLM 生成的描述（流式）
- `full_response`: 完整描述内容

**示例：**
```bash
curl -X POST http://localhost:7002/int_retrieval_audio \
  -F "audio=@/path/to/your/audio.mp3"
```

---

## 配置说明

各服务通过 `Config` 类管理配置，主要配置项：

### Weaviate 向量数据库
- `WEAVIATE_HTTP_HOST`: HTTP 主机地址
- `WEAVIATE_HTTP_PORT`: HTTP 端口 (8080)
- `WEAVIATE_GRPC_HOST`: gRPC 主机地址
- `WEAVIATE_GRPC_PORT`: gRPC 端口 (50051)
- `WEAVIATE_API_KEY`: API 密钥

### ArangoDB 图数据库
- `DB_HOST`: 数据库地址
- `DB_USERNAME`: 用户名
- `DB_PASSWORD`: 密码
- `DB_NAME`: 数据库名

### 模型路径
- `MODEL_PATH` / `TEXT_MODEL_PATH`: BGE-M3 文本向量模型
- `IMAGE_MODEL_PATH`: BGE-VL 图片向量模型
- `VIDEO_MODEL_PATH`: VideoCLIP-XL 视频向量模型

### 外部服务
- `OLLAMA_API_URL`: LLM API 地址
- `WHISPER_URL`: 音频转文字服务
- `LLM_NER_URL`: 实体抽取服务
- `IMAGE_DESCRIPTION_URL`: 图片描述服务
- `VIDEO_DESCRIPTION_URL`: 视频描述服务

---

## 依赖

```
fastapi
uvicorn
weaviate-client
sentence-transformers
transformers
arango
torch
opencv-python
numpy
aiohttp
requests
tqdm
```

---

## 启动服务

```bash
# 启动文本检索服务
python text_retrieval_new.py

# 启动图片/视频检索服务
python image_video_retrieval_new.py

# 启动音频检索服务
python audio_retrieval_new.py
```

或使用 uvicorn：
```bash
uvicorn text_retrieval_new:app --host 0.0.0.0 --port 7000
uvicorn image_video_retrieval_new:app --host 0.0.0.0 --port 7001
uvicorn audio_retrieval_new:app --host 0.0.0.0 --port 7002
```

---

## Python 调用示例

```python
import requests

# 文本检索
response = requests.post(
    "http://localhost:7000/int_retrieval_text",
    json={"question": "国防大学的主要职能", "is_concise": True}
)

# 图片检索
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:7001/int_retrieval_image",
        files={"image_file": f}
    )

# 音频检索
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:7002/int_retrieval_audio",
        files={"audio": f}
    )
```