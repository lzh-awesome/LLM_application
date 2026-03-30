# 智能报告生成服务

基于 FastAPI 的智能报告生成系统，结合固定模版、数据库和挂载知识库数据进行知识增强后的报告生成。

## 版本信息

- 当前版本: v0.0.3

## 功能概述

本服务提供以下核心功能：

| 接口 | 功能描述 |
|------|----------|
| `/generateOutline` | 根据输入内容生成政府公文式标题大纲 |
| `/outlineOverview` | 对大纲进行扩写，添加概括性描述 |
| `/paragraph_generator` | 根据标题和概述生成完整段落内容 |
| `/topic_and_paragraph` | 主题和章节内容批量生成 |
| `/generate_complete_content` | 整合生成：标题→概述→正文 |
| `/detect` | 图像检测与检索 |
| `/chat/completions` | 聊天补全接口（流式响应） |
| `/health` | 健康检查 |
| `/status` | 服务状态查询 |
| `/config` | 获取当前配置 |
| `/update-config` | 更新服务配置 |

## 技术架构

### 核心依赖

- **FastAPI**: Web服务框架
- **Ollama**: 大语言模型服务（默认 qwen3:32b）
- **ArangoDB**: 实体数据存储
- **Weaviate**: 向量数据库（知识检索）
- **BGE-M3**: 文本嵌入模型
- **BGE-Reranker-v2-m3**: 重排序模型

### 知识增强流程

```
输入内容 → 关键词提取 → NER实体识别 → 向量检索 → 知识召回 → 报告生成
```

## 安装与配置

### 1. 环境准备

```bash
# 创建conda环境
conda create -n rag python=3.10
conda activate rag

# 安装依赖
pip install fastapi uvicorn aiohttp torch transformers
pip install sentence-transformers weaviate-client arango
pip install langchain-openai pydantic yaml psutil
```

### 2. 模型准备

下载以下模型到指定目录：
- BGE-M3 嵌入模型 → `./bge-m3`
- BGE-Reranker-v2-m3 → `./bge-reranker-v2-m3`

### 3. 配置文件

修改 `config.yaml` 配置各项服务地址：

```yaml
api:
  server_address: http://your-ollama-server:7003
  keyword_extract_url: http://your-server:8100/llm_topic_and_key_word_extract
  ner_extract_url: http://your-server:8100/ner_extract
  detect_api_url: http://your-server:8002/YOLO
  retrieval_api_url: http://your-server:7001/int_retrieval_image

models:
  brief_model: qwen3:32b      # 简洁模式使用的模型
  normal_model: qwen3:32b     # 正常模式使用的模型
  bge_model_path: ./bge-m3
  reranker_model_path: ./bge-reranker-v2-m3

arango:
  host: http://your-arango-server:8529
  database: wikidata
  username: root
  password: root

weaviate:
  host: 127.0.0.1
  http_port: 8080
  grpc_port: 50051
  api_key: your-api-key
```

## 启动服务

```bash
# 直接启动
python ai_report_refactored.py

# 或使用 uvicorn
uvicorn ai_report_refactored:app --host 0.0.0.0 --port 5003
```

服务默认运行在端口 **5003**。

## API 使用示例

### 生成大纲

```bash
curl -X POST http://localhost:5003/generateOutline \
  -H "Content-Type: application/json" \
  -d '{
    "content": "美军作战点名系统的设计与研究",
    "is_brief": true,
    "direct_reply": false
  }'
```

响应格式（流式）：
```
data: {"id":"chatcmpl","object":"chat.completion.chunk","choices":[{"delta":{"content":"## 主标题..."},"index":0}]}
data: [DONE]
```

### 大纲扩写

```bash
curl -X POST http://localhost:5003/outlineOverview \
  -H "Content-Type: application/json" \
  -d '{
    "generate_outline": "## 美军作战点名系统\n### 一、研究背景\n#### 1.1 背景",
    "is_brief": false
  }'
```

### 段落生成

```bash
curl -X POST http://localhost:5003/paragraph_generator \
  -H "Content-Type: application/json" \
  -d '{
    "title": "研究背景",
    "paragraph": "随着信息化战争的发展...",
    "is_brief": false
  }'
```

### 完整内容生成

```bash
curl -X POST http://localhost:5003/generate_complete_content \
  -H "Content-Type: application/json" \
  -d '{
    "title": "网络空间作战能力建设研究",
    "is_brief": false
  }'
```

### 健康检查

```bash
curl http://localhost:5003/health
```

响应：
```json
{
  "status": "healthy",
  "version": "0.0.3",
  "config": {
    "brief_model": "qwen3:32b",
    "normal_model": "qwen3:32b"
  }
}
```

### 更新配置

```bash
curl -X POST http://localhost:5003/update-config \
  -H "Content-Type: application/json" \
  -d '{
    "server_address": "http://new-server:7003",
    "brief_model": "new-model",
    "restart_service": true
  }'
```

## 请求参数说明

### 公共参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `is_brief` | bool | 是否使用简洁模式（减少知识检索） |
| `direct_reply` | bool | 是否返回完整响应（而非流式） |

### generateOutline

| 参数 | 必填 | 说明 |
|------|------|------|
| `content` | 是 | 输入的主题内容 |

### outlineOverview

| 参数 | 必填 | 说明 |
|------|------|------|
| `generate_outline` | 是 | 待扩写的大纲内容 |

### paragraph_generator

| 参数 | 必填 | 说明 |
|------|------|------|
| `title` | 是 | 标题 |
| `paragraph` | 是 | 概述内容 |

### topic_and_paragraph

| 参数 | 必填 | 说明 |
|------|------|------|
| `title` | 是 | 主题名称 |
| `chapters` | 是 | 章节列表 `[{chapterName, order}]` |
| `table` | 否 | 表格定义 `[{columns, rows}]` |

## 日志

服务运行日志保存在 `app.log` 文件中。

## 注意事项

1. 确保 Ollama 服务已启动并加载相应模型
2. 确保 ArangoDB 和 Weaviate 服务可用
3. GPU 环境可加速嵌入模型推理
4. 服务依赖 `prompt.py` 文件中的 prompt1/prompt2 模板