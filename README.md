# LLM 应用开发项目

本项目包含四个核心模块，提供智能问答、智能检索、智能报告以及weaviate向量库管理功能。

## 项目结构

```
LLM/
├── wenda/          # 智能问答模块
├── retrieval/      # 智能检索模块
├── report/         # 智能报告模块
└── weaviate/       # 向量库操作模块
```

---

## 1. 智能问答模块 (wenda)

基于 FastAPI 构建的智能问答服务，提供多种问答能力。

### 主要功能

| 功能 | 接口 | 说明 |
|------|------|------|
| 简洁/深入问答 | `/brief_or_profound` | 支持简洁模式和深入模式两种回答风格 |
| 知识图谱问答 | `/deepseek_kg_wenda_test` | 结合知识图谱的多跳推理问答 |
| 知识图谱纠错 | `/deepseek_kg_wenda_correct` | 用户实体选择纠错后继续问答 |
| Weaviate问答 | `/kg_wenda_weaviate` | Weaviate向量库版本的知识图谱问答 |
| 话题总结 | `/topic_summary` | 对话话题自动总结 |
| 问题推荐 | `/recommend_questions` | 基于上下文的问题推荐 |
| 翻译服务 | `/translate` | 多语言翻译 |
| 实体查询 | `/entity_query` | 实体信息查询 |
| 媒体检索 | `/media_search` | 多媒体资源检索 |

### 核心特性

- **多跳推理**: 基于知识图谱的关系推理，自动扩展查询路径
- **NER实体提取**: LLM驱动的实体识别，自动定位查询主体
- **工具调用**: 支持研究报告、图片生成/搜索、视频搜索、音频检索等工具
- **置信度分级**: 根据实体匹配分数自动选择最佳回答策略
- **流式输出**: SSE流式响应，实时返回生成内容

### 技术栈

- FastAPI + Uvicorn
- Weaviate 向量数据库
- BGE-M3 嵌入模型
- ArangoDB 图数据库
- LangChain-Chatchat 推理框架

### 默认端口

`8104`

---

## 2. 智能检索模块 (retrieval)

提供文本、音频、图像、视频等多模态检索服务。

### 模块组成

| 文件 | 功能 | 端口 |
|------|------|------|
| `text_retrieval_new.py` | 文本语义检索 | 7000 |
| `audio_retrieval_new.py` | 音频声纹检索 | - |
| `image_video_retrieval_new.py` | 图像/视频检索 | - |

### 主要功能

- **文本检索**: 基于 BGE-M3 向量嵌入的语义检索
- **实体关联**: 返回相关实体ID和名称列表
- **流式答案**: 结合 LLM 的语义解析和答案生成
- **阈值控制**: 支持 certainty 阈值和固定数量两种查询模式

### 接口说明

- `/int_retrieval_text`: 文本智能检索问答
- `/retrieve_related_entities`: 仅返回相关实体列表

### 技术栈

- Weaviate 向量数据库 (gRPC + HTTP)
- SentenceTransformer (BGE-M3)
- Ollama LLM API
- FastAPI SSE 流式响应

---

## 3. 智能报告模块 (report)

分步骤生成版本，结合固定模板、数据库和知识库进行报告生成。

### 版本

`v0.0.3` - 分步骤生成版本

### 主要接口

| 接口 | 功能 |
|------|------|
| `/generateOutline` | 根据内容生成政府公文式大纲 |
| `/outlineOverview` | 大纲扩写，添加概括性描述 |
| `/paragraph_generator` | 根据标题和概述生成正文段落 |
| `/generate_complete_content` | 一键生成完整内容（概述+正文） |
| `/topic_and_paragraph` | 主题描述和章节内容批量生成 |
| `/chat/completions` | 通用聊天补全 |
| `/detect` | 图像检测与检索 |

### 核心特性

- **知识增强**: 关键词提取 + NER实体识别 + 向量检索增强
- **分步生成**: 大纲→扩写→段落的渐进式生成流程
- **表格生成**: 自动查询实体描述并生成 Markdown 表格
- **配置管理**: 支持 YAML 配置文件热更新
- **服务管理**: 服务状态监控与远程重启
- **流式输出**: 全流程 SSE 流式响应

### 技术栈

- LangChain (ChatOpenAI)
- Weaviate 向量库
- ArangoDB 文档数据库
- CrossEncoder 重排序模型
- BGE-M3 嵌入模型
- Ollama / vLLM 推理框架

### 默认端口

`5003`

---

## 4. 向量库操作模块 (weaviate)

Weaviate 向量数据库管理工具，提供完整的 CRUD 操作。

### 目录结构

```
weaviate/
├── config/         # 配置管理
├── core/           # 核心客户端
├── embedding/      # 嵌入服务 (文本/图像/视频)
├── collections/    # 集合操作 (实体/链接/图像/视频)
├── scripts/        # 命令行脚本
└── tests/          # 测试文件
```

### CLI 命令

```bash
# 列出所有集合
python main.py list

# 测试连接
python main.py test

# 删除集合
python main.py delete <collection_name>
```

### 脚本工具

| 脚本 | 功能 |
|------|------|
| `add_entity.py` | 添加实体数据 |
| `add_link.py` | 添加关系链接 |
| `delete_entity.py` | 删除实体 |
| `delete_link.py` | 删除链接 |
| `query_entity.py` | 查询实体 |
| `query_link.py` | 查询链接 |
| `import_images.py` | 批量导入图像 |
| `import_videos.py` | 批量导入视频 |

### 支持的集合类型

- **entity**: 实体集合
- **link**: 关系链接集合
- **image**: 图像集合
- **video**: 视频集合

### 技术栈

- Weaviate Client (Python SDK v4)
- ArangoDB 客户端
- Pydantic 配置模型

---

## 环境依赖

### 数据库

- **Weaviate**: 向量数据库 (HTTP: 8080, gRPC: 50051)
- **ArangoDB**: 图数据库

### 模型服务

- **嵌入模型**: BGE-M3
- **重排序模型**: CrossEncoder
- **LLM**: Ollama / vLLM (支持多种模型接入)

### Python 依赖

- fastapi, uvicorn
- weaviate-client
- sentence-transformers
- langchain-openai
- arango
- aiohttp
- pyyaml

---

## 快速启动

```bash
# 智能问答服务
cd wenda && python app/main.py --port 8104

# 文本检索服务
cd retrieval && python text_retrieval_new.py

# 智能报告服务
cd report && python ai_report_refactored.py

# 向量库管理
cd weaviate && python main.py test
```

---

## 配置说明

各模块通过 YAML 配置文件管理参数：

- `wenda`: 配置嵌入模型、LLM服务地址、数据库连接等
- `report`: `config.yaml` - API地址、模型路径、数据库参数
- `weaviate`: `settings.py` - Weaviate/ArangoDB 连接参数

---

## API 文档

服务启动后访问 Swagger UI：

- 智能问答: `http://localhost:8104/docs`
- 文本检索: `http://localhost:7000/docs`
- 智能报告: `http://localhost:5003/docs`
