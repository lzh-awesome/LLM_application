# 知识图谱问答服务

基于知识图谱和向量检索的智能问答系统，支持多模态问答、简洁/深入模式切换、实体识别等功能。

## 功能特性

- **知识图谱问答** - 基于实体识别和多跳推理的深度问答
- **简洁/深入模式** - 可切换回答风格，适应不同场景
- **问题推荐** - 根据上下文推荐相关问题
- **话题总结** - 对对话内容进行主题归纳
- **翻译服务** - 多语言翻译支持
- **实体查询** - 知识图谱实体信息检索
- **多模态问答** - 支持图片和视频问答

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI 应用层                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │  Chat   │ │ KG QA   │ │ Media   │ │ Entity  │ ...   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
├─────────────────────────────────────────────────────────┤
│                    服务层 (Services)                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │   LLM   │ │  vLLM   │ │ Embedding│ │ Reranker│       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│  ┌─────────────────┐ ┌─────────────────┐               │
│  │    ArangoDB     │ │    Weaviate     │               │
│  └─────────────────┘ └─────────────────┘               │
├─────────────────────────────────────────────────────────┤
│                    配置层 (Config)                       │
│              Pydantic + YAML 配置管理                   │
└─────────────────────────────────────────────────────────┘
```

## 项目结构

```
wenda/
├── app/
│   ├── main.py              # 应用入口
│   ├── config.py            # 配置管理
│   ├── dependencies.py      # 依赖注入
│   ├── routers/             # 路由模块
│   │   ├── chat.py          # 简洁/深入问答
│   │   ├── kg_wenda.py      # 知识图谱问答
│   │   ├── media.py         # 图片/视频问答
│   │   ├── entity.py        # 实体查询
│   │   ├── recommend.py     # 问题推荐
│   │   ├── topic.py         # 话题总结
│   │   └── translate.py     # 翻译服务
│   ├── services/            # 服务层
│   │   ├── llm.py           # LLM 服务
│   │   ├── vllm.py          # 多模态服务
│   │   ├── embedding.py     # 向量嵌入
│   │   ├── reranker.py      # 重排序服务
│   │   ├── ner.py           # 实体识别
│   │   ├── kg_qa.py         # KG QA 核心逻辑
│   │   └── database.py      # 数据库服务
│   ├── schemas/             # 数据模型
│   └── utils/               # 工具函数
│       ├── helpers.py       # 通用工具
│       ├── prompts.py       # 提示词模板
│       ├── streaming.py     # 流式响应工具
│       └── triple_utils.py  # 三元组处理
├── config.yaml              # 配置文件
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn openai weaviate-client python-arango pydantic yaml pillow aiohttp
```

### 2. 配置文件

编辑 `config.yaml`，配置各项服务：

```yaml
arango:
  host: http://localhost:8529
  database_name: wiki
  username: root
  password: root
  entity_collection: entity
  entity_description_collection: entity_description
  link_collection: link

llm:
  api_key: your-api-key
  base_url: http://localhost:8000/v1
  model: qwen3:32b
  ner_url: http://localhost:8100/ner_extract
  ollama_api_url: http://localhost:11434/api/generate

weaviate:
  http_host: localhost
  http_port: 8080
  grpc_host: localhost
  grpc_port: 50051
  api_key: your-weaviate-key

embedding_model:
  bge_m3_path: /path/to/bge-m3

reranker:
  model_path: /path/to/bge-reranker-v2-m3
  max_length: 512

vllm:
  api_base: http://localhost:7018/v1
  api_key: not-used
  model: MiniCPM-o-2_6

cors:
  origins:
    - http://localhost
    - '*'

logging:
  file: app.log
  level: DEBUG
```

### 3. 启动服务

```bash
# 默认端口 8104
python -m app.main

# 指定端口
python -m app.main --port 8105

# 指定配置文件
python -m app.main --config config_prod.yaml
```

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/brief_or_profound` | POST | 简洁/深入模式问答 |
| `/deepseek_kg_wenda_test` | POST | 知识图谱问答测试 |
| `/deepseek_kg_wenda_correct` | POST | 实体纠错问答 |
| `/kg_wenda_weaviate` | POST | Weaviate 版 KG 问答 |
| `/wenda/image` | POST | 图片问答 |
| `/wenda/video` | POST | 视频问答 |
| `/entity_query` | POST | 实体查询 |
| `/recommend_questions` | POST | 问题推荐 |
| `/topic_summary` | POST | 话题总结 |
| `/translate` | POST | 翻译服务 |
| `/health` | GET | 健康检查 |

### 请求示例

**简洁/深入问答**
```bash
curl -X POST http://localhost:8104/brief_or_profound \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "什么是知识图谱？"}], "is_concise": true}'
```

**知识图谱问答**
```bash
curl -X POST http://localhost:8104/deepseek_kg_wenda_test \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "张三的上级是谁？"}]}'
```

**图片问答**
```bash
curl -X POST http://localhost:8104/wenda/image \
  -F "media=@image.jpg" \
  -F "user_text=这张图片描述了什么"
```

**视频问答**
```bash
curl -X POST http://localhost:8104/wenda/video \
  -F "media=@video.mp4" \
  -F "user_text=请描述这段视频的内容"
```

## 技术栈

- **Web框架**: FastAPI + Uvicorn
- **图数据库**: ArangoDB
- **向量数据库**: Weaviate
- **LLM**: OpenAI 兼容 API (Qwen/DeepSeek)
- **多模态**: vLLM + MiniCPM-o-2_6
- **Embedding**: BGE-M3
- **Reranker**: BGE-Reranker-v2-m3
- **NER**: 自定义 NER 服务

## 核心流程

### 知识图谱问答流程

```
用户问题 → NER实体识别 → 向量检索实体 → 置信度判断
    ↓
高置信度(≥0.9) → 多跳推理 → 流式响应
    ↓
中置信度(0.5-0.9) → 返回候选实体列表 → 用户选择
    ↓
低置信度(<0.5) → 问答对检索 → LLM生成回答
```

## License

MIT License