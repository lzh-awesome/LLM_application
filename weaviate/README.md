# Weaviate 向量库管理工具

一个模块化的 Weaviate 向量库管理工具，支持文本、图片、视频等多种数据类型的存储和检索。

## 功能特性

- **多数据类型支持**：实体、关系（三元组）、图片、视频
- **多种检索方式**：向量相似度检索、置信度阈值检索、属性过滤检索
- **灵活配置**：支持 YAML 配置文件和环境变量
- **批量处理**：支持批量导入、去重、进度显示
- **模块化设计**：清晰的职责划分，易于扩展

## 目录结构

```
weaviate/
├── config/                 # 配置管理
│   ├── __init__.py
│   ├── settings.py         # 配置加载（YAML + 环境变量）
│   └── config.yaml         # 配置文件模板
│
├── core/                   # 核心模块
│   ├── __init__.py
│   ├── client.py           # 数据库客户端管理
│   └── operations.py       # 通用 CRUD 操作
│
├── embedding/              # 向量化模块
│   ├── __init__.py
│   ├── text.py             # 文本向量化（BGE-M3）
│   ├── image.py            # 图片向量化（BGE-VL）
│   └── video.py            # 视频向量化（VideoCLIP-XL）
│
├── collections/            # 集合管理模块
│   ├── __init__.py
│   ├── base.py             # 集合基类
│   ├── entity.py           # 实体集合
│   ├── link.py             # 关系（三元组）集合
│   ├── image.py            # 图片集合
│   └── video.py            # 视频集合
│
├── scripts/                # 脚本入口
│   ├── add_entity.py       # 添加实体
│   ├── add_link.py         # 添加关系
│   ├── delete_entity.py    # 删除实体
│   ├── delete_link.py      # 删除关系
│   ├── query_entity.py     # 查询实体
│   ├── query_link.py       # 查询关系
│   ├── import_images.py    # 批量导入图片
│   └── import_videos.py    # 批量导入视频
│
├── tests/
│   └── test_connection.py  # 连接测试
│
└── main.py                 # 主入口（集合管理）
```

## 安装依赖

```bash
pip install weaviate-client sentence-transformers transformers torch tqdm pyyaml arango-client opencv-python numpy
```

## 配置

### 方式一：YAML 配置文件

复制 `config/config.yaml` 并修改：

```yaml
# Weaviate 数据库配置
weaviate:
  http_host: "weaviate"
  http_port: 8080
  http_secure: false
  grpc_host: "weaviate"
  grpc_port: 50051
  grpc_secure: false
  api_key: "test-secret-key"

# ArangoDB 数据库配置
arango:
  host: "http://10.117.254.37:8529"
  database: "wiki"
  username: "root"
  password: "root"

# 模型路径配置
models:
  bge_m3_path: "/path/to/bge-m3"
  bge_vl_path: "/path/to/BGE-VL-base"
  video_clip_path: "/path/to/VideoCLIP-XL"
  video_clip_weights: "/path/to/VideoCLIP-XL.bin"

# 批处理大小
batch_size: 1000
```

### 方式二：环境变量

```bash
# Weaviate
export WEAVIATE_HTTP_HOST="weaviate"
export WEAVIATE_HTTP_PORT="8080"
export WEAVIATE_API_KEY="test-secret-key"

# ArangoDB
export ARANGO_HOST="http://10.117.254.37:8529"
export ARANGO_DATABASE="wiki"
export ARANGO_USERNAME="root"
export ARANGO_PASSWORD="root"

# 模型路径
export BGE_M3_PATH="/path/to/bge-m3"
export BGE_VL_PATH="/path/to/BGE-VL-base"
```

## 使用示例

### 1. 测试连接

```bash
python tests/test_connection.py
```

### 2. 集合管理

```bash
# 列出所有集合
python main.py list

# 测试连接
python main.py test

# 删除集合
python main.py delete collection_name
```

### 3. 实体管理

#### 添加实体

```bash
# 从 ArangoDB 导入实体
python scripts/add_entity.py --arango-collection entity_description --weaviate-collection kg_wenda --create

# 从 ArangoDB 的其他集合导入
python scripts/add_entity.py --arango-collection entity_description_large --weaviate-collection kg_wenda_large --create
```

#### 查询实体

```bash
# 查询实体（返回 Top-K）
python scripts/query_entity.py "搜索关键词" -k 10

# 按置信度阈值查询
python scripts/query_entity.py "搜索关键词" --certainty 0.8

# 指定集合
python scripts/query_entity.py "搜索关键词" --collection kg_wenda_large -k 5
```

#### 删除实体

```bash
# 删除单个实体
python scripts/delete_entity.py --id "entity_description/12345"

# 批量删除
python scripts/delete_entity.py --ids "id1,id2,id3"

# 指定集合
python scripts/delete_entity.py --id "xxx" --collection kg_wenda_large
```

### 4. 关系（三元组）管理

#### 添加关系

```bash
# 从 ArangoDB 导入关系
python scripts/add_link.py --link-collection link --entity-collection entity --weaviate-collection natural_triples --create
```

#### 查询关系

```bash
# 查询相关三元组
python scripts/query_link.py "北京的人口是多少"

# 指定集合和返回数量
python scripts/query_link.py "某个问题" --collection kg_triples -k 15
```

#### 删除关系

```bash
python scripts/delete_link.py --id "link/12345"
```

### 5. 图片管理

#### 批量导入图片

```bash
# 导入图片目录（自动创建图搜图和文搜图两个集合）
python scripts/import_images.py /path/to/images --create-image --create-name
```

图片目录结构要求：
```
/images/
├── 类别1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── 类别2/
│   ├── image1.jpg
│   └── ...
└── ...
```

#### 代码调用示例

```python
from config import get_settings
from core import WeaviateClient
from embedding import TextEmbedding, ImageEmbedding
from collections import ImageCollection, ImageNameCollection

# 初始化
settings = get_settings()
client = WeaviateClient(settings.weaviate)
text_embedding = TextEmbedding(settings.models.bge_m3_path)
image_embedding = ImageEmbedding(settings.models.bge_vl_path)

# 图搜图
image_coll = ImageCollection(client.client)
query_vector = image_embedding.encode_single("/path/to/query.jpg")
results = image_coll.query(query_vector, k=10)

# 文搜图
name_coll = ImageNameCollection(client.client)
query_vector = text_embedding.encode_single("搜索关键词")
results = name_coll.query(query_vector, k=10)

# 关闭连接
client.close()
```

### 6. 视频管理

#### 批量导入视频

```bash
# 导入视频目录
python scripts/import_videos.py /path/to/videos --create-video --create-name
```

视频目录结构要求：
```
/videos/
├── 类别1/
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
├── 类别2/
│   └── ...
└── ...
```

#### 代码调用示例

```python
from config import get_settings
from core import WeaviateClient
from embedding import TextEmbedding, VideoEmbedding
from collections import VideoCollection, VideoNameCollection

# 初始化
settings = get_settings()
client = WeaviateClient(settings.weaviate)
text_embedding = TextEmbedding(settings.models.bge_m3_path)
video_embedding = VideoEmbedding(
    settings.models.video_clip_path,
    settings.models.video_clip_weights
)

# 视频搜视频
video_coll = VideoCollection(client.client)
query_vector = video_embedding.encode_single("/path/to/query.mp4")
results = video_coll.query(query_vector, k=10)

# 文搜视频
name_coll = VideoNameCollection(client.client)
query_vector = text_embedding.encode_single("搜索关键词")
results = name_coll.query(query_vector, k=10)

client.close()
```

## API 参考

### VectorStoreOperations（通用操作类）

```python
from core import VectorStoreOperations

ops = VectorStoreOperations(client, "collection_name")

# 集合管理
ops.exists()                          # 检查集合是否存在
ops.create(properties, description)   # 创建集合
ops.delete()                          # 删除集合

# 数据操作
ops.insert(properties, vector)        # 插入单个文档
ops.batch_insert(docs, vectors)       # 批量插入

# 查询
ops.query_by_vector(vector, k=10)     # Top-K 查询
ops.query_by_certainty(vector, 0.8)   # 置信度查询
ops.query_by_filter(field, value)     # 属性过滤查询

# 删除
ops.delete_by_id(id_value)            # 根据 ID 删除
```

### EntityCollection（实体集合）

```python
from collections import EntityCollection

entity_coll = EntityCollection(client, "kg_wenda")

# 创建集合
entity_coll.create_collection()

# 添加实体
entities = [{"_id": "1", "name": "实体名", "desc": "描述"}]
vectors = text_embedding.encode([e["name"] for e in entities])
entity_coll.add(entities, vectors)

# 查询
results = entity_coll.query(vector, k=10)
results = entity_coll.query_by_certainty(vector, min_certainty=0.8)

# 删除
entity_coll.delete(entity_id)
```

### LinkCollection（关系集合）

```python
from collections import LinkCollection

link_coll = LinkCollection(client, "natural_triples")

# 添加关系
triples = [{
    "_id": "link/1",
    "triple_name": "北京的首都是中国",
    "_from": "entity1_id",
    "_to": "entity2_id"
}]
vectors = text_embedding.encode([t["triple_name"] for t in triples])
link_coll.add(triples, vectors)

# 查询
results = link_coll.query(vector, k=10)
```

## 集合说明

| 集合类型 | 默认集合名 | 用途 | 属性 |
|---------|-----------|------|------|
| EntityCollection | kg_wenda | 实体检索 | id_, name, desc |
| LinkCollection | natural_triples | 关系检索 | id_, triple_name, _from, _to |
| ImageCollection | image_retrieval | 图搜图 | image_name, image_id, image_base64, image_md5 |
| ImageNameCollection | image_name_retrieval | 文搜图 | image_name, image_base64, image_md5 |
| VideoCollection | video_retrieval | 视频搜视频 | video_name, video_base64, video_md5 |
| VideoNameCollection | video_name_retrieval | 文搜视频 | video_name, video_base64, video_md5 |

## 扩展新数据类型

1. 在 `embedding/` 下创建新的向量化类
2. 在 `collections/` 下继承 `BaseCollection` 创建新的集合类
3. 在 `scripts/` 下创建对应的脚本

示例：

```python
# collections/audio.py
from .base import BaseCollection

class AudioCollection(BaseCollection):
    COLLECTION_NAME = "audio_retrieval"
    DESCRIPTION = "音频集合"
    PROPERTIES = [
        {"name": "audio_name", "dataType": ["text"], ...},
        {"name": "audio_md5", "dataType": ["text"], ...},
    ]

    def add(self, audios, vectors, dedup=True):
        # 实现添加逻辑
        pass

    def query(self, vector, k=10):
        return self.query_by_vector(vector, k)

    def delete(self, audio_id):
        return self.ops.delete_by_id(audio_id, "audio_name")
```

## 注意事项

1. **模型加载**：向量化模型会在首次使用时延迟加载，避免内存浪费
2. **去重机制**：图片和视频默认基于 MD5 去重，实体和关系基于 ID 去重
3. **批处理**：大数据量导入时会自动分批处理，避免内存溢出
4. **连接管理**：Weaviate 客户端使用单例模式，全局共享连接

## License

MIT