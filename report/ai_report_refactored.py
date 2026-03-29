"""
智能报告生成服务 v0.0.3
分步骤生成版本，结合固定模版、数据库和挂载知识库数据进行知识增强后的报告生成
"""

# ==================== 导入 ====================
import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional, List, AsyncIterator, Dict, Any

import aiohttp
import psutil
import torch
import uvicorn
import weaviate
import yaml
from aiohttp import ClientSession, ClientTimeout, FormData
from arango import ArangoClient
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModel
from weaviate.auth import AuthApiKey

from prompt import prompt1, prompt2


# ==================== 日志配置 ====================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ==================== 配置管理 ====================
class ConfigManager:
    """配置管理器（单例模式）"""
    _instance = None
    _config = None

    def __new__(cls, config_file: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_file)
        return cls._instance

    def _load_config(self, config_file: str):
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def reload(self, config_file: str = "config.yaml"):
        self._load_config(config_file)


class AppConfig:
    """应用配置类"""

    def __init__(self, config_file: str = "config.yaml"):
        self.config = ConfigManager(config_file)

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return self.config.get('api.server_address') + "/v1"

    @property
    def OLLAMA_API_URL(self) -> str:
        return self.config.get('api.server_address') + "/api/generate"

    @property
    def KEYWORD_EXTRACT_URL(self) -> str:
        return self.config.get('api.keyword_extract_url')

    @property
    def NER_EXTRACT_URL(self) -> str:
        return self.config.get('api.ner_extract_url')

    @property
    def DETECT_API_URL(self) -> str:
        return self.config.get('api.detect_api_url')

    @property
    def RETRIEVAL_API_URL(self) -> str:
        return self.config.get('api.retrieval_api_url')

    @property
    def BRIEF_MODEL(self) -> str:
        return self.config.get('models.brief_model')

    @property
    def NORMAL_MODEL(self) -> str:
        return self.config.get('models.normal_model')

    @property
    def RERANKER_MODEL_PATH(self) -> str:
        return self.config.get('models.reranker_model_path')

    @property
    def BGE_MODEL_PATH(self) -> str:
        return self.config.get('models.bge_model_path')

    @property
    def ARANGO_HOST(self) -> str:
        return self.config.get('arango.host')

    @property
    def ARANGO_USERNAME(self) -> str:
        return self.config.get('arango.username')

    @property
    def ARANGO_PASSWORD(self) -> str:
        return self.config.get('arango.password')

    @property
    def ARANGO_DATABASE(self) -> str:
        return self.config.get('arango.database')

    @property
    def ARANGO_SOURCE_COLLECTION(self) -> str:
        return self.config.get('arango.source_collection')

    @property
    def WEAVIATE_HOST(self) -> str:
        return self.config.get('weaviate.host')

    @property
    def WEAVIATE_HTTP_PORT(self) -> int:
        return self.config.get('weaviate.http_port')

    @property
    def WEAVIATE_GRPC_PORT(self) -> int:
        return self.config.get('weaviate.grpc_port')

    @property
    def WEAVIATE_API_KEY(self) -> str:
        return self.config.get('weaviate.api_key')

    def get_model_name(self, is_brief: bool) -> str:
        return self.BRIEF_MODEL if is_brief else self.NORMAL_MODEL

    def get_langchain_model(self, is_brief: bool = True) -> str:
        return self.get_model_name(is_brief)


# 全局配置实例
config = AppConfig()


# ==================== 服务管理 ====================
class ConfigUpdate(BaseModel):
    server_address: Optional[str] = None
    brief_model: Optional[str] = None
    normal_model: Optional[str] = None
    restart_service: Optional[bool] = True


class ServiceManager:
    def __init__(self, config_file="config.yaml", service_port=5003):
        self.config_file = config_file
        self.service_port = service_port
        self.service_command = None

    def find_process_by_port(self, port):
        """根据端口查找进程PID"""
        if not isinstance(port, int) or not (1 <= port <= 65535):
            logger.error(f"无效的端口号: {port}")
            return None
        try:
            connections = psutil.net_connections(kind='inet')
            for conn in connections:
                if conn.laddr.port == port and conn.pid:
                    logger.info(f"找到进程 PID: {conn.pid}, 端口: {port}")
                    return conn.pid
            logger.info(f"未找到占用端口 {port} 的进程")
            return None
        except psutil.AccessDenied:
            logger.warning("权限不足，尝试使用进程遍历方式")
            return self._find_process_by_port_fallback(port)
        except Exception as e:
            logger.error(f"查找进程时出错: {e}")
            return None

    def _find_process_by_port_fallback(self, port):
        """权限不足时的回退方法"""
        try:
            for proc in psutil.process_iter():
                try:
                    for conn in proc.connections(kind='inet'):
                        if conn.laddr.port == port:
                            return proc.pid
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            return None
        except Exception:
            return None

    def update_nested_dict(self, config_dict, new_values, path=""):
        """递归更新嵌套字典"""
        for key, value in new_values.items():
            if isinstance(value, dict) and key in config_dict and isinstance(config_dict[key], dict):
                self.update_nested_dict(config_dict[key], value, f"{path}.{key}" if path else key)
            else:
                config_dict[key] = value
                logger.info(f"更新配置: {path}.{key} = {value}" if path else f"更新配置: {key} = {value}")

    def update_config_yaml(self, new_values):
        """改进的配置文件更新方法"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as file:
                config_dict = yaml.safe_load(file)

            backup_file = f"{self.config_file}.backup"
            os.rename(self.config_file, backup_file)

            update_mappings = {
                'api.server_address': ['api', 'server_address'],
                'api.ollama_base_url': ['api', 'ollama_base_url'],
                'api.ollama_api_url': ['api', 'ollama_api_url'],
                'api.keyword_extract_url': ['api', 'keyword_extract_url'],
                'api.detect_api_url': ['api', 'detect_api_url'],
                'api.retrieval_api_url': ['api', 'retrieval_api_url'],
                'models.brief_model': ['models', 'brief_model'],
                'models.normal_model': ['models', 'normal_model'],
                'models.reranker_model_path': ['models', 'reranker_model_path'],
                'models.bge_model_path': ['models', 'bge_model_path'],
                'arango.host': ['arango', 'host'],
                'arango.username': ['arango', 'username'],
                'arango.password': ['arango', 'password'],
                'arango.database': ['arango', 'database'],
                'arango.source_collection': ['arango', 'source_collection'],
                'weaviate.host': ['weaviate', 'host'],
                'weaviate.http_port': ['weaviate', 'http_port'],
                'weaviate.grpc_port': ['weaviate', 'grpc_port'],
                'weaviate.api_key': ['weaviate', 'api_key'],
            }

            for key, value in new_values.items():
                if key in update_mappings:
                    path = update_mappings[key]
                    config_dict[path[0]][path[1]] = value
                    logger.info(f"更新配置: {key} = {value}")

            with open(self.config_file, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, allow_unicode=True)

            logger.info("配置文件更新成功")
            return True

        except Exception as e:
            logger.error(f"更新配置文件失败: {e}")
            if os.path.exists(f"{self.config_file}.backup"):
                os.rename(f"{self.config_file}.backup", self.config_file)
            return False

    def update_config_with_regex(self, replacements):
        """使用正则表达式更新配置文件"""
        try:
            backup_file = f"{self.config_file}.backup"
            with open(self.config_file, 'r', encoding='utf-8') as file:
                content = file.read()
            with open(backup_file, 'w', encoding='utf-8') as file:
                file.write(content)
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            with open(self.config_file, 'w', encoding='utf-8') as file:
                file.write(content)
            logger.info("配置文件更新成功（正则方式）")
            return True
        except Exception as e:
            logger.error(f"更新配置文件失败: {e}")
            if os.path.exists(backup_file):
                os.rename(backup_file, self.config_file)
            return False

    def start_service(self, command=None):
        """启动服务"""
        if not command:
            command = self.service_command
        if not command:
            logger.error("未设置服务启动命令")
            return False
        try:
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            logger.info(f"服务已启动，PID: {process.pid}")
            return True
        except Exception as e:
            logger.error(f"启动服务失败: {e}")
            return False

    def restart_service(self, start_command=None):
        """重启服务"""
        try:
            port = "5003"
            default_start_command = "/root/anaconda3/bin/conda run -n rag --no-capture-output python /TRS/wyc/deepseekr1/sevices/ai_report_v0_5.py "
            command_to_use = start_command if start_command else default_start_command
            full_command = ["./restart_service.sh", port, command_to_use]
            logging.info(f"执行重启服务命令: {' '.join(full_command)}")
            result = subprocess.run(full_command, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logging.info("服务重启成功")
                logging.info(f"输出: {result.stdout}")
                return True
            else:
                logging.error(f"服务重启失败，返回码: {result.returncode}")
                logging.error(f"错误输出: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logging.error("重启服务命令执行超时")
            return False
        except FileNotFoundError:
            logging.error("找不到 restart_service.sh 脚本文件")
            return False
        except Exception as e:
            logging.error(f"重启服务时发生错误: {str(e)}")
            return False


service_manager = ServiceManager()


# ==================== 工具类 ====================
class BGESentenceEncoder:
    """BGE句子编码器"""

    def __init__(self, model_path: str = None, device: Optional[str] = None):
        if model_path is None:
            model_path = config.BGE_MODEL_PATH
        logger.info(f"初始化 BGESentenceEncoder，模型路径: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"BGESentenceEncoder 初始化完成，设备: {self.device}")

    def encode(self, sentences: List[str], instruction: Optional[str] = None,
               normalize: bool = True, **tokenizer_kwargs) -> torch.Tensor:
        if instruction:
            sentences = [instruction + s for s in sentences]
        default_args = {'padding': True, 'truncation': True, 'return_tensors': 'pt'}
        tokenizer_args = {**default_args, **tokenizer_kwargs}
        try:
            encoded_input = self.tokenizer(sentences, **tokenizer_args).to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]
            if normalize:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu()
        except Exception as e:
            logger.error(f"BGESentenceEncoder.encode 编码失败: {e}", exc_info=True)
            raise


# ==================== 全局初始化 ====================
logger.info("应用启动：初始化FastAPI实例")
app = FastAPI(title="智能报告v0.0.3",
              summary="该版本是分步骤生成的版本,并结合固定模版,数据库和挂载知识库数据进行知识增强后的报告生成",
              version="0.0.3")

# CORS 配置
origins = ["http://localhost", "http://localhost:8080", "http://127.0.0.1",
           "http://127.0.0.1:8080", "http://172.100.0.137:4306", "*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# 加载模型和数据库连接
try:
    reranker = CrossEncoder(config.RERANKER_MODEL_PATH,
                            device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Reranker模型加载成功，设备：{'cuda' if torch.cuda.is_available() else 'cpu'}")
except Exception as e:
    logger.error(f"Reranker模型加载失败: {e}", exc_info=True)

try:
    encode = BGESentenceEncoder(device="cuda:1")
    logger.info("BGESentenceEncoder 实例创建成功")

    model = ChatOpenAI(model=config.get_langchain_model(is_brief=True),
                       api_key="ollama", base_url=config.OLLAMA_BASE_URL)
    logger.info("Langchain ChatOpenAI 模型实例创建成功")

    client_weaviate = weaviate.connect_to_custom(
        http_host=config.WEAVIATE_HOST, http_port=config.WEAVIATE_HTTP_PORT, http_secure=False,
        grpc_host=config.WEAVIATE_HOST, grpc_port=config.WEAVIATE_GRPC_PORT, grpc_secure=False,
        auth_credentials=AuthApiKey(config.WEAVIATE_API_KEY)
    )
    logger.info(f"Weaviate 客户端连接成功: {config.WEAVIATE_HOST}:{config.WEAVIATE_HTTP_PORT}")

    db = connect_to_arangodb(config.ARANGO_HOST, config.ARANGO_USERNAME,
                             config.ARANGO_PASSWORD, config.ARANGO_DATABASE)
    entity_disc_list = fetch_entity_names(db, config.ARANGO_SOURCE_COLLECTION)
    logger.info(f"从 ArangoDB 获取到 {len(entity_disc_list)} 条实体描述")

    reranker = CrossEncoder(config.RERANKER_MODEL_PATH,
                            device='cpu' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Reranker模型重新加载成功")
except Exception as e:
    logger.critical(f"全局初始化失败: {e}", exc_info=True)

logger.info(f"Ollama API URL: {config.OLLAMA_API_URL}")

# 图片保存目录
RESULT_IMG_DIR = Path("data_img")
RESULT_IMG_DIR.mkdir(parents=True, exist_ok=True)
timeout = ClientTimeout(total=30)


# ==================== 辅助函数 ====================
def normalize_and_split(text) -> List[str]:
    """标准化文本中的中文标点并分割"""
    replacements = {"，": ",", "；": ";", "、": ",", "。": ".", " ": " "}
    normalized_text = text
    for cn, en in replacements.items():
        normalized_text = normalized_text.replace(cn, en)
    pattern = r"[\s,;]+"
    return [item for item in re.split(pattern, normalized_text) if item]


def build_stream_chunk(content: str, model_name: str, stage: str = None,
                       finish_reason: str = None, is_error: bool = False) -> dict:
    """构建流式响应块（统一函数）"""
    chunk = {
        "id": "chatcmpl",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "delta": {"content": content if not is_error else json.dumps(content, ensure_ascii=False)},
            "index": 0,
            "finish_reason": finish_reason
        }],
        "service_tier": None,
        "system_fingerprint": None
    }
    if stage:
        chunk["stage"] = stage
    if is_error:
        chunk["error"] = True
    return chunk


def send_stream_end_signals(model_name: str) -> str:
    """生成流式响应结束信号"""
    final_chunk = build_stream_chunk("", model_name, finish_reason="stop")
    return f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"


def connect_to_arangodb(host, username, password, database):
    """连接 ArangoDB"""
    logger.info(f"连接 ArangoDB，主机: {host}, 数据库: {database}")
    try:
        client = ArangoClient(hosts=host)
        db = client.db(database, username=username, password=password)
        logger.info("ArangoDB 连接成功")
        return db
    except Exception as e:
        logger.critical(f"连接 ArangoDB 失败: {e}", exc_info=True)
        raise


def fetch_entity_names(db, collection_name):
    """从 ArangoDB 集合获取实体名称"""
    logger.info(f"从 ArangoDB 集合 '{collection_name}' 获取实体名称")
    try:
        entity_collection = db.collection(collection_name)
        entity_cursor = entity_collection.all()
        entity_name = [doc['desc'] for doc in entity_cursor
                       if 'name' in doc and isinstance(doc['name'], str) and doc['name'].strip()]
        logger.info(f"成功从 ArangoDB 获取 {len(entity_name)} 个实体描述")
        return entity_name
    except Exception as e:
        logger.error(f"从 ArangoDB 获取实体名称失败: {e}", exc_info=True)
        return []


def query_vector_collection(client: weaviate.WeaviateClient, collection_name: str, vector, k: int) -> list:
    """查询 Weaviate 向量集合"""
    logger.debug(f"调用 query_vector_collection，集合: {collection_name}, k: {k}")
    try:
        collection = client.collections.get(collection_name)
        vector_data = vector.tolist() if isinstance(vector, torch.Tensor) else vector
        response = collection.query.near_vector(
            near_vector=vector_data, limit=k,
            return_properties=["id_", "name", "desc"],
            return_metadata=["distance", "certainty"]
        )
        results = []
        for obj in response.objects:
            results.append({
                "id_": obj.properties.get("id_"),
                "name": obj.properties.get("name"),
                "desc": obj.properties.get("desc"),
                "certainty": obj.metadata.certainty
            })
        logger.info(f"Weaviate 查询成功，返回 {len(results)} 条结果")
        return results
    except Exception as e:
        logger.error(f"Weaviate 查询异常: {e}", exc_info=True)
        return []


def extract_entities_jieba(title):
    """使用 Jieba 提取实体"""
    import jieba
    import jieba.posseg as pseg
    jieba.add_word("网络空间作战")
    jieba.add_word("高级点名系统")
    words = pseg.cut(title)
    entities = {'organization': [], 'location': [], 'concept': [], 'system': []}
    for word, flag in words:
        if flag == 'nt':
            entities['organization'].append(word)
        elif flag == 'ns':
            entities['location'].append(word)
        elif word in ['网络空间作战', '高级点名系统']:
            entities['concept'].append(word)
    return entities


def create_overview_prompt(title: str, is_brief: bool) -> str:
    """创建概述生成提示词"""
    return f"请将以下大纲进行扩写为一句话50字,直接生成总结,不要有标题和多余的回复和解释:\n{title}  /no_think"


def create_paragraph_prompt(title: str, overview: str) -> str:
    """创建正文生成提示词"""
    return f"""
请根据以下标题和概述生成一篇完整的文章。要求：
1. 使用 Markdown 格式
2. 文章结构清晰，段落分明
3. 内容要具体且详实
4. 确保内容与标题和概述保持一致
5. 可以分段,不要有标题,如果需要标题请从正文5级以下标题开始即标题至少#####然后更多
6. 行文流畅自然
7. 适当添加示例或案例来支持论点
8. 两三段就行不要生成过多段落
9. 按照总分的结构整体结构不要有总结或者综上所述等字样表述

标题：{title}
概述：{overview}

请生成一篇完整的 Markdown 格式文章，包含：
- 不需要包含标题
- 详细的内容展开
- 适当的段落划分
- 重点内容强调
- 请不要生成"```"这种格式的代码块
- 不要有总结字样只生成内容无需总结
- 必要的列表或表格（如果适用） /no_think
"""


# ==================== API调用函数 ====================
async def call_ollama_stream(prompt: str, model_name: str = None, t: Optional[float] = None) -> AsyncIterator[str]:
    """流式调用 Ollama API"""
    if model_name is None:
        model_name = config.NORMAL_MODEL
    if t is None:
        t = 0.7
    logger.debug(f"调用 call_ollama_stream - 模型: {model_name}, temperature: {t}")
    try:
        llm = ChatOpenAI(model=model_name, temperature=t, max_tokens=5096,
                         base_url=config.OLLAMA_BASE_URL, api_key="ollama", streaming=True)
        logger.info(f"开始流式调用 Ollama API: {config.OLLAMA_BASE_URL}")
        async for chunk in llm.astream(prompt):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
        logger.info("Ollama API 流式响应完成")
    except Exception as e:
        error_message = f"Ollama 请求异常: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise RuntimeError(error_message) from e


async def call_ollama_non_stream(prompt: str, model_name: str = None, t: Optional[float] = None) -> str:
    """非流式调用 Ollama API"""
    if model_name is None:
        model_name = config.NORMAL_MODEL
    if t is None:
        t = 0.7
    logger.debug(f"调用 call_ollama_non_stream - 模型: {model_name}, temperature: {t}")
    try:
        llm = ChatOpenAI(model=model_name, temperature=t, max_tokens=2048,
                         base_url=config.OLLAMA_BASE_URL, api_key="ollama", streaming=False)
        logger.info(f"开始调用 Ollama API: {config.OLLAMA_BASE_URL}")
        response = await llm.ainvoke(prompt)
        logger.info("Ollama API 响应完成")
        return response.content
    except Exception as e:
        error_message = f"Ollama 请求异常: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise RuntimeError(error_message) from e


async def key_word_extract(url, data):
    """关键词提取"""
    logger.debug(f"调用 key_word_extract，URL: {url}, 数据: {data}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data, timeout=60) as response:
                if response.status != 200:
                    error_message = f"关键词提取服务请求失败: 状态码 {response.status}"
                    logger.error(error_message)
                    return error_message
                response_data = await response.json()
                if response_data and isinstance(response_data.get("result"), list) and len(response_data["result"]) > 1:
                    key_words = response_data["result"][1].get("主题词")
                    logger.info(f"关键词提取成功，结果: {key_words}")
                    return key_words
                else:
                    logger.warning(f"关键词提取服务返回数据结构异常: {response_data}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"关键词提取客户端错误: {e}", exc_info=True)
            return f"关键词提取客户端错误: {str(e)}"
        except asyncio.TimeoutError:
            logger.error("关键词提取请求超时")
            return "关键词提取请求超时"
        except Exception as e:
            logger.error(f"关键词提取发生异常: {str(e)}", exc_info=True)
            return f"关键词提取发生异常: {str(e)}"


async def ner_extract(url, data):
    """NER实体提取"""
    logger.debug(f"调用 ner_extract，URL: {url}, 数据: {data}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data, timeout=60) as response:
                if response.status != 200:
                    error_message = f"NER提取服务请求失败: 状态码 {response.status}"
                    logger.error(error_message)
                    return error_message
                response_data = await response.json()
                return response_data['result'][0]
        except aiohttp.ClientError as e:
            logger.error(f"NER提取客户端错误: {e}", exc_info=True)
            return f"NER提取客户端错误: {str(e)}"
        except asyncio.TimeoutError:
            logger.error("NER提取请求超时")
            return "NER提取请求超时"
        except Exception as e:
            logger.error(f"NER提取发生异常: {str(e)}", exc_info=True)
            return f"NER提取发生异常: {str(e)}"


async def search_related_knowledge(key_words):
    """搜索相关知识"""
    logger.info(f"开始搜索相关知识，关键词: {key_words}")
    if not key_words:
        logger.warning("关键词为空，无法搜索相关知识。")
        return []
    if not isinstance(key_words, str):
        logger.warning(f"关键词类型不是字符串: {type(key_words)}")
        return []

    topic_words_list = key_words.split(",")
    related_knowledge_list = []
    MAX_KNOWLEDGE_COUNT = 30
    KNOWLEDGE_PER_KEYWORD = 5

    for word_idx, i in enumerate(topic_words_list):
        i = i.strip()
        if not i:
            continue
        if len(related_knowledge_list) >= MAX_KNOWLEDGE_COUNT:
            logger.info(f"已达到相关知识召回上限 {MAX_KNOWLEDGE_COUNT}，停止搜索。")
            break
        try:
            encoded_vector = encode.encode([i])
            if encoded_vector.ndim == 2:
                i_v = encoded_vector.tolist()[0]
            else:
                i_v = encoded_vector.tolist()
            logger.debug(f"为关键词 '{i}' (索引 {word_idx}) 生成嵌入向量")
            related_knowledge = query_vector_collection(client_weaviate, 'kg_wenda_large', i_v, k=KNOWLEDGE_PER_KEYWORD)
            logger.debug(f"关键词 '{i}' 召回 {len(related_knowledge)} 条知识")
            for k_item in related_knowledge:
                if len(related_knowledge_list) >= MAX_KNOWLEDGE_COUNT:
                    break
                desc_len = len(k_item.get('desc', ''))
                if desc_len > 1000 or desc_len < 50:
                    logger.debug(f"跳过知识 '{k_item.get('name')}'，描述长度 ({desc_len}) 不符合要求。")
                    continue
                else:
                    related_knowledge_list.append(k_item['name'])
                    related_knowledge_list.append(k_item['desc'])
        except Exception as e:
            logger.error(f"搜索关键词 '{i}' 相关知识时发生错误: {e}", exc_info=True)
            continue

    logger.info(f"相关知识搜索完成，共召回 {len(related_knowledge_list) // 2} 条知识。")
    return related_knowledge_list


async def call_detection_service(client: ClientSession, img_bytes: bytes, filename: str, content_type: str):
    """调用图像检测服务"""
    form_data = FormData()
    form_data.add_field('image', img_bytes, filename=filename, content_type=content_type)
    async with client.post(config.DETECT_API_URL, data=form_data, timeout=timeout) as response:
        if response.status != 200:
            raise HTTPException(status_code=response.status, detail="图像识别服务调用失败")
        return await response.json()


async def call_retrieval_service(client: ClientSession, img_bytes: bytes, filename: str, content_type: str):
    """调用图像检索服务"""
    form_data = FormData()
    form_data.add_field('image_file', img_bytes, filename=filename, content_type=content_type)
    async with client.post(config.RETRIEVAL_API_URL, data=form_data, timeout=timeout) as response:
        if response.status != 200:
            raise HTTPException(status_code=response.status, detail="图像检索服务调用失败")
        return await response.json()


async def detect_task(file: UploadFile):
    """图像检测任务"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件")
    img_bytes = await file.read()
    async with ClientSession() as client:
        detection_result = await call_detection_service(client, img_bytes, file.filename, file.content_type)
        base64_img = detection_result.get("图片")
        if not base64_img:
            raise HTTPException(status_code=500, detail="识别结果中未包含图片数据")
        retrieval_result = await call_retrieval_service(client, img_bytes, file.filename, file.content_type)
        related_image = retrieval_result.get("results")[0].get("related_image")
        related_image_id = retrieval_result.get("results")[0].get("related_image_id")
        return {
            "detection_result": detection_result,
            "retrieval_result": related_image,
            "related_image_id": str(related_image_id).split("-")[1]
        }


# ==================== API路由 ====================
@app.get("/status")
async def get_status():
    """获取服务状态"""
    pid = service_manager.find_process_by_port(service_manager.service_port)
    with open('config.yaml', 'r') as file:
        config_yaml = yaml.safe_load(file)
    server_url = config_yaml.get('api', {}).get('server_address', '')
    brief_model = config_yaml.get('models', {}).get('brief_model', '')
    normal_model = config_yaml.get('models', {}).get('normal_model', '')
    return {
        "service_port": service_manager.service_port,
        "is_running": pid is not None,
        "pid": pid,
        "server_url": server_url,
        "brief_model": brief_model,
        "normal_model": normal_model
    }


@app.get("/config")
async def get_config():
    """获取当前配置"""
    try:
        with open(service_manager.config_file, 'r', encoding='utf-8') as file:
            config_yaml = yaml.safe_load(file)
        return {"config": config_yaml}
    except Exception as e:
        logger.error(f"读取配置文件时出错: {e}")
        raise HTTPException(status_code=500, detail=f"读取配置文件失败: {str(e)}")


@app.post("/update-config")
async def update_config(update_data: ConfigUpdate):
    """更新配置文件"""
    try:
        new_values = {}
        if update_data.server_address:
            new_values['api.server_address'] = update_data.server_address
        if update_data.brief_model:
            new_values['models.brief_model'] = update_data.brief_model
        if update_data.normal_model:
            new_values['models.normal_model'] = update_data.normal_model
        if not new_values:
            raise HTTPException(status_code=400, detail="没有提供要更新的配置项")

        success = service_manager.update_config_yaml(new_values)
        if not success:
            raise HTTPException(status_code=500, detail="更新配置文件失败")

        result = {"message": "配置更新成功", "updated_config": new_values}
        if update_data.restart_service:
            restart_success = service_manager.restart_service()
            result["restart_success"] = restart_success
            if restart_success:
                result["message"] += "，服务重启成功"
            else:
                result["message"] += "，但服务重启失败"
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置时出错: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.post("/update-config-regex")
async def update_config_regex(replacements: list):
    """使用正则表达式更新配置文件"""
    try:
        regex_rules = [(item["pattern"], item["replacement"]) for item in replacements]
        success = service_manager.update_config_with_regex(regex_rules)
        if not success:
            raise HTTPException(status_code=500, detail="更新配置文件失败")
        return {"message": "配置文件更新成功（正则方式）"}
    except Exception as e:
        logger.error(f"正则更新配置时出错: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.post("/restart-service")
async def restart_service(start_command: Optional[str] = None):
    """重启服务"""
    try:
        success = service_manager.restart_service(start_command)
        if success:
            return {"message": "服务重启成功"}
        else:
            raise HTTPException(status_code=500, detail="服务重启失败")
    except Exception as e:
        logger.error(f"重启服务时出错: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "version": "0.0.3",
        "config": {
            "brief_model": config.BRIEF_MODEL,
            "normal_model": config.NORMAL_MODEL,
            "ollama_base_url": config.OLLAMA_BASE_URL
        }
    }


@app.post("/chat/completions")
async def chat_completions(request: Request):
    """聊天补全接口"""
    logger.info("接收到 /chat/completions 请求")
    try:
        body = await request.json()
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not all(isinstance(msg, dict) and "content" in msg for msg in messages):
            logger.warning(f"无效的 messages 格式: {messages}")
            raise HTTPException(status_code=400, detail="Invalid messages format")

        content = messages[-1]["content"]
        logger.info(f"用户输入内容: {content}")

        if '研究报告' in content:
            prompt = prompt1 + content
        elif '作战' in content:
            prompt = prompt2 + content
        else:
            prompt = content

        logger.info(f"构建的 Prompt: {prompt[:200]}...")

        async def stream_response():
            full_response = ""
            try:
                logger.info("开始流式生成响应")
                async for new_text in call_ollama_stream(prompt):
                    if isinstance(new_text, dict) and "error" in new_text:
                        logger.error(f"Ollama 流式生成错误: {new_text.get('error')}")
                        yield f"data: {json.dumps(new_text, ensure_ascii=False)}\n\n"
                        break
                    full_response += new_text
                    chunk = {
                        "id": "chatcmpl", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": "deepseek-chat",
                        "choices": [{"delta": {"content": new_text}, "index": 0, "finish_reason": None}],
                        "service_tier": None, "system_fingerprint": None
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                logger.info("流式生成完成，发送最终信号")
                final_chunk = {
                    "id": "chatcmpl", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": "deepseek-chat",
                    "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                    "service_tier": None, "system_fingerprint": None
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'full_response': full_response}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"流式响应生成过程中发生错误: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            finally:
                yield "data: [DONE]\n\n"
                logger.info("流式响应结束")

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/chat/completions 请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/generateOutline")
async def generate_outline(request: Request):
    """生成大纲"""
    logger.info("接收到 /generateOutline 请求")
    try:
        body = await request.json()
        content = body.get("content")
        is_brief = body.get("is_brief")
        logger.info(f"生成大纲内容: {content}")

        if not content:
            logger.warning("请求体中缺少 'content' 字段")
            raise HTTPException(status_code=400, detail="Missing 'content' in request body")

        prompt = f"""
根据给定的内容，提取出相关的关键词和主要信息，并生成一个具有逻辑结构的政府公文式标题大纲。
输入内容：{content}
请参照以下通用模板格式生成标准的政府公文大纲：
## 主标题
### 一、----
#### 1.1 -----
##### 1.1.1 --------
...
请根据输入内容的具体特点，灵活调整标题表述，确保大纲符合政府公文规范。以标准Markdown格式输出，无需额外说明。 /no_think
"""
        logger.debug(f"生成大纲的Prompt: {prompt}")

        if body.get("direct_reply"):
            full_response = await generate_full_outline(prompt, is_brief)
            return {"full_response": full_response}
        else:
            return StreamingResponse(stream_response_outline(prompt, is_brief), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/generateOutline 请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


async def generate_full_outline(prompt: str, is_brief: bool) -> str:
    """生成完整大纲"""
    full_response = ""
    try:
        logger.info("开始生成完整大纲")
        model_name = config.get_model_name(is_brief)
        async for new_text in call_ollama_stream(prompt, model_name, t=1.0):
            if isinstance(new_text, dict) and "error" in new_text:
                logger.error(f"Ollama 生成完整大纲错误: {new_text.get('error')}")
                raise Exception(new_text["error"])
            full_response += new_text
        logger.info(f"完整大纲生成完成，长度: {len(full_response)}")
        return full_response
    except Exception as e:
        logger.error(f"生成完整大纲过程中发生错误: {e}", exc_info=True)
        raise e


async def stream_response_outline(prompt: str, is_brief: bool):
    """流式生成大纲响应"""
    try:
        logger.info("开始流式生成大纲")
        model_name = config.get_model_name(is_brief)
        async for text_chunk in call_ollama_stream(prompt, model_name, t=0.1):
            chunk = build_stream_chunk(text_chunk, model_name)
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield send_stream_end_signals(model_name)
        logger.info("大纲流式生成完成")
    except Exception as e:
        logger.error(f"流式生成大纲失败: {e}", exc_info=True)
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"


@app.post("/outlineOverview")
async def outline_overview(request: Request):
    """大纲扩写概述"""
    logger.info("接收到 /outlineOverview 请求")
    try:
        body = await request.json()
        outline = body.get("generate_outline")
        is_brief = body.get("is_brief")
        direct_reply = body.get("direct_reply", False)

        logger.info(f"大纲概述请求，大纲内容长度: {len(outline) if outline else 0}")

        if not outline:
            logger.warning("请求体中缺少 'generate_outline' 字段")
            raise HTTPException(status_code=400, detail="Missing 'generate_outline' in request")

        # 知识检索（仅在非简洁模式下）
        related_knowledge_new = ""
        if not is_brief:
            keyword_payload = {"text": outline}
            key_words = await key_word_extract(config.KEYWORD_EXTRACT_URL, keyword_payload)
            if key_words:
                logger.info(f"提取到的关键词: {key_words}")
                related_knowledge_list = await search_related_knowledge(key_words)
                logger.info(f"召回 {len(related_knowledge_list) // 2} 条相关知识")
                key_words_ner = await ner_extract(config.NER_EXTRACT_URL, {"text": str(outline).split("\n")[0]})
                encoded_vector = encode.encode([key_words_ner])
                i_v = encoded_vector[0].tolist()
                related_knowledge = query_vector_collection(client_weaviate, 'kg_wenda_large', i_v, k=1)
                if len(related_knowledge) > 0 and related_knowledge[0].get('certainty') > 0.8:
                    related_knowledge_new = related_knowledge[0].get('desc')
                else:
                    related_knowledge_new = str(related_knowledge_list)

        # 统一的prompt模板
        knowledge_section = f"相关知识关联如下:\n```{related_knowledge_new}```\n\n" if related_knowledge_new else ""
        prompt = f"""
请将以下大纲进行扩写,不要有多余的回复和解释:
1. 保持原有标题层级结构和编号方式不变
2. 在每个标题下方添加 2-3 句概括性描述
3. 描述要简明扼要,突出重点
4. 确保描述与标题主题相关
5. 按原格式返回完整大纲

格式示例:
## 美军作战点名系统的设计与研究
### 一、研究背景与意义
#### 1.1 研究背景
（此处可添加发展历程、当前存在的问题或技术挑战）
#### 1.2 研究意义
（说明本研究对提升作战效率、指挥协同或军事信息化的价值）
...

{knowledge_section}请以标准Markdown格式回复

我要扩充的大纲内容如下,不要有多余的回复:
{outline}  /no_think
"""

        logger.debug(f"生成大纲概述的Prompt长度: {len(prompt)}")

        if direct_reply:
            full_response = await generate_full_overview(prompt, is_brief)
            return {"full_response": full_response}
        else:
            return StreamingResponse(stream_overview_response(prompt, is_brief), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/outlineOverview 请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


async def generate_full_overview(prompt: str, is_brief: bool) -> str:
    """生成完整扩写大纲"""
    full_response = ""
    try:
        logger.info("开始生成完整大纲扩写")
        model_name = config.get_model_name(is_brief)
        async for new_text in call_ollama_stream(prompt, model_name):
            if isinstance(new_text, dict) and "error" in new_text:
                raise Exception(new_text["error"])
            full_response += new_text
        logger.info(f"完整扩写大纲生成完成，长度: {len(full_response)}")
        return full_response
    except Exception as e:
        logger.error(f"生成完整扩写大纲过程中发生错误: {e}", exc_info=True)
        raise e


async def stream_overview_response(prompt: str, is_brief: bool):
    """流式生成大纲扩写响应"""
    full_response = ""
    try:
        logger.info("开始流式生成大纲扩写")
        model_name = config.get_model_name(is_brief)
        async for new_text in call_ollama_stream(prompt, model_name):
            if isinstance(new_text, dict) and "error" in new_text:
                error_chunk = build_stream_chunk(new_text, model_name, is_error=True)
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                break
            full_response += new_text
            chunk = build_stream_chunk(new_text, model_name)
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = build_stream_chunk("", model_name, finish_reason="stop")
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'full_response': full_response}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"流式生成扩写大纲过程中发生错误: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


@app.post("/paragraph_generator")
async def paragraph_generator(request: Request):
    """段落生成"""
    logger.info("接收到 /paragraph_generator 请求")
    try:
        body = await request.json()
        title = body.get("title")
        paragraph = body.get("paragraph")
        is_brief = body.get("is_brief")
        direct_reply = body.get("direct_reply", False)
        logger.info(f"段落生成请求，标题: '{title}'")

        if not title or not paragraph:
            logger.warning(f"缺少必要字段: title={title}, paragraph={paragraph}")
            raise HTTPException(status_code=400, detail="Missing required fields: 'title' or 'paragraph'")

        prompt = create_paragraph_prompt(title, paragraph)
        logger.debug(f"段落生成Prompt长度: {len(prompt)}")

        if direct_reply:
            full_response = await generate_full_paragraph(prompt, is_brief)
            return {"full_response": full_response}
        else:
            return StreamingResponse(stream_paragraph_response(prompt, is_brief), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/paragraph_generator 请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


async def generate_full_paragraph(prompt: str, is_brief: bool) -> str:
    """生成完整段落内容"""
    full_response = ""
    model_name = config.get_model_name(is_brief)
    try:
        logger.info("开始生成完整段落内容")
        async for new_text in call_ollama_stream(prompt, model_name=model_name, t=0.7):
            if isinstance(new_text, dict) and "error" in new_text:
                raise Exception(new_text["error"])
            full_response += new_text
        logger.info(f"完整段落内容生成完成，长度: {len(full_response)}")
        return full_response
    except Exception as e:
        logger.error(f"生成完整段落内容过程中发生错误: {e}", exc_info=True)
        raise e


async def stream_paragraph_response(prompt: str, is_brief: bool):
    """流式生成段落内容响应"""
    full_response = ""
    model_name = config.get_model_name(is_brief)
    try:
        logger.info("开始流式生成段落内容")
        async for new_text in call_ollama_stream(prompt, model_name, t=1.5):
            if isinstance(new_text, dict) and "error" in new_text:
                error_chunk = build_stream_chunk(new_text, model_name, is_error=True)
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                break
            full_response += new_text
            chunk = build_stream_chunk(new_text, model_name)
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = build_stream_chunk("", model_name, finish_reason="stop")
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'full_response': full_response}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"流式生成段落内容过程中发生错误: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


@app.post("/topic_and_paragraph")
async def topic_summary(request: Request):
    """主题和段落生成"""
    logger.info("接收到 /topic_and_paragraph 请求")
    try:
        data = await request.json()
        if data is None:
            logger.warning("请求体为空")
            raise HTTPException(status_code=400, detail="Invalid messages format")

        title = data.get("title")
        chapters = data.get("chapters")
        table_disc = data.get("table")

        if title is None:
            raise HTTPException(status_code=400, detail="Invalid topic format: 'title' is missing")
        if chapters is None:
            raise HTTPException(status_code=400, detail="Invalid chapters format: 'chapters' is missing")

        logger.info(f"主题: '{title}', 章节数量: {len(chapters)}")

        # 提取关键词并搜索相关知识
        keyword_payload = {"text": title}
        key_words = await key_word_extract(config.KEYWORD_EXTRACT_URL, keyword_payload)
        related_knowledge_list = await search_related_knowledge(key_words) if key_words else []
        logger.info(f"为标题 '{title}' 召回 {len(related_knowledge_list) // 2} 条相关知识。")

        # 生成主题描述
        topic_prompt = f"作为专业的报告撰写专家,请针对主题'{title}'生成一段精炼的描述。参考提供的资料:{str(related_knowledge_list)},但不局限于此。请直接输出200字左右的主题描述段落,突出重点要素,语言简洁专业。无需其他解释说明。 /no_think"
        try:
            resp_title = await model.ainvoke(topic_prompt)
            title_content = {"content": resp_title.content.split("\n\n")[-1].strip()}
            logger.info(f"主题 '{title}' 描述生成成功。")
        except Exception as e:
            logger.error(f"主题 '{title}' 描述生成失败: {e}", exc_info=True)
            title_content = {"content": "未能生成主题描述。"}

        chapters_list = []
        table_list = []

        # 生成各章节内容
        for chapter in chapters:
            chapter_name = chapter.get("chapterName")
            chapter_order = chapter.get("order")
            logger.info(f"处理章节: '{chapter_name}' (序号: {chapter_order})")

            if chapter_name is None:
                logger.warning(f"章节 (序号: {chapter_order}) 缺少 'chapterName' 字段，跳过。")
                continue

            chapter_keyword_payload = {"text": chapter_name}
            chapter_key_words = await key_word_extract(config.KEYWORD_EXTRACT_URL, chapter_keyword_payload)
            chapter_related_knowledge = await search_related_knowledge(chapter_key_words) if chapter_key_words else []
            logger.info(f"为章节 '{chapter_name}' 召回 {len(chapter_related_knowledge) // 2} 条相关知识。")

            chapter_prompt = f"作为专业报告撰写专家,请针对主题'{title}'和段落标题'{chapter_name}'(第{chapter_order}段),生成3段内容,采用总分总结构。可参考以下资料:{str(chapter_related_knowledge)},但不局限于此。请直接输出报告正文内容,段落连贯,每段200字左右。无需其他说明。 /no_think"
            try:
                resp_chapter = await model.ainvoke(chapter_prompt)
                chapter_content = resp_chapter.content.split("\n\n")[-1].strip()
                logger.info(f"章节 '{chapter_name}' 内容生成成功。")
            except Exception as e:
                logger.error(f"章节 '{chapter_name}' 内容生成失败: {e}", exc_info=True)
                chapter_content = "未能生成章节内容。"

            chapters_list.append({"chapterName": chapter_name, "order": chapter_order, "content": chapter_content})

        # 生成表格
        if table_disc:
            logger.info("检测到表格生成请求。")
            for t_disc_item in table_disc:
                table_columns = t_disc_item.get("columns")
                table_rows_str = t_disc_item.get("rows")

                if not table_columns or not table_rows_str:
                    logger.warning(f"表格定义缺少列或行信息，跳过。")
                    continue

                names = normalize_and_split(table_rows_str)
                name_disc_arr = []

                for name in names:
                    if not name.strip():
                        continue
                    try:
                        encoded_vector = encode.encode([name])
                        i_v = encoded_vector.tolist()[0] if encoded_vector.ndim == 2 else encoded_vector.tolist()
                        res = query_vector_collection(client_weaviate, 'kg_wenda_large', i_v, k=1)
                        if res and len(res) == 1 and res[0].get('certainty', 0) > 0.7:
                            name_disc_arr.append(res[0].get("desc"))
                        else:
                            name_disc_arr.append(None)
                    except Exception as e:
                        logger.error(f"查询行名称 '{name}' 描述时出错: {e}", exc_info=True)
                        name_disc_arr.append(None)

                valid_name_disc_arr = [desc for desc in name_disc_arr if desc is not None and desc.strip()]
                if not valid_name_disc_arr:
                    logger.warning(f"未能为任何表格行名称找到有效描述，跳过表格生成。")
                    table_list.append({"table": f"未能为表格生成内容。原始行名称: {table_rows_str}"})
                    continue

                names_prompt_str = ";".join(valid_name_disc_arr)
                table_prompt = f"下面为我的相关人物描述;\n{names_prompt_str}\n请根据我的相关人物描述去生成包含行名为人物名称列名为以下内容的表格;\n{table_columns}\n请以markdown的形式生成表格,不要有多余的解释和回复。 /no_think"
                try:
                    tb_resp = await model.ainvoke(table_prompt)
                    table_res = tb_resp.content.split("\n\n")[-1].strip()
                    table_list.append({"table": table_res})
                    logger.info("表格生成成功。")
                except Exception as e:
                    logger.error(f"表格生成失败: {e}", exc_info=True)
                    table_list.append({"table": "未能生成表格。")

        final_response = {
            "title": title,
            "content": title_content.get("content"),
            "chapters": chapters_list,
            "table": table_list
        }
        logger.info("/topic_and_paragraph 请求处理完成。")
        return final_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/topic_and_paragraph 请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/detect")
async def detect_image(file: UploadFile):
    """图像检测接口"""
    try:
        result = await detect_task(file)
        return JSONResponse(content=result)
    except HTTPException as he:
        logger.error(f'HTTPException: {he.detail}')
        raise he
    except Exception as e:
        logger.exception('未知错误')
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.post("/generate_complete_content")
async def generate_complete_content(request: Request):
    """整合接口：根据标题生成概述，然后生成正文内容"""
    logger.info("接收到 /generate_complete_content 请求")
    try:
        body = await request.json()
        title = body.get("title")
        is_brief = body.get("is_brief", False)
        direct_reply = body.get("direct_reply", False)

        logger.info(f"完整内容生成请求，标题: '{title}', 简洁模式: {is_brief}")

        if not title:
            logger.warning("请求体中缺少 'title' 字段")
            raise HTTPException(status_code=400, detail="Missing 'title' in request body")

        if direct_reply:
            full_response = await generate_complete_content_direct(title, is_brief)
            return {"full_response": full_response}
        else:
            return StreamingResponse(
                stream_complete_content_response(title, is_brief),
                media_type="text/event-stream"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/generate_complete_content 请求处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


async def generate_complete_content_direct(title: str, is_brief: bool) -> str:
    """直接生成完整内容（非流式）"""
    try:
        logger.info("开始生成完整内容（直接模式）")
        overview_prompt = create_overview_prompt(title, is_brief)
        overview_content = await call_ollama_non_stream(overview_prompt, config.get_model_name(is_brief), t=0.01)
        logger.info(f"概述生成完成，长度: {len(overview_content)}")

        paragraph_prompt = create_paragraph_prompt(title, overview_content)
        paragraph_content = await call_ollama_non_stream(paragraph_prompt, config.get_model_name(is_brief), t=0.7)
        logger.info(f"正文生成完成，长度: {len(paragraph_content)}")

        complete_content = f"""## 概述\n\n{overview_content}\n\n## 正文内容\n\n{paragraph_content}"""
        return complete_content
    except Exception as e:
        logger.error(f"生成完整内容过程中发生错误: {e}", exc_info=True)
        raise e


async def stream_complete_content_response(title: str, is_brief: bool):
    """流式生成完整内容响应"""
    overview_content = ""
    try:
        logger.info("开始流式生成完整内容")
        overview_prompt = create_overview_prompt(title, is_brief)
        model_name = config.get_model_name(is_brief)

        # 第一阶段：生成概述
        async for text_chunk in call_ollama_stream(overview_prompt, model_name, t=0.1):
            if isinstance(text_chunk, dict) and "error" in text_chunk:
                error_chunk = build_stream_chunk(text_chunk, model_name, "overview_error", is_error=True)
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                return
            overview_content += text_chunk
            chunk = build_stream_chunk(text_chunk, model_name, "overview")
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        logger.info(f"概述生成完成，长度: {len(overview_content)}")

        # 第二阶段：生成正文
        paragraph_prompt = create_paragraph_prompt(title, overview_content)
        paragraph_content = ""

        async for text_chunk in call_ollama_stream(paragraph_prompt, model_name, t=0.7):
            if isinstance(text_chunk, dict) and "error" in text_chunk:
                error_chunk = build_stream_chunk(text_chunk, model_name, "paragraph_error", is_error=True)
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                return
            paragraph_content += text_chunk
            chunk = build_stream_chunk(text_chunk, model_name, "paragraph")
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        logger.info(f"正文生成完成，长度: {len(paragraph_content)}")

        # 发送完成信号
        final_chunk = build_stream_chunk("", model_name, "complete", finish_reason="stop")
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

        complete_content = f"""#### 概述\n\n{overview_content}\n\n#### 正文内容\n\n{paragraph_content}"""
        summary_chunk = {
            "full_response": complete_content,
            "overview_content": overview_content,
            "paragraph_content": paragraph_content,
            "stage": "summary"
        }
        yield f"data: {json.dumps(summary_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        logger.info("完整内容流式生成完成")

    except Exception as e:
        logger.error(f"流式生成完整内容过程中发生错误: {e}", exc_info=True)
        error_chunk = {"error": {"message": str(e), "type": "server_error", "stage": "unknown"}}
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"


async def create_enhanced_overview_prompt(title: str, is_brief: bool) -> str:
    """创建增强版概述生成提示词（包含知识检索）"""
    if is_brief:
        return create_overview_prompt(title, is_brief)

    try:
        keyword_payload = {"text": title}
        key_words = await key_word_extract(config.KEYWORD_EXTRACT_URL, keyword_payload)

        if not key_words:
            related_knowledge_new = ""
        else:
            related_knowledge_list = await search_related_knowledge(key_words)
            key_words_ner = await ner_extract(config.NER_EXTRACT_URL, {"text": str(title).split("\n")[0]})
            encoded_vector = encode.encode([key_words_ner])
            i_v = encoded_vector[0].tolist()
            related_knowledge = query_vector_collection(client_weaviate, 'kg_wenda_large', i_v, k=1)

            if len(related_knowledge) > 0 and related_knowledge[0].get('certainty') > 0.8:
                related_knowledge_new = related_knowledge[0].get('desc')
            else:
                related_knowledge_new = str(related_knowledge_list)

        knowledge_section = f"相关知识关联如下:\n```{related_knowledge_new}```\n\n" if related_knowledge_new else ""
        return f"""
请将以下大纲进行扩写,不要有多余的回复和解释:
1. 保持原有标题层级结构和编号方式不变
2. 在每个标题下方添加 2-3 句概括性描述
3. 描述要简明扼要,突出重点
{knowledge_section}我要扩充的大纲内容如下,不要有多余的回复:
{title}  /no_think
"""
    except Exception as e:
        logger.error(f"创建增强版概述提示词失败: {e}", exc_info=True)
        return create_overview_prompt(title, is_brief)


# ==================== 主入口 ====================
if __name__ == "__main__":
    logger.info("应用程序主入口点执行。")
    uvicorn.run(app, host="0.0.0.0", port=5003)
    logger.info("Uvicorn 服务器已停止。")