"""配置管理模块 - 使用 Pydantic Settings 实现类型安全的配置访问"""

import argparse
from functools import lru_cache
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class ArangoConfig(BaseModel):
    """ArangoDB 配置"""
    host: str
    username: str
    password: str
    database_name: str
    entity_collection: str
    entity_description_collection: str
    link_collection: str


class LLMConfig(BaseModel):
    """LLM 配置"""
    api_key: str
    base_url: str
    model: str
    ner_url: str
    ollama_api_url: str


class WeaviateConfig(BaseModel):
    """Weaviate 配置"""
    http_host: str
    http_port: int
    grpc_host: str
    grpc_port: int
    api_key: str


class EmbeddingModelConfig(BaseModel):
    """Embedding 模型配置"""
    bge_m3_path: str


class RerankerConfig(BaseModel):
    """Reranker 模型配置"""
    model_path: str
    max_length: int = 512


class VLLMConfig(BaseModel):
    """vLLM 多模态模型配置"""
    api_base: str
    api_key: str
    model: str


class CORSConfig(BaseModel):
    """CORS 配置"""
    origins: List[str] = Field(default_factory=lambda: ["*"])


class LoggingConfig(BaseModel):
    """日志配置"""
    file: str = "app.log"
    level: str = "DEBUG"


class Settings(BaseModel):
    """应用配置"""
    arango: ArangoConfig
    llm: LLMConfig
    weaviate: WeaviateConfig
    embedding_model: EmbeddingModelConfig
    reranker: RerankerConfig
    vllm: VLLMConfig
    cors: CORSConfig = CORSConfig()
    logging: LoggingConfig = LoggingConfig()


def get_config_path() -> str:
    """获取配置文件路径，支持命令行参数"""
    parser = argparse.ArgumentParser(description="启动问答服务")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='指定配置文件路径 (例如: --config config_prod.yaml)'
    )
    args, _ = parser.parse_known_args()
    return args.config


@lru_cache()# 缓存函数结果：第一次调用 load_settings() 时，会执行函数体加载配置文件并返回结果。后续再调用该函数时，会直接返回第一次缓存的结果，而不会重新读取和解析配置文件
def load_settings() -> Settings:
    """加载配置并缓存"""
    config_path = get_config_path()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            raise ValueError("配置文件为空")

        return Settings(**config_data)

    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"解析配置文件失败: {e}")


# 全局配置实例
settings = load_settings()