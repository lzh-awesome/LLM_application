"""
配置管理模块
支持 YAML 配置文件和环境变量
"""
from dataclasses import dataclass, field
from typing import Optional
import os
import yaml


@dataclass
class WeaviateConfig:
    """Weaviate 数据库配置"""
    http_host: str = "weaviate"
    http_port: int = 8080
    http_secure: bool = False
    grpc_host: str = "weaviate"
    grpc_port: int = 50051
    grpc_secure: bool = False
    api_key: str = "test-secret-key"


@dataclass
class ArangoConfig:
    """ArangoDB 数据库配置"""
    host: str = "http://10.117.254.37:8529"
    database: str = "wiki"
    username: str = "root"
    password: str = "root"


@dataclass
class ModelConfig:
    """模型路径配置"""
    bge_m3_path: str = "/workspace/codes/deepseek/deepseek_model/bge-m3"
    bge_vl_path: str = "/workspace/codes/deepseek/multimodal/retrieval/image_retrieval/BGE-VL-base"
    video_clip_path: str = "/workspace/codes/deepseek/multimodal/int_retrieval/video_retrieval/VideoCLIP-XL"
    video_clip_weights: str = "/workspace/codes/deepseek/multimodal/int_retrieval/video_retrieval/VideoCLIP-XL/VideoCLIP-XL.bin"


@dataclass
class Settings:
    """全局配置"""
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)
    arango: ArangoConfig = field(default_factory=ArangoConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 1000

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """从 YAML 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        weaviate_data = data.get("weaviate", {})
        arango_data = data.get("arango", {})
        models_data = data.get("models", {})

        return cls(
            weaviate=WeaviateConfig(**weaviate_data),
            arango=ArangoConfig(**arango_data),
            models=ModelConfig(**models_data),
            batch_size=data.get("batch_size", 1000)
        )

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载配置"""
        weaviate = WeaviateConfig(
            http_host=os.getenv("WEAVIATE_HTTP_HOST", "weaviate"),
            http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
            http_secure=os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true",
            grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "weaviate"),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
            grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true",
            api_key=os.getenv("WEAVIATE_API_KEY", "test-secret-key")
        )

        arango = ArangoConfig(
            host=os.getenv("ARANGO_HOST", "http://10.117.254.37:8529"),
            database=os.getenv("ARANGO_DATABASE", "wiki"),
            username=os.getenv("ARANGO_USERNAME", "root"),
            password=os.getenv("ARANGO_PASSWORD", "root")
        )

        models = ModelConfig(
            bge_m3_path=os.getenv("BGE_M3_PATH", "/workspace/codes/deepseek/deepseek_model/bge-m3"),
            bge_vl_path=os.getenv("BGE_VL_PATH", "/workspace/codes/deepseek/multimodal/retrieval/image_retrieval/BGE-VL-base"),
            video_clip_path=os.getenv("VIDEO_CLIP_PATH", "/workspace/codes/deepseek/multimodal/int_retrieval/video_retrieval/VideoCLIP-XL"),
            video_clip_weights=os.getenv("VIDEO_CLIP_WEIGHTS", "/workspace/codes/deepseek/multimodal/int_retrieval/video_retrieval/VideoCLIP-XL/VideoCLIP-XL.bin")
        )

        return cls(
            weaviate=weaviate,
            arango=arango,
            models=models,
            batch_size=int(os.getenv("BATCH_SIZE", "1000"))
        )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        """
        加载配置（优先级：config_path > 环境变量 > 默认值）
        """
        if config_path and os.path.exists(config_path):
            return cls.from_yaml(config_path)

        # 检查默认配置文件路径
        default_paths = [
            "config.yaml",
            "config/config.yaml",
            os.path.join(os.path.dirname(__file__), "config.yaml")
        ]

        for path in default_paths:
            if os.path.exists(path):
                return cls.from_yaml(path)

        # 如果没有配置文件，从环境变量加载
        return cls.from_env()


# 全局配置实例（延迟加载）
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = Settings.load(config_path)
    return _settings


def reset_settings():
    """重置全局配置（主要用于测试）"""
    global _settings
    _settings = None