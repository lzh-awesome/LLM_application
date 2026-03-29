"""依赖注入模块"""

import logging
from typing import Generator

from app.services import (
    arango_service,
    embedding_service,
    llm_service,
    reranker_service,
    weaviate_service,
)

logger = logging.getLogger(__name__)


def get_llm_service():
    """获取 LLM 服务实例"""
    return llm_service


def get_embedding_service():
    """获取 Embedding 服务实例"""
    return embedding_service


def get_arango_service():
    """获取 ArangoDB 服务实例"""
    return arango_service


def get_weaviate_service():
    """获取 Weaviate 服务实例"""
    return weaviate_service


def get_reranker_service():
    """获取 Reranker 服务实例"""
    return reranker_service


def init_services() -> None:
    """初始化所有服务"""
    logger.info("Initializing services...")

    # 初始化 Embedding 模型
    embedding_service.load_model()
    logger.info("Embedding model loaded")

    # 初始化 Reranker 模型
    reranker_service.load_model()
    logger.info("Reranker model loaded")

    # 初始化数据库连接
    arango_service.connect()
    weaviate_service.connect()
    logger.info("Database connections established")


def shutdown_services() -> None:
    """关闭所有服务"""
    logger.info("Shutting down services...")

    arango_service.disconnect()
    weaviate_service.disconnect()
    logger.info("Services shutdown complete")