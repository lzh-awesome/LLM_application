"""FastAPI 应用入口"""

import argparse
import asyncio
import logging

import uvicorn
from contextlib import asynccontextmanager # 简单说：它确保应用在启动和关闭时能正确执行初始化和清理工作。

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dependencies import init_services, shutdown_services
from app.routers import (
    chat_router,
    entity_router,
    kg_wenda_router,
    media_router,
    recommend_router,
    topic_router,
    translate_router,
)

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.logging.file, encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info("Application starting up...")
    init_services()

    # Warmup LLM
    try:
        await warmup_llm()
    except Exception as e:
        logger.warning(f"LLM warmup failed: {e}")

    yield

    # Shutdown
    logger.info("Application shutting down...")
    shutdown_services()


async def warmup_llm():
    """预热 LLM 模型"""
    logger.info("Warming up LLM...")
    from app.services import llm_service

    response = await llm_service.chat(
        messages=[
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好"},
        ],
        max_tokens=50,
    )
    logger.info(f"LLM warmup successful: {response[:50]}...")


# 创建 FastAPI 应用
app = FastAPI(
    title="问答服务 API",
    description="提供简洁/深入模式问答、问题推荐、话题总结、翻译等功能",
    version="2.0.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(topic_router)
app.include_router(recommend_router)
app.include_router(translate_router)
app.include_router(entity_router)
app.include_router(kg_wenda_router)
app.include_router(media_router)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动问答服务")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8104,
        help="服务监听端口",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )