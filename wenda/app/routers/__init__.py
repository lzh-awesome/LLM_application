"""routers 模块"""

from app.routers.chat import router as chat_router
from app.routers.entity import router as entity_router
from app.routers.recommend import router as recommend_router
from app.routers.topic import router as topic_router
from app.routers.translate import router as translate_router
from app.routers.kg_wenda import router as kg_wenda_router
from app.routers.media import router as media_router

__all__ = [
    "chat_router",
    "topic_router",
    "recommend_router",
    "translate_router",
    "entity_router",
    "kg_wenda_router",
    "media_router",
]