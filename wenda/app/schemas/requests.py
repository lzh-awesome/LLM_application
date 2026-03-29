"""Pydantic 请求/响应模型定义"""

from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field


# ============ Chat 相关 ============

class Message(BaseModel):
    """聊天消息"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    messages: List[Message]
    is_concise: bool = True
    reply_direct: bool = False


class ChatResponse(BaseModel):
    """聊天响应（流式，实际使用 StreamingResponse）"""
    pass


# ============ Topic Summary 相关 ============

class SummarizeTopicRequest(BaseModel):
    """会话主题总结请求"""
    text: str = Field(..., description="待总结的文本内容")


class SummarizeTopicResponse(BaseModel):
    """会话主题总结响应"""
    topic: str


# ============ Recommend Questions 相关 ============

class RecommendQuestionsRequest(BaseModel):
    """问题推荐请求"""
    text: str = Field(..., description="用户输入的问题")


class RecommendQuestionsResponse(BaseModel):
    """问题推荐响应"""
    questions: List[str]


# ============ Translate 相关 ============

class TranslateRequest(BaseModel):
    """翻译请求"""
    text: str = Field(..., description="待翻译的中文文本")


class TranslateResponse(BaseModel):
    """翻译响应"""
    translation: str


# ============ Entity Match Tags 相关 ============

class MatchTagsRequest(BaseModel):
    """标签匹配请求"""
    tags: List[str] = Field(default_factory=list, description="标签列表")
    entity: Optional[Dict] = None


class MatchTagsResponse(BaseModel):
    """标签匹配响应"""
    matched_tags: Union[List[str], Set[str]]


# ============ Entity Data 处理相关 ============

class EntityData(BaseModel):
    """实体数据"""
    labels: Optional[Dict[str, str]] = None
    descs: Optional[Dict[str, str]] = None
    claims: Optional[Dict[str, List]] = None


class ProcessedEntity(BaseModel):
    """处理后的实体数据"""
    labels: Optional[str] = None
    descs: Optional[str] = None