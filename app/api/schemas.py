"""接口出入参定义。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    user_id: str = Field(default="demo", description="用户标识")
    title: str = Field(default="新的会话")


class ChatRequest(BaseModel):
    session_id: str = Field(description="会话 id")
    message: str = Field(min_length=1, description="用户输入")
    mode: str = Field(default="hybrid_rerank", description="检索配置")


class ShopOrderRequest(BaseModel):
    """商城页人工购买：与 AI 代买共用 run_tool 的参数校验。"""

    session_id: str = Field(default="demo", description="商城会话 id")
    sku: str = Field(default="", description="商品型号")
    quantity: int = Field(default=1, ge=1, le=10, description="购买数量 1~10")
    address: str = Field(default="", max_length=120, description="收货地址")


class AgentBuyRequest(BaseModel):
    """商城页让觅淘助手代买：自然语言 → LLM 自主选工具 → 下单。"""

    session_id: str = Field(default="demo", description="商城会话 id")
    message: str = Field(min_length=1, max_length=300, description="给助手的自然语言指令")
    address: str = Field(default="", max_length=120, description="页面上已填的地址，作为会话事实传给助手")


class ShopChatRequest(BaseModel):
    """商城 AI 导购的多轮对话输入。"""

    session_id: str = Field(default="demo", description="商城会话 id")
    message: str = Field(min_length=1, max_length=500, description="用户这一轮说的话")
    address: str = Field(default="", max_length=120, description="页面上填写的地址，作为会话事实")


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    chunks: int
    created_at: str


class UploadResult(BaseModel):
    doc_id: str
    doc_name: str
    chunks: int
    total_vectors: int
    bm25_size: int


class ChatMeta(BaseModel):
    session_id: str
    intents: list[str]
    confidence: float
    slots: dict[str, Any]
    mode: str
    mode_label: str
