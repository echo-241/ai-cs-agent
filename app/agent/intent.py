"""意图识别。

相对原 ChatService 的改动：
1. 增加 doc_qa 意图（知识库问答），原实现只有物流/改地址/退款/投诉四类；
2. 修复原实现不传对话历史的缺陷 —— 原代码 `recognize(user_input)` 丢掉了 history，
   导致「我想退货」→「订单号是 xxx」这类跨轮槽位填充必然失败；
3. 意图返回后做白名单校验，模型给出未定义意图时降级到 chitchat，避免 KeyError。
"""
from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.common.clients import get_chat_model

INTENTS = ["doc_qa", "order_query", "place_order", "return_refund", "complaint", "chitchat"]

PROMPT = """
你是电商售后客服的意图识别器。根据用户输入与上下文，识别意图并抽取槽位。

## 可选意图（intents，可多选）
- doc_qa：咨询售后政策、商品型号、错误码、保修、发票、退款规则等知识库问题
- order_query：查询订单物流、发货进度、签收情况
- place_order：要购买、下单、再买一台某型号商品
- return_refund：申请退货、换货、退款
- complaint：投诉、表达不满、要求升级处理
- chitchat：问候、寒暄、与业务无关的闲聊

## 槽位（slots）
- order_id：订单号，通常为一串数字或字母数字组合
- model：商品型号，例如 XR-200、XR-200S、XG-100 Pro
- error_code：错误码，例如 E1024、R002、LW05
- reason：退换货或投诉的原因摘要
- sku：下单时想买的商品型号
- quantity：下单数量
- address：收货地址

## 判定规则
1. 问「E1024 是什么意思」「XR-200S 电池多大」「保修多久」→ doc_qa
2. 问「我的快递到哪了」→ order_query
3. 说「我要买一台 XR-200S」「帮我下单」→ place_order
4. 说「我要退货」「不想要了」→ return_refund
5. 说「太差了」「要投诉」→ complaint
6. 只打招呼或与业务无关 → chitchat
7. 一句话可以同时包含多个意图，例如「机器报 E1024，我要退货」
8. 若上下文已出现过订单号/型号/地址，本轮未重复提及也要继承，写入对应槽位

## 置信度
0.9-1.0 意图明确且关键槽位完整；0.7-0.9 意图明确但缺槽位；
0.5-0.7 意图模糊需推测；0.0-0.5 无法判断。
""".strip()


class IntentResult(BaseModel):
    intents: list[str] = Field(default_factory=list, description="意图列表")
    slots: dict[str, Any] = Field(default_factory=dict, description="槽位字典")
    confidence: float = Field(default=0.0, description="置信度 0~1")


class IntentRecognizer:
    def __init__(self) -> None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT),
            ("ai", "最近对话：\n{chat_history}"),
            ("human", "用户输入：{user_input}"),
        ])
        self._chain = prompt | get_chat_model().with_structured_output(IntentResult)

    def recognize(self, user_input: str, chat_history: str = "") -> IntentResult:
        try:
            result = self._chain.invoke({"chat_history": chat_history or "（无）", "user_input": user_input})
        except Exception:  # noqa: BLE001
            # 模型不可用时不阻断主链路，降级为知识库问答
            return IntentResult(intents=["doc_qa"], slots={}, confidence=0.0)

        if not isinstance(result, IntentResult):
            return IntentResult(intents=["doc_qa"], slots={}, confidence=0.0)

        valid = [i for i in result.intents if i in INTENTS]
        if not valid:
            valid = ["chitchat"]
        return IntentResult(intents=valid, slots=result.slots or {}, confidence=result.confidence)
