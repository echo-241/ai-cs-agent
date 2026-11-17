"""Agent 编排：意图识别 → 路由 → 检索/工具 → 生成。

路由策略：
- doc_qa        → 混合检索 + 生成（带引用）
- order_query / place_order → Function Calling：模型自主选工具，结果回灌后生成回复；
                   模型未调用工具时走原来的硬编码查询 / 追问兜底
- return_refund → 混合检索（退换货政策）+ 建工单
- complaint     → 建投诉工单 + 安抚话术
- chitchat      → 直接闲聊，不检索
"""
from __future__ import annotations

import json
import time
from typing import Generator

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.agent.intent import IntentRecognizer, IntentResult
from app.agent.memory import Memory
from app.agent.tools import CATALOG, MOCK_ORDERS, TOOL_ANSWER_PROMPT, decide_tool_call, run_tool
from app.common import db
from app.common.clients import get_chat_model
from app.config.settings import settings

BASE_PROMPT = """
你是「星域智能」的售后客服助手。

回答要求：
1. 只依据【参考资料】作答，不得编造条款、金额、时效或型号参数；
2. 先给结论，再给依据；引用资料时用 [1][2] 标注对应编号；
3. 资料里没有写明的内容，直接说「资料中未写明」，并建议转人工；
4. 涉及金额、时效、保修期、型号参数时，必须给出具体数值；
5. 语气简洁专业，不要寒暄铺垫，不要输出思考过程。
""".strip()

CHITCHAT_PROMPT = """
你是「星域智能」的售后客服助手。用户此刻在闲聊或打招呼。
用一两句话礼貌回应，并主动提示你可以协助的范围：售后政策、商品型号、错误码、退换货与退款进度。
不要编造任何业务数据。
""".strip()

TICKET_PROMPT = """
你是「星域智能」的售后客服助手。用户提出退换货或投诉。
请先共情，再说明处理路径与需要用户提供的材料（订单号、型号、故障描述或错误码）。
不要承诺具体赔付金额或时限，不要编造政策条款。
""".strip()

MODE_LABEL = {
    "hybrid_rerank": "混合检索 + 重排",
    "hybrid": "混合检索",
    "dense_rerank": "向量检索 + 重排",
    "dense": "纯向量检索",
    "sparse": "纯 BM25",
}


class AgentOrchestrator:
    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self.intent = IntentRecognizer()

    # ------------------------------------------------------------ 工具
    def _query_order(self, order_id: str) -> str:
        order = MOCK_ORDERS.get(order_id)
        if not order:
            return f"订单 {order_id} 未查询到，请核对订单号是否为 18 位字符。"
        return (f"订单 {order_id}：{order['item']}，当前状态 {order['status']}，"
                f"承运商 {order['carrier']}，预计送达 {order['eta']}。")

    # ------------------------------------------------------------ 主流程
    def stream(self, session_id: str, user_input: str,
               mode: str = "hybrid_rerank") -> Generator[dict, None, None]:
        memory = Memory(session_id)
        memory.add_user(user_input)

        started = time.perf_counter()
        result: IntentResult = self.intent.recognize(user_input, memory.history_text())
        memory.update_facts(result.slots)

        primary = result.intents[0]
        yield {
            "type": "route",
            "intents": result.intents,
            "confidence": result.confidence,
            "slots": result.slots,
            "mode": mode,
            "mode_label": MODE_LABEL.get(mode, mode),
        }

        # ---------------------------------------------- 分流
        if primary == "chitchat":
            messages = [SystemMessage(content=CHITCHAT_PROMPT)] + memory.build_context()
            messages.append(HumanMessage(content=user_input))
            text = yield from self._stream_tokens(messages, session_id, memory, [], "chitchat")
            note, _ = self._extra_note(session_id, user_input, result.intents, primary)
            if note:
                yield {"type": "token", "text": note}
                memory.add_ai(note.strip())
            yield {"type": "done", "answer": text + note,
                   "elapsed_ms": round((time.perf_counter() - started) * 1000)}
            return

        if primary in ("order_query", "place_order"):
            # Function Calling：调哪个工具、传什么参数由模型决定，这里不写业务 if-else。
            call = decide_tool_call(user_input, memory.facts())
            if call:
                yield {"type": "tool", "name": call["name"], "args": call["args"]}
                outcome = run_tool(call["name"], call["args"], session_id)
                yield {
                    "type": "tool_result",
                    "ok": outcome["ok"],
                    "message": outcome.get("message", ""),
                    "data": outcome.get("data"),
                }
                yield {"type": "sources", "items": []}
                system = SystemMessage(
                    content=f"{TOOL_ANSWER_PROMPT}\n\n【工具结果】\n{json.dumps(outcome, ensure_ascii=False)}"
                )
                messages = [system] + memory.build_context() + [HumanMessage(content=user_input)]
                text = yield from self._stream_tokens(messages, session_id, memory, [], primary)
                note, extra_ticket = self._extra_note(session_id, user_input, result.intents, primary)
                if note:
                    yield {"type": "token", "text": note}
                    memory.add_ai(note.strip())
                    text += note
                yield {
                    "type": "done", "answer": text,
                    "tool": call["name"], "tool_ok": outcome["ok"],
                    "ticket_id": extra_ticket,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                }
                return

            # 模型未调用工具（不可用或判断无需调用）：沿用原有兜底，保证链路不中断
            if primary == "order_query":
                order_id = result.slots.get("order_id") or memory.facts().get("order_id")
                answer = (self._query_order(str(order_id)) if order_id else
                          "请提供一下订单号，我这边帮你查物流。（订单号一般在订单详情页顶部，18 位字符）")
            else:
                answer = f"请告诉我商品型号和收货地址，我帮你下单。（可售型号：{'、'.join(CATALOG)}）"
            note, _ = self._extra_note(session_id, user_input, result.intents, primary)
            answer += note
            memory.add_ai(answer)
            yield {"type": "sources", "items": []}
            for chunk in self._chunk_text(answer):
                yield {"type": "token", "text": chunk}
            yield {"type": "done", "answer": answer,
                   "elapsed_ms": round((time.perf_counter() - started) * 1000)}
            return

        # ---------------------------------------------- 需要知识库
        hits = self.retriever.search(user_input, mode=mode, recall_k=settings.recall_k, final_k=settings.final_k)
        sources = [
            {
                "n": i,
                "doc": h.get("doc_name", ""),
                "page": h.get("page"),
                "heading": h.get("heading") or (h["text"].split("\n")[0][:40]),
                "score": round(float(h.get("rerank_score", h.get("score", 0.0)) or 0.0), 4),
            }
            for i, h in enumerate(hits, 1)
        ]
        yield {"type": "sources", "items": sources}

        ticket_id = None
        if primary in ("return_refund", "complaint"):
            ticket_id = db.create_ticket(session_id, primary, user_input)

        context = "\n\n".join(
            f"[{i}] 来源：{h.get('doc_name', '')} 第{h.get('page', '?')}页\n{h['text']}"
            for i, h in enumerate(hits, 1)
        )

        system_prompt = BASE_PROMPT if primary in ("doc_qa", "return_refund") else TICKET_PROMPT
        messages = [SystemMessage(content=system_prompt)] + memory.build_context()
        messages.append(SystemMessage(content=f"【参考资料】\n{context or '（未检索到相关资料）'}"))
        messages.append(HumanMessage(content=user_input))

        text = yield from self._stream_tokens(messages, session_id, memory, sources, primary)
        note, extra_ticket = self._extra_note(session_id, user_input, result.intents, primary)
        if note:
            yield {"type": "token", "text": note}
            memory.add_ai(note.strip())
            text += note
        yield {
            "type": "done",
            "answer": text,
            "ticket_id": ticket_id or extra_ticket,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    # ------------------------------------------------------------ 内部
    @staticmethod
    def _extra_note(session_id: str, user_input: str, intents: list[str],
                    primary: str) -> tuple[str, int | None]:
        """处理主意图之外的次要意图。

        一句话可能同时含多个意图（「机器报 E1024，我要退货」），
        主意图决定回答内容，次要的退换货/投诉在这里登记工单，回一句附言。
        """
        secondary = [i for i in intents if i != primary and i in ("return_refund", "complaint")]
        if not secondary:
            return "", None
        label = "退换货" if secondary[0] == "return_refund" else "投诉"
        ticket_id = db.create_ticket(session_id, secondary[0], user_input)
        return f"\n\n另外，已为你登记{label}工单（#{ticket_id}），客服会尽快跟进。", ticket_id

    def _stream_tokens(self, messages, session_id: str, memory: Memory,
                       sources: list, primary: str) -> Generator[dict, None, str]:
        buffer: list[str] = []
        try:
            for chunk in get_chat_model().stream(messages):
                piece = getattr(chunk, "content", "") or ""
                if not piece:
                    continue
                buffer.append(piece)
                yield {"type": "token", "text": piece}
        except Exception as exc:  # noqa: BLE001
            fallback = f"抱歉，生成回答时出现异常（{exc.__class__.__name__}），请稍后重试。"
            buffer.append(fallback)
            yield {"type": "token", "text": fallback}

        answer = "".join(buffer)
        memory.add_ai(answer)
        memory.maybe_summarize()
        return answer

    @staticmethod
    def _chunk_text(text: str, size: int = 12) -> Generator[str, None, None]:
        for i in range(0, len(text), size):
            yield text[i:i + size]
            time.sleep(0.01)
