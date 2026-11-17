"""三层记忆：事实槽 + 增量摘要 + 最近消息。

原 ChatService 的三层结构保留，两处改动：
1. 存储从进程内 dict 改为 SQLite，进程重启与多 worker 下不丢会话；
2. 事实与摘要改用 SystemMessage 承载 —— 原实现用 AIMessage，
   模型会把「关键事实」误当成自己说过的话，容易在后续轮次里混淆。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.common import db
from app.common.clients import get_chat_model
from app.config.settings import settings

SUMMARY_PROMPT = """
你是对话摘要助手。请把【旧摘要】与【新对话】合并成一段新摘要。

要求：
1. 保留订单号、商品型号、错误码、退换货诉求等关键事实；
2. 去掉寒暄与重复表述；
3. 总长度不超过 200 字；
4. 只输出摘要正文，不要加「摘要：」前缀。
""".strip()


class Memory:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    # ------------------------------------------------------------ 事实
    def facts(self) -> dict:
        return db.get_facts(self.session_id)

    def update_facts(self, slots: dict) -> None:
        db.put_facts(self.session_id, slots or {})

    # ------------------------------------------------------------ 消息
    def add_user(self, text: str) -> None:
        db.add_message(self.session_id, "user", text)

    def add_ai(self, text: str) -> None:
        db.add_message(self.session_id, "assistant", text)

    def history_text(self, limit: int = 6) -> str:
        rows = db.recent_messages(self.session_id, limit)
        return "\n".join(f"{'用户' if r['role'] == 'user' else '客服'}：{r['content']}" for r in rows)

    # ------------------------------------------------------------ 摘要
    def maybe_summarize(self) -> None:
        summary, upto_id = db.get_summary(self.session_id)
        pending = db.messages_since(self.session_id, after_id=upto_id)
        if len(pending) < settings.summary_threshold:
            return

        dialogue = "\n".join(
            f"{'用户' if m['role'] == 'user' else '客服'}：{m['content']}" for m in pending
        )
        prompt = f"旧摘要：{summary or '（无）'}\n新对话：\n{dialogue}"
        try:
            new_summary = get_chat_model().invoke(
                [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=prompt)]
            ).content
        except Exception:  # noqa: BLE001
            return
        db.save_summary(self.session_id, str(new_summary).strip(), pending[-1]["id"])

    # ------------------------------------------------------------ 上下文
    def build_context(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = []

        facts = self.facts()
        if facts:
            pairs = "；".join(f"{k}={v}" for k, v in facts.items())
            messages.append(SystemMessage(content=f"【已确认的会话事实】{pairs}"))

        summary, _ = db.get_summary(self.session_id)
        if summary:
            messages.append(SystemMessage(content=f"【历史对话摘要】{summary}"))

        for row in db.recent_messages(self.session_id, settings.history_window):
            cls = HumanMessage if row["role"] == "user" else AIMessage
            messages.append(cls(content=row["content"]))
        return messages
