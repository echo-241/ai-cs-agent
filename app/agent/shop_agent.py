"""觅淘商城的多轮对话助手：能聊天导购，也能在对话里自主下单 / 查单。

和旧的 /api/shop/agent-buy（一次性「说买什么」）不同：
- 每轮把最近对话历史一起交给 LLM，问过什么、看过哪款都记得住；
- 不强制调用工具，聊天、推荐、对比、售后咨询都正常回答；
- 用户表达购买意图时，LLM 依据 function schema 自主生成 tool_calls，
  工具结果回灌后再生成一句自然语言回复；
- 对话写进 SQLite，刷新页面还能接着聊。
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.tools import TOOL_SPECS, run_tool
from app.common import db, shop_db
from app.common.clients import get_tool_model

SYSTEM = """你是「觅淘」商城的智能导购助手，和用户自然聊天，也能在对话里直接帮他下单、查订单。

要求：
1. 可以打招呼、介绍和对比商品、按预算推荐、回答参数与售后问题；不要一上来就催下单；
2. 用户明确要购买时调用 create_order，要查物流 / 订单时调用 query_order；
   参数不全就先向用户追问一句，绝不编造型号、地址、订单号或金额；
3. 用户没另说地址时，用【会话事实】里的收货地址填 address；地址为空必须先追问；
4. 订单号、金额、库存、状态一律以工具结果为准，绝不编造；
5. 语气像真人导购，简洁自然，一两句话为主，不要输出 JSON、思考过程或工具名；
6. 一次只调用一个工具。
"""


def _catalog_text() -> str:
    try:
        items = shop_db.list_products()
    except Exception:  # noqa: BLE001 —— MySQL 暂不可用时退回演示目录
        items = [dict(p) for p in shop_db.SEED_PRODUCTS]
    lines = []
    for p in items:
        meta = shop_db.PRODUCT_META.get(p["sku"], {})
        lines.append(
            f"- {p['sku']}｜{p['item']}｜¥{p['price']}｜库存 {p['stock']} 件｜{meta.get('desc', '')}"
        )
    return "\n".join(lines)


def _recent_orders_text(session_id: str) -> str:
    try:
        rows = shop_db.list_orders(session_id, 3)
    except Exception:  # noqa: BLE001
        return "（无）"
    return "；".join(f"{r['order_id']}（{r['item']}，{r['status']}）" for r in rows) or "（无）"


def _system_text(session_id: str, address: str) -> str:
    return (SYSTEM
            + "\n\n【在售商品】\n" + _catalog_text()
            + "\n\n【会话事实】\n收货地址：" + (address or "（未填写）")
            + "\n该会话最近订单：" + _recent_orders_text(session_id))


def chat(session_id: str, message: str, address: str = "") -> dict:
    """返回 {reply, tool, ok}；tool 里带工具名、参数与执行结果，供前端展示调用过程。"""
    history = db.recent_messages(session_id, 10)
    history_msgs = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]
    system = _system_text(session_id, address)
    base = [SystemMessage(content=system)] + history_msgs + [HumanMessage(content=message)]

    model = get_tool_model()
    try:
        first = model.bind_tools(TOOL_SPECS).invoke(base)
    except Exception as exc:  # noqa: BLE001 —— 模型不可用不阻断页面
        return {
            "reply": f"模型服务暂时不可用（{exc.__class__.__name__}），可以先直接点「立即购买」，或稍后再聊。",
            "tool": None, "ok": False,
        }

    calls = getattr(first, "tool_calls", None) or []
    tool_info = None
    if calls:
        call = calls[0]
        name = str(call.get("name", ""))
        args = call.get("args") or {}
        outcome = run_tool(name, args, session_id, source="agent")
        tool_info = {
            "name": name, "args": args, "ok": outcome["ok"],
            "message": outcome.get("message", ""), "data": outcome.get("data"),
        }
        # 结果回灌：只保留一个 SystemMessage，避免多 system 消息在模型侧报错
        followup_system = SystemMessage(
            content=system + "\n\n【本轮工具调用结果】\n" + json.dumps(outcome, ensure_ascii=False)
            + "\n请依据以上结果用自然语言回复用户；失败时按提示向用户追问，不要编造数据。"
        )
        try:
            second = model.invoke(
                [followup_system] + history_msgs + [HumanMessage(content=message)]
            )
            reply = str(getattr(second, "content", "") or "").strip()
        except Exception:  # noqa: BLE001
            reply = ""
        if not reply:
            reply = outcome.get("message", "操作已完成。")
    else:
        reply = str(getattr(first, "content", "") or "").strip() or "抱歉，我刚才没听清，可以再说一遍吗？"

    db.add_message(session_id, "user", message)
    db.add_message(session_id, "assistant", reply)
    return {"reply": reply, "tool": tool_info, "ok": True}
