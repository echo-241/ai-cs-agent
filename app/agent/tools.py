"""业务工具（Function Calling）：下单 / 订单查询。

链路：
  1. TOOL_SPECS 由 Pydantic 参数模型自动生成 JSON Schema，交给模型；
  2. decide_tool_call() 由模型自主决定「调不调 / 调哪个 / 传什么参数」；
  3. run_tool() 做白名单分发、参数校验、业务校验、异常兜底；
  4. 结果回灌模型生成自然语言回复（见 orchestrator._tool_turn）。

约定：工具层不向 SSE 主链路抛异常。模型给的参数一律视为不可信输入，
失败统一转成 {"ok": false, "message": ...}，让模型据此追问，而不是崩掉整个对话。
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.common import shop_db as store
from app.common.clients import get_tool_model

logger = logging.getLogger(__name__)

# 商品与订单统一落 MySQL（shop_db）；这里的 CATALOG 只作展示与兜底，
# 实际价格、库存、可售型号一律以 shop_db.products 表为准。
CATALOG: dict[str, dict[str, Any]] = {
    p["sku"]: {"item": p["item"], "price": p["price"], "stock": p["stock"]}
    for p in store.SEED_PRODUCTS
}

# 与旧版共用的三条演示订单（真实项目应替换为订单服务调用）
MOCK_ORDERS = {
    "06715421bjfab412": {"status": "运输中", "carrier": "顺丰速运", "eta": "2026-11-12", "item": "XR-200S 扫地机器人"},
    "06715421bjfab413": {"status": "已签收", "carrier": "京东物流", "eta": "2026-11-08", "item": "XG-100 Pro 空气净化器"},
    "06715421bjfab414": {"status": "待发货", "carrier": "—", "eta": "预计 24 小时内出库", "item": "XR-300 Plus 扫地机器人"},
}


# ---------------------------------------------------------------- 参数定义（同时用于校验与 schema）
class CreateOrderArgs(BaseModel):
    """用户明确要购买 / 下单某型号商品时，创建一笔新订单。"""

    sku: str = Field(description="商品型号，如 XR-200S、XR-300 Plus；用户没说就传空字符串，不要编造")
    quantity: int = Field(default=1, ge=1, le=10, description="购买数量，1~10 件")
    address: str = Field(default="", max_length=120, description="收货地址；用户没给就传空字符串，不要编造")


class QueryOrderArgs(BaseModel):
    """用户要查订单状态 / 物流进度，或只发来一个订单号时，查询订单。"""

    order_id: str = Field(default="", max_length=32, description="订单号；用户没给就传空字符串，不要编造")


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": CreateOrderArgs.__doc__.strip(),
            "parameters": CreateOrderArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": QueryOrderArgs.__doc__.strip(),
            "parameters": QueryOrderArgs.model_json_schema(),
        },
    },
]

DECIDE_PROMPT = """
你是电商客服的工具调度器。根据用户的话，自主判断是否需要调用业务工具。

规则：
1. 用户要买 / 下单 / 再买一台 → create_order；
2. 用户查订单、查物流、查发货进度，或只发来一个订单号 → query_order；
3. 咨询售后政策、型号参数、故障码，或闲聊 → 不调用工具；
4. 参数不全时仍然调用工具，缺失字段传空字符串，绝不编造型号、地址、订单号；
   工具会返回校验错误，由客服据此向用户追问；
5. 【会话已确认事实】里的型号、地址、订单号可以直接复用；
6. 一次只调用一个工具。
""".strip()

TOOL_ANSWER_PROMPT = """
你是「星域智能」的售后客服助手。系统刚替你调用了一个业务工具，结果见【工具结果】。

回答要求：
1. 只复述工具结果里的内容，不得编造订单号、金额、时效、物流信息；
2. 工具失败（ok=false）时，按 message 里的缺项向用户追问一句，不要自行补参数；
3. 成功时一两句话说明结果，并提示可继续查询订单；
4. 简洁专业，不要寒暄，不要输出 JSON 或思考过程。
""".strip()


class ToolError(Exception):
    """业务校验失败（可预期，直接转成给模型读的错误信息）。"""


# ---------------------------------------------------------------- 工具实现
def _create_order(args: dict, session_id: str, source: str) -> dict:
    p = CreateOrderArgs.model_validate(args)

    # 可售型号、价格、库存一律以 MySQL 为准，避免常量目录与真实库存不一致
    product = store.get_product(store.sku_key(p.sku))
    if not product:
        raise ToolError(f"型号「{p.sku or '未提供'}」不在可售目录，可选：{store.list_sku_names()}")

    address = p.address.strip()
    if len(address) < 6:
        raise ToolError("还缺少收货地址（需含省市区与详细地址），请向用户补充后再下单")

    sku = product["sku"]
    if p.quantity > product["stock"]:
        raise ToolError(f"{sku} 库存不足（剩余 {product['stock']} 件），请减少数量")

    # 幂等：人工重复点击或模型重复触发，直接返回已有订单，不重复扣库存
    dup = store.find_recent_order(session_id, sku, p.quantity, address)
    if dup:
        return {
            "data": dict(dup),
            "message": f"检测到相同参数的近期订单，未重复创建。订单号 {dup['order_id']}，"
                       f"{dup['item']} ×{dup['quantity']}，金额 ¥{dup['amount']}，状态 {dup['status']}。",
        }

    order_id = "SO" + datetime.now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4].upper()
    amount = product["price"] * p.quantity
    data = store.create_order(
        order_id=order_id, session_id=session_id, sku=sku, item=product["item"],
        quantity=p.quantity, amount=amount, address=address, status="待发货", source=source,
    )
    label = "人工购买" if source == "user" else "觅淘助手代买"
    return {
        "data": data,
        "message": f"下单成功（{label}）：订单号 {order_id}，{product['item']} ×{p.quantity}，"
                   f"金额 ¥{amount}，状态 待发货，收货地址 {address}。",
    }


def _query_order(args: dict, session_id: str, source: str) -> dict:
    p = QueryOrderArgs.model_validate(args)
    order_id = p.order_id.strip()
    if not order_id:
        raise ToolError("还缺少订单号，请向用户索取订单号后再查询")
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,32}", order_id):
        raise ToolError(f"订单号「{order_id}」格式不正确，应为 6~32 位字母数字组合")

    row = store.get_order(order_id.upper())
    if row:
        label = "人工购买" if row.get("source") == "user" else "觅淘助手代买"
        message = (f"订单 {order_id}：{row['item']} ×{row['quantity']}，金额 ¥{row['amount']}，"
                   f"状态 {row['status']}，来源 {label}，收货地址 {row['address']}，"
                   f"下单时间 {row['created_at']}。")
        return {"data": dict(row), "message": message}

    mock = MOCK_ORDERS.get(order_id.lower())
    if not mock:
        raise ToolError(f"未查询到订单 {order_id}，请核对订单号是否正确")
    data = {"order_id": order_id, **mock}
    message = (f"订单 {order_id}：{mock['item']}，当前状态 {mock['status']}，"
               f"承运商 {mock['carrier']}，预计送达 {mock['eta']}。")
    return {"data": data, "message": message}


_HANDLERS: dict[str, Callable[[dict, str, str], dict]] = {
    "create_order": _create_order,
    "query_order": _query_order,
}


def run_tool(name: str, args: Any, session_id: str, source: str = "agent") -> dict:
    """唯一执行入口：白名单 + 参数校验 + 业务校验 + 统一异常兜底。

    source=user 表示商城页人工购买，source=agent 表示 AI 助手代买；
    两者共用同一套校验、库存扣减与幂等规则，只写同一个 MySQL 订单表。
    """
    try:
        if name not in _HANDLERS:
            raise ToolError(f"未注册的工具「{name}」，可选：{list(_HANDLERS)}")
        if not isinstance(args, dict):
            raise ToolError("工具参数必须是 JSON 对象")
        return {"ok": True, "tool": name, "source": source,
                **_HANDLERS[name](args, session_id, source)}
    except ValidationError as exc:
        detail = "；".join(
            f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]
        )
        message = f"参数校验失败：{detail}。请向用户确认后重试，不要编造参数。"
        return {"ok": False, "tool": name, "error": "参数校验失败", "detail": detail, "message": message}
    except ToolError as exc:
        return {"ok": False, "tool": name, "error": "业务校验失败", "detail": str(exc), "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 —— 数据库等意外异常不阻断对话
        logger.exception("工具 %s 执行异常", name)
        return {
            "ok": False, "tool": name, "error": "工具执行异常",
            "detail": f"{exc.__class__.__name__}: {exc}",
            "message": "业务操作未完成，系统暂时不可用，请稍后重试或转人工。",
        }


# ---------------------------------------------------------------- 模型自主决策
def decide_tool_call(user_input: str, facts: dict | None = None) -> dict | None:
    """把工具 schema 交给模型，由模型决定是否调用。异常时降级到原链路。"""
    try:
        system = DECIDE_PROMPT + "\n\n【会话已确认事实】" + json.dumps(facts or {}, ensure_ascii=False)
        resp = get_tool_model().bind_tools(TOOL_SPECS).invoke(
            [SystemMessage(content=system), HumanMessage(content=user_input)]
        )
        calls = getattr(resp, "tool_calls", None) or []
        if not calls:
            return None
        call = calls[0]
        return {"name": str(call.get("name", "")), "args": call.get("args") or {}}
    except Exception as exc:  # noqa: BLE001 —— 模型不可用不影响原有问答
        logger.warning("工具决策失败，降级原链路：%s", exc)
        return None
