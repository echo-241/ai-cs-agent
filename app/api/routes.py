"""HTTP 接口。"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agent import shop_agent
from app.agent.orchestrator import AgentOrchestrator
from app.agent.tools import decide_tool_call, run_tool
from app.api.schemas import AgentBuyRequest, ChatRequest, SessionCreate, ShopChatRequest, ShopOrderRequest
from app.common import db, shop_db
from app.config.settings import UPLOAD_DIR, settings
from app.retrieval.pipeline import MODES
from app.runtime import get_bm25, get_pipeline, get_retriever, get_store

router = APIRouter(prefix="/api")
_orchestrator: AgentOrchestrator | None = None


def orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(get_retriever())
    return _orchestrator


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "collection": settings.collection,
        "chunks": len(get_bm25()),
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "rerank_model": settings.rerank_model,
    }


# ---------------------------------------------------------------- 文档
@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise HTTPException(400, f"仅支持 pdf / md / txt，收到 {suffix or '未知'}")

    target = UPLOAD_DIR / (file.filename or f"upload{suffix}")
    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        stats = get_pipeline().ingest_file(target)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"入库失败：{exc}") from exc
    return stats


@router.get("/documents")
def list_documents() -> dict:
    docs = db.list_documents()
    return {"total": len(docs), "items": docs, "vectors": len(get_bm25())}


def _resolve_doc_id(raw: str) -> str:
    """允许用短 id（doc_id 结尾的 8 位哈希）访问，避免 URL 里出现中文。

    例如完整 doc_id 是「02_故障错误码手册-f5a99445」，
    传「f5a99445」也能定位到同一份文档。
    """
    doc_ids = [d["doc_id"] for d in db.list_documents()]
    if raw in doc_ids:
        return raw
    matches = [d for d in doc_ids if d.endswith(raw)]
    return matches[0] if len(matches) == 1 else raw


@router.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, offset: int = 0, limit: int = 20) -> dict:
    """按 doc_id 翻看入库后的分块，用于核对切分质量。

    Milvus 的 query 不保证顺序，这里一次取回该文档全部分块后按 chunk_index 排序再切片，
    避免分页错乱（单文档分块数在千级以内，开销可忽略）。
    """
    resolved = _resolve_doc_id(doc_id)
    store = get_store()
    rows = store.client.query(
        store.collection,
        filter=f'doc_id == "{resolved}"',
        output_fields=["chunk_index", "page", "text"],
        limit=1000,
    )
    if not rows:
        raise HTTPException(404, f"未找到 doc_id={doc_id}，可先访问 /api/documents 查看清单")

    rows.sort(key=lambda r: r["chunk_index"])
    page = rows[offset:offset + limit]
    return {
        "doc_id": resolved,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "items": [
            {
                "chunk_index": r["chunk_index"],
                "page": r["page"],
                "chars": len(r["text"]),
                "text": r["text"],
            }
            for r in page
        ],
    }


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    get_store().delete_doc(doc_id)
    get_bm25().remove_doc(doc_id)
    get_bm25().save()
    with db.connect() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    return {"deleted": doc_id, "vectors": get_store().count()}


# ---------------------------------------------------------------- 会话
@router.post("/sessions")
def create_session(payload: SessionCreate) -> dict:
    session_id = uuid.uuid4().hex[:16]
    db.ensure_session(session_id, payload.user_id, payload.title)
    return {"session_id": session_id, "user_id": payload.user_id, "title": payload.title}


@router.get("/sessions")
def sessions(user_id: str = "demo") -> dict:
    return {"items": db.list_sessions(user_id)}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    db.delete_session(session_id)
    return {"deleted": session_id}


@router.get("/sessions/{session_id}/messages")
def session_messages(session_id: str) -> dict:
    return {"items": db.recent_messages(session_id, 100)}


# ---------------------------------------------------------------- 对话
@router.post("/chat")
def chat(payload: ChatRequest) -> StreamingResponse:
    if payload.mode not in MODES:
        raise HTTPException(400, f"未知检索配置 {payload.mode}，可选 {MODES}")
    db.ensure_session(payload.session_id, "demo", "会话")

    def event_stream():
        try:
            for event in orchestrator().stream(payload.session_id, payload.message, payload.mode):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            err = {"type": "error", "message": f"{exc.__class__.__name__}: {exc}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- 商城（人工购买 / 助手代买）
@router.get("/shop/products")
def shop_products() -> dict:
    try:
        items = shop_db.list_products()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"商品读取失败：{exc}") from exc
    meta = getattr(shop_db, "PRODUCT_META", {})
    for item in items:
        extra = meta.get(item["sku"], {})
        item["image"] = extra.get("image", "")
        item["tag"] = extra.get("tag", "")
        item["desc"] = extra.get("desc", "")
    return {"items": items}


@router.get("/shop/orders")
def shop_orders(session_id: str = "demo", limit: int = 20) -> dict:
    try:
        return {"items": shop_db.list_orders(session_id, min(limit, 100))}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"订单读取失败：{exc}") from exc


@router.post("/shop/orders")
def shop_create_order(payload: ShopOrderRequest) -> dict:
    """人工点「立即购买」：与 AI 代买走同一个 run_tool，只是 source=user。"""
    args = {"sku": payload.sku, "quantity": payload.quantity, "address": payload.address}
    out = run_tool("create_order", args, payload.session_id, source="user")
    if not out["ok"]:
        raise HTTPException(400, out["message"])
    return out


@router.post("/shop/chat")
def shop_chat(payload: ShopChatRequest) -> dict:
    """商城 AI 导购：多轮对话，聊天与下单都在这一个入口里完成。"""
    return shop_agent.chat(payload.session_id, payload.message, payload.address)


@router.get("/shop/chat/history")
def shop_chat_history(session_id: str = "demo", limit: int = 20) -> dict:
    """刷新页面后恢复聊天记录（多轮记忆落 SQLite）。"""
    return {"items": db.recent_messages(session_id, min(limit, 50))}


@router.post("/shop/agent-buy")
def shop_agent_buy(payload: AgentBuyRequest) -> dict:
    """觅淘助手代买：LLM 依据 function schema 自主决定调用哪个工具、填什么参数。"""
    facts = {"address": payload.address} if payload.address else {}
    call = decide_tool_call(payload.message, facts)
    if not call:
        return {
            "ok": False, "call": None,
            "message": "觅淘助手暂时没能识别下单/查单意图（需要可用的模型服务），"
                       "请把型号和地址说清楚，或改用「立即购买」。",
        }
    out = run_tool(call["name"], call["args"], payload.session_id, source="agent")
    return {"ok": out["ok"], "call": call, "outcome": out}


# ---------------------------------------------------------------- 检索调试
@router.get("/retrieve")
def retrieve(q: str, mode: str = "hybrid_rerank", k: int = 4) -> dict:
    hits, debug = get_retriever().search(q, mode=mode, final_k=k, with_debug=True)
    return {
        "query": q,
        "mode": mode,
        "debug": debug,
        "hits": [
            {
                "n": i,
                "doc": h.get("doc_name"),
                "page": h.get("page"),
                "heading": h.get("heading"),
                "dense_rank": h.get("dense_rank"),
                "sparse_rank": h.get("sparse_rank"),
                "rrf_score": round(h.get("rrf_score", 0.0), 5),
                "score": round(float(h.get("rerank_score", h.get("score", 0.0)) or 0.0), 4),
                "text": h["text"][:160],
            }
            for i, h in enumerate(hits, 1)
        ],
    }


@router.get("/stats")
def stats() -> dict:
    return {"vectors": get_store().count(), "bm25": len(get_bm25())}
