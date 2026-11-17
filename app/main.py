"""FastAPI 入口。

启动：python -m app.main
文档：http://127.0.0.1:8080/docs
页面：http://127.0.0.1:8080/
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config.settings import PROJECT_ROOT
from app.runtime import init_runtime

STATIC_DIR = PROJECT_ROOT / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_runtime()
    yield


app = FastAPI(
    title="星域智能 售后客服 Agent",
    description="意图识别 + 三层记忆 + 混合检索（BM25 + 稠密 + RRF + 重排）",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/chunks")
def chunks_page():
    """分块浏览页：左侧选文档，右侧按顺序看每个 chunk 的原文。"""
    return FileResponse(str(STATIC_DIR / "chunks.html"))


@app.get("/shop")
def shop_page():
    """觅淘商城：人工购买 + 觅淘助手代买，共用 MySQL 订单表。"""
    return FileResponse(str(STATIC_DIR / "shop.html"))


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
