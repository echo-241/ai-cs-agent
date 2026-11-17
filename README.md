# 星域智能 · 售后客服 Agent

电商售后场景的智能客服系统：**意图识别 + 三层记忆 + 混合检索（BM25 + 稠密 + RRF + 重排）**，
FastAPI 提供接口，Vue 3 单页做问答界面，全链路可用一条 pytest 跑通。
另带「觅淘」商城页（`/shop`）：既能自己下单，也能让 AI 助手代买，两种路径共用 MySQL 订单表。

本项目在作者原有两份代码基础上重构整合：

| 来源 | 保留的部分 | 改动 |
|---|---|---|
| `AI-Project02`（RAG 问答） | 离线建库、查询改写、重排、引用生成的分层结构 | 切分改为结构化；向量库由 Chroma 换为 Milvus；新增 BM25 与 RRF |
| `LangGraph/ChatService`（意图与记忆） | 意图识别、事实槽、增量摘要的三层记忆设计 | 存储改为 SQLite；修复意图不传历史的缺陷；新增 doc_qa 意图与路由 |

---

## 一、它解决什么问题

电商售后知识库有三类典型问题：

1. **编号类**：错误码 `E1028`、型号 `XR-200S`、退款码 `R002`、物流码 `LW05`——
   外形高度相似，用户只报编号，不描述含义；
2. **政策条款类**：「七天无理由的运费谁出」「换货后保修是否重算」——散落在多份文档；
3. **多轮追问**：用户先说「我要退货」，被追问后才补「订单号是 xxx」。

系统对应做三件事：混合检索解决 1、2 的召回；意图路由 + 三层记忆解决 3 的上下文延续。

---

## 二、架构

```
                      ┌─────────────────────────────────────────┐
  用户 ── HTTP/SSE ──▶│ FastAPI  app/api/routes.py              │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │ AgentOrchestrator                       │
                      │  ① IntentRecognizer（结构化输出 + 白名单）│
                      │  ② Memory（事实槽 / 摘要 / 最近消息）     │
                      │  ③ 路由：doc_qa / order_query / ...      │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │ Retriever（五档可切换）                  │
                      │  dense ─┐                               │
                      │         ├─▶ RRF(k=60) ─▶ qwen3-rerank   │
                      │  BM25  ─┘                               │
                      └───────┬──────────────────┬──────────────┘
                              ▼                  ▼
                    Milvus（204 稠密向量）   BM25 内存索引（jieba）
                              │                  │
                              └──────┬───────────┘
                                     ▼
                          离线建库 IngestionPipeline
                          PDF → 结构化切分 → 向量化 → 入库
```

### 关键设计

**1. 结构化切分，而不是整页递归切分**

原实现直接对每页文本做 `RecursiveCharacterTextSplitter`，PDF 换行会把标题甩到上一块末尾，
出现「标题在 A 块、正文在 B 块」。现在先按标题把文档切成小节，再对超长小节按 200 字/20 重叠细分，
并把标题回填到每个子块开头。

粒度没变，只修正了边界。效果：检索命中的块自带「2.2.3 E1028 充电电流异常」这样的上下文，
既提升召回，也让回答可以精确引用到条款。

**2. 单 collection + doc_id 过滤**

原实现一个文件一个 Chroma collection（collection 名即文件 MD5），只能查单个文档。
改为 Milvus 单 collection + `doc_id` 标量过滤，支持多文档共存与「限定某几份文档检索」。

**3. 手写 RRF，而不是用库内置的 hybrid**

`score(d) = Σ 1/(k + rank_i(d))`，k=60。稠密返回余弦相似度、BM25 返回无上界词频分，
量纲不可比，直接加权求和必须先归一化；RRF 只用名次，对量纲免疫。

手写而不是调用 Milvus 内置 hybrid 的原因：内置版本把融合藏进库里，
答辩被问「融合代码在哪」「k 改成 10 会怎样」时答不上来。

**4. BM25 用 jieba + rank_bm25，而不是 Milvus 内置 BM25**

内置 BM25 的 k1/b 不可调、分词器是黑盒，且无法单独跑「纯 BM25」这一组配置做对比。

分词时**先用正则把编号整体切出来**（`E1024`、`XR-200`、`RF20261108000123`），
再对剩余中文做 jieba，编号内的连字符去掉后小写归一——
否则 `XR-200` 会被切成 `XR`、`-`、`200`，BM25 的精确匹配优势直接消失。

**5. Function Calling：下单 / 订单查询工具**

`app/agent/tools.py` 用 Pydantic 定义 `CreateOrderArgs`、`QueryOrderArgs`，
同一份模型既生成工具的 JSON Schema（交给 LLM），又负责运行时参数校验，不会出现「schema 与校验不一致」。
意图识别只判断要不要进业务分支；**调哪个工具、参数怎么填、校验失败后追问什么，全部由 LLM 依据 schema 自主决定**。

```
LLM(bind_tools) → tool_calls → run_tool（白名单 / Pydantic 校验 / 型号归一化 / 库存 / 幂等）
                → {ok, message, data} 回灌 LLM → SSE 流式答复
```

工具层不向主链路抛异常：参数错、库存不足、订单不存在、数据库异常都转成结构化 `ok=false`，
交给模型据此追问；模型未调用工具或调用失败时降级到原有订单查询逻辑，问答链路不中断。
订单写入 SQLite `orders` 表，5 分钟内同参数重复触发直接返回原订单，避免模型重复调用造成重复下单。
前端把 `tool` / `tool_result` 两个 SSE 事件显示成「工具 create_order → 成功」，便于演示调用过程。

**6. 意图识别必须带历史**

原 `ChatService.handle()` 调用 `self.intent.recognize(req.user_input)`，
丢掉了 `IntentRecognizer` 本来支持的 `chat_history` 参数。
后果是「我想退货」→「订单号是 xxx」这类跨轮槽位填充必然失败。
现在把最近 6 条对话摘要传入，并把识别出的槽位写进事实槽，下一轮可直接继承。

**7. 商城：电商化页面 + 多轮 AI 导购**

`GET /shop` 是商城页：顶部导航 / 搜索 / Banner、商品大图卡片、订单表带来源标签。
白底商品图在 `app/static/images/`，由图像模型生成（**演示素材，非品牌实拍**）；
图片路径、角标、卖点集中在 `shop_db.PRODUCT_META`，卖点参数取自知识库《商品型号与规格手册》。
商品、库存、订单落本机 MySQL（`mitao_shop`，配置在 `.env`），`shop_db.py` 首次调用自动建库建表。

右侧是**真的多轮聊天**，不是一次性「说买什么」的输入框。`app/agent/shop_agent.py` 每轮把
最近 10 条对话 + 在售商品 + 默认地址 + 该会话最近订单注入 system prompt，用 `bind_tools`
让 LLM 自己决定是聊天、推荐、对比，还是调用 `create_order` / `query_order`；工具结果回灌后
再生成自然语言回复。对话写进 SQLite，刷新页面可恢复（`/api/shop/chat/history`）。

```
人工点「立即购买」 ─▶ /api/shop/orders ─────┐
                                             ├─▶ run_tool(...) ─▶ MySQL 事务扣库存 + 写订单
对话里说「帮我买」 ─▶ /api/shop/chat ───────┘   ▲
                       │  └─ 多轮历史 + 商品上下文 + bind_tools → tool_calls
                       └─ 不调工具时就是普通聊天 / 导购
```

两条路径只是 `source=user / agent` 的区别，参数校验、型号归一化、库存校验、幂等、
异常兜底全部复用 `app/agent/tools.py` 同一份代码；用 `WHERE stock>=quantity` 防超卖。
聊天气泡里直接展示工具名、参数与结果，便于演示 Function Calling。
实测（真实模型）：「预算 2000 买哪款」→ 推荐不调工具；「帮我买一台 XR-200S」→ 自主下单；
「刚才那单多少钱」→ 自主调 `query_order` 并带上历史订单号回答。

**数据库选型：MySQL 而不是 Postgres。** 本机已有 MySQL 8.4，向量检索由 Milvus 承担，
本项目用不到 pgvector / JSONB；换 Postgres 只会多一个要安装和维护的服务。

---

## 三、快速开始

### 1. 起 Milvus

```bash
cd deploy
docker compose up -d          # etcd / minio / milvus / attu
```

> ⚠️ MinIO 官方镜像自 2026 年 9 月起已下架，直接 `docker compose up` 会卡在拉取 minio。
> 用官方推荐的平替镜像打标即可（入口命令与环境变量完全一致，compose 无需修改）：
>
> ```bash
> docker pull docker.1ms.run/pgsty/minio:latest
> docker tag docker.1ms.run/pgsty/minio:latest minio/minio:RELEASE.2024-04-18T19-09-19Z
> ```

### 2. 装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

```bash
cp .env.example .env          # 填入 DASHSCOPE_API_KEY；商城另需 MYSQL_USER / MYSQL_PASSWORD
```
本机已按 `.env.example` 的 MySQL 项配好 `.env`（该文件被 `.gitignore` 忽略，不会提交密码）。
MySQL 需为 8.x 且允许 `127.0.0.1:3306` 连接；首次访问商城接口会自动建 `mitao_shop` 库和两张表。

### 4. 生成语料并建库

```bash
python scripts/gen_corpus.py  # 生成 8 份 PDF 语料到 data/docs
python scripts/build_index.py # 切分 → 向量化 → 入 Milvus + BM25
```

### 5. 启动

```bash
python -m app.main
# 客服页   http://127.0.0.1:8080/
# 商城页   http://127.0.0.1:8080/shop   （客服页左下角也有入口）
# 接口文档 http://127.0.0.1:8080/docs
```

### 6. 验收测试

```bash
pytest tests/test_e2e.py -s
```

覆盖：上传文档 → 普通问答 → 文档问答 → 多轮追问，共 6 个用例。

新增 `pytest tests/test_tools.py tests/test_shop.py tests/test_shop_chat.py -q`：工具 schema、
模型自主决策、参数校验、异常兜底、SSE 编排、商城页面、人工/助手两种下单路径，
以及多轮对话上下文与对话内下单。用 `tests/conftest.py` 的内存存储替换 MySQL，
不连数据库、不打真实模型。

---

## 四、目录结构

```
ai-cs-agent/
├── app/
│   ├── main.py                 FastAPI 入口
│   ├── runtime.py              进程内单例（向量库 / BM25 / 检索器）
│   ├── config/settings.py      全部配置项
│   ├── common/
│   │   ├── clients.py          embedding / chat 客户端与批量重试
│   │   ├── db.py               SQLite 持久层（文档/会话/消息/事实/摘要/工单）
│   │   └── shop_db.py          MySQL 商城持久层（商品/库存/订单，人工与 AI 共用）
│   ├── indexing/
│   │   ├── loader.py           PDF/MD 加载 + 结构化切分
│   │   ├── bm25_index.py       jieba 分词 + rank_bm25 + 持久化
│   │   ├── milvus_store.py     Milvus 单 collection + doc_id 过滤
│   │   └── pipeline.py         离线建库编排
│   ├── retrieval/
│   │   ├── fusion.py           手写 RRF
│   │   ├── rerank.py           qwen3-rerank 交叉编码重排
│   │   └── pipeline.py         五档配置切换
│   ├── agent/
│   │   ├── intent.py           意图识别（结构化输出 + 白名单兜底）
│   │   ├── memory.py           三层记忆
│   │   ├── tools.py            下单 / 订单查询工具（Function Calling + 校验兜底）
│   │   ├── shop_agent.py       商城多轮 AI 导购（历史 + 商品上下文 + 工具回灌）
│   │   └── orchestrator.py     路由与流式生成
│   ├── api/                    接口层（含 /api/shop/* 商城接口）
│   └── static/
│       ├── index.html          Vue 3 单页问答
│       ├── shop.html           Vue 3 商城页（电商视觉 + 多轮 AI 导购）
│       └── images/             4 张白底商品演示图（图像模型生成）
├── scripts/                    语料生成 / 建库
├── eval/                       评测集与复测脚本
└── tests/
    ├── conftest.py             内存商城存储夹具（测试不依赖 MySQL）
    ├── test_e2e.py             上传→问答→追问全链路验收
    ├── test_tools.py           Function Calling 工具层
    ├── test_shop.py            商城页面与两种下单路径
    └── test_shop_chat.py       多轮对话、上下文、对话内下单
```

---

## 五、评测方法（可复现）

**语料**：8 份 PDF，204 个 chunk，约 3 万字。其中《故障代码全表》60 条错误码外形高度相似
（E1024 / E1025 / E1028 …），构成天然的困难负样本。

**题目**：60 道，按真实提问分布分四类——编号类 35、术语类 12、口语化 11、跨节 2。

**gold 标注**：不手工标注。每题给出「能回答该问题的原文片段」，用片段在 chunk 里做子串匹配反查，
一条事实命中多个 chunk 时全部计入（多相关块判定）。标注零主观，可完全复现。

**配置**：五组，同一批 chunk、同一 embedding、同一 rerank 模型，只改检索组合方式。

| 配置 | 含义 |
|---|---|
| `dense` | 纯向量（朴素基线） |
| `dense_rerank` | 向量 + 重排（改造前系统的真实形态） |
| `sparse` | 纯 BM25 |
| `hybrid` | BM25 + 向量 + RRF |
| `hybrid_rerank` | BM25 + 向量 + RRF + 重排（改造后） |

**命令**：

```bash
python eval/build_eval.py     # 反查 gold，生成 questions.jsonl
python eval/run_eval.py       # 五组配置的 Recall 复测
python eval/e2e_eval.py       # 端到端准确率（生成 + LLM 判定）
```

评测结果见 `eval/REPORT.md`。

---

## 六、已知局限

1. **语料规模小**：204 个 chunk，真实企业知识库在千级以上，Recall@5 的天花板效应明显；
2. **单轮评测**：评测集是单轮问答，多轮对话只在 e2e 测试里覆盖，未做量化；
3. **订单数据是演示数据**：`MOCK_ORDERS` 仅三条，商品价格与库存为 mock，没有支付与真实物流；
   商品图是图像模型生成的演示素材，不是品牌实拍；AI 导购依赖模型服务可用；
   新订单真实落在本地 MySQL `mitao_shop.orders`（`source` 区分人工/助手），
   SQLite 里旧的 `orders` 表保留但不再写入；幂等靠 5 分钟同参数去重，真实项目应改用客户端幂等键；
4. **BM25 索引在进程内**：万级以上 chunk 需要换 Elasticsearch 或 Milvus 内置 BM25；
5. **判定员与被测模型同源**：端到端评测用同一个 chat 模型做判定，存在同源偏好，
   已把全部原始答案落盘供人工复核。
