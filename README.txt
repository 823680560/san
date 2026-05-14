================================================================================
                    环卫行业智能问答系统
           Sanitation Industry Intelligent Q&A System
================================================================================

一、项目简介
--------------------------------------------------------------------------------
  本系统是一个面向环卫行业的智能问答平台，基于 RAG（检索增强生成）和 NL2SQL
  技术，实现对环卫项目数据和法律法规资料的智能检索与分析回答。

  核心能力：
  - 法律法规检索问答：上传 PDF/JSON 格式法规文件，自动构建向量索引，支持
    自然语言检索（如"政府采购法对供应商的资格要求是什么？"）
  - 环卫项目数据查询：上传 Excel 项目数据，自动导入 SQLite 数据库，支持
    自然语言转 SQL 查询（如"广东省合同金额最高的10个项目是哪些？"）
  - 自动路由对话：用户无需关心查哪个库，Agent 自动判断意图并调用对应工具


二、技术栈
--------------------------------------------------------------------------------
  Web 框架         FastAPI + Uvicorn + Jinja2
  Agent 框架       LangChain (ReAct Agent, ZERO_SHOT_REACT_DESCRIPTION)
  嵌入模型         BGE-M3（1024 维，HuggingFace 本地加载）
  精排模型         BGE-Reranker-v2-m3（CrossEncoder）
  向量库           FAISS (IndexFlatIP)
  全文检索         BM25s + jieba 分词
  NL2SQL           Vanna + SQLGlot（SQLite 精确查询 + DuckDB 聚合分析）
  数据库           SQLite + FTS5 全文索引
  LLM              DeepSeek API / Ollama 本地模型（Qwen2.5 / GLM-4 等）
  Docker 部署      纯 CPU 环境，16核32G 推荐


三、目录结构
--------------------------------------------------------------------------------
  san/
  ├── configs/                  # 全局配置
  │   ├── config.py             #   路径、模型参数、检索 top_k、chunk 参数
  │   └── prompt_config.py      #   Agent/评估/NL2SQL 所有提示词
  ├── data/
  │   ├── law/                  #   法律法规文件（PDF/JSON）
  │   └── project/              #   环卫项目 Excel + SQLite 数据库
  ├── knowledge_base/           #   FAISS 向量索引（运行时自动生成）
  ├── models/                   #   本地模型权重（需提前下载）
  │   ├── bge-m3/               #   嵌入模型 ~2.2GB
  │   └── bge-reranker-v2-m3/   #   精排模型 ~2.2GB
  ├── server/
  │   ├── agent/                #   ReAct Agent（工厂模式 + 线程安全）
  │   │   ├── agent_factory.py
  │   │   └── tools/            #   law_search / project_search_db / web_search
  │   ├── embeddings/           #   嵌入模型加载
  │   └── rag/                  #   RAG 核心
  │       ├── data_loader.py    #     多格式解析（8种）：Excel/JSON/PDF/CSV等
  │       ├── jieba_bm25.py     #     中文 BM25 检索器
  │       ├── vector_store.py   #     FAISS 索引构建/加载/持久化
  │       ├── reranker.py       #     CrossEncoder 重排序
  │       ├── project_db.py     #     SQLite + FTS5 + jieba 分词导入（多进程）
  │       ├── project_query_engine.py  # NL2SQL 引擎（分类+生成+校验+执行）
  │       ├── vanna_setup.py    #     Vanna 训练数据（DDL+文档+140+问答对）
  │       ├── vanna_llm.py      #     Vanna LLM 适配器
  │       └── sql_validator.py  #     SQLGlot 安全校验
  ├── web/
  │   ├── app.py                #   FastAPI 应用入口 + lifespan
  │   ├── api/                  #   upload / vectorize / chat 三个 API 模块
  │   ├── templates/            #   Jinja2 页面（首页/上传/向量化/对话）
  │   └── static/               #   静态资源
  ├── eval/                     # 评测系统
  │   ├── main_evaluation.py    #   主评测入口
  │   ├── sql_evaluator.py      #   NL2SQL 评测（VE/EX/CM/SIM 四维指标）
  │   ├── retrieval_evaluator.py #  检索评测
  │   ├── generation_evaluator.py # 生成质量评测（LLM as Judge）
  │   └── datasets/             #   评测数据集
  ├── docker/                   # Docker 部署
  │   ├── Dockerfile
  │   ├── docker-compose.yml
  │   ├── docker-compose.gpu.yml
  │   ├── requirements.txt
  │   └── .env.example
  └── main.py                   # CLI 命令行入口


四、快速开始
--------------------------------------------------------------------------------

  1. 下载模型（首次）
     --------------------------------------------------
     cd san/
     pip install huggingface_hub
     export HF_ENDPOINT=https://hf-mirror.com   # 国内加速
     hf download BAAI/bge-m3 --local-dir models/bge-m3
     hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
     # 每个模型约 2.2GB，支持断点续传

  2. 准备数据
     --------------------------------------------------
     将以下文件放入对应目录：
     - data/law/招采政策法规.json       # 法律法规 JSON
     - data/law/*.pdf                    # 法律法规 PDF（可选）
     - data/project/环卫项目数据.xlsx    # 环卫项目 Excel

  3. 配置环境变量
     --------------------------------------------------
     cd docker/
     cp .env.example .env
     编辑 .env，填入 DeepSeek API Key（或其他兼容 OpenAI API 的服务）：
       LLM_API_KEY=sk-your-key-here

  4. 启动服务
     --------------------------------------------------
     docker compose up -d
     访问 http://服务器IP:8000

     首次启动时间较长（安装配置 + 搭建环境 + 数据导入 + 向量索引构建）

  5. 使用
     --------------------------------------------------
     页面                        功能
     /              首页，可跳转到各功能页
     /upload        上传数据文件（支持 Excel/PDF/JSON/CSV/TXT/DOCX）
     /vectorize     对已上传文件构建 FAISS 向量索引
     /chat          选择知识库开始对话（auto=自动路由）

  6. Ollama 本地模型（可选，默认用在线 API）
     --------------------------------------------------
     docker compose --profile ollama up -d
     docker exec san-ollama ollama pull qwen2.5:7b
     修改 .env 中 LLM 配置指向 Ollama


五、数据流概览
--------------------------------------------------------------------------------

  流程 1：项目数据 Excel → SQLite → NL2SQL
  ─────────────────────────────────────────
    Excel → pandas 读取 → 列名映射 → SQLite 写入 → jieba 分词 → FTS5 索引
    → Vanna 训练（DDL + 文档 + 问答对）
    查询时：用户问题 → 分类器（exact/fuzzy/aggregation）→ Vanna 生成 SQL
    → SQLGlot 校验 → SQLite/DuckDB 执行 → 结果格式化 → LLM 生成回答

  流程 2：法律法规 JSON/PDF → FAISS → RAG
  ─────────────────────────────────────────
    JSON/PDF → DataLoader 解析 → SentenceSplitter 分片（800/100）
    → BGE-M3 向量化 → FAISS IndexFlatIP → 持久化
    查询时：向量检索(top30) + BM25(top60) → QueryFusion(60) → Reranker(5)
    → LLM 生成回答

  流程 3：对话 Agent 自动路由
  ─────────────────────────────────────────
    用户提问 → ReAct Agent 判断意图 → 调用对应工具
    - 法律问题 → law_search（RAG 检索）
    - 项目查询 → project_search_db（NL2SQL）
    - 其他   → search_internet（联网搜索）
    → LLM 综合工具返回结果生成最终回答


六、关键配置参数
--------------------------------------------------------------------------------

  参数                    默认值       说明
  EMBEDDING_BATCH_SIZE    128          嵌入向量化批大小（环境变量可覆盖）
  EMBEDDING_DIM           1024         BGE-M3 向量维度
  LAW_CHUNK_SIZE          800          法律文本分片大小
  LAW_VECTOR_TOP_K        30           向量检索候选数
  LAW_BM25_TOP_K          60           BM25 检索候选数
  LAW_FUSION_TOP_K        60           Fusion 向 Reranker 传递数
  LAW_RERANKER_TOP_N      5            Reranker 精排输出数
  LAW_RETRIEVER_WEIGHTS   [0.2, 0.8]   融合权重 [BM25, Vector]


七、性能参考（16核32G 纯 CPU 环境）
--------------------------------------------------------------------------------

  场景                              耗时
  Docker 镜像构建（首次含依赖下载）  3-5 分钟，看资源拉取速度计硬件配置情况可能出现较大浮动
  容器启动（增量，索引已存在）       30-60 秒
  容器启动（全量重建，含向量化）     14-22 分钟，与数据量相关
  单次对话（Agent 模式）             5-20 秒
  NL2SQL 单次查询                    3-8 秒


八、评测系统
--------------------------------------------------------------------------------

  cd eval/
  python main_evaluation.py

  支持四种评测模式：
  - NL2SQL 评测：VE（结果正确）/ EX（SQL精确匹配）/ CM（列匹配）/ SIM（综合）
  - 检索评测：Recall@k / Precision@k / MRR
  - 生成评测：Correctness / Relevancy / Faithfulness / Recall（LLM as Judge）
  - 端到端评测：完整对话流程评估


九、常见问题
--------------------------------------------------------------------------------

  Q: 启动后访问页面报 500 错误？
  A: 检查 .env 中 LLM_API_KEY 是否正确配置，首次启动需等待模型加载完毕。

  Q: 向量化很慢？
  A: 纯 CPU 环境 BGE-M3 嵌入是瓶颈。可增大 EMBEDDING_BATCH_SIZE 到 256
     （32G 内存），或设置环境变量 EMBEDDING_BATCH_SIZE=256。

  Q: FTS5 全文搜索不准确？
  A: 因为 jieba 分词后使用了前缀匹配（'关键词*'），若分词结果与搜索词不一致，
     可尝试调整 query 用词，或在数据库中确认 FTS5 索引已正确构建。

  Q: 如何切换到其他 LLM？
  A: 修改 .env 中的 LLM_MODEL / LLM_API_KEY / LLM_BASE_URL 为任意兼容
     OpenAI API 的服务地址。Ollama 本地模型也支持，见 DEPLOY.md。

  Q: 数据更新后如何重建？
  A: 删除 knowledge_base/law/*.json 和 data/project/sanitation_projects.db
     及 .hash 文件后重启容器即可自动重建。


十、相关文档
--------------------------------------------------------------------------------
  DEPLOY.md              详细部署文档（GPU/CPU/Docker/Ollama）
  技术架构与数据流.md     系统架构、数据流、目录结构全解
  eval/评估操作命令.md     评测系统使用指南
