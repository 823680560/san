"""
FastAPI Web 主应用
提供文件上传、向量化构建、RAG 对话三大功能
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from web.api.upload_api import router as upload_router
from web.api.vectorize_api import router as vectorize_router
from web.api.chat_api import router as chat_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _init_models():
    """初始化 llama_index 全局设置，避免默认使用 OpenAI"""
    from llama_index.core import Settings

    if Settings._embed_model is None:
        from configs.config import EMBEDDING_MODEL, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_SERVICE, EMBEDDING_NORMALIZE, EMBEDDING_BATCH_SIZE, OLLAMA_BASE_URL, resolve_model_path
        try:
            if EMBEDDING_SERVICE == "ollama":
                from llama_index.embeddings.ollama import OllamaEmbedding
                Settings.embed_model = OllamaEmbedding(
                    model_name=EMBEDDING_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    normalize=EMBEDDING_NORMALIZE,
                    embed_batch_size=EMBEDDING_BATCH_SIZE,
                )
                logger.info(f"已初始化 Ollama 嵌入模型: {EMBEDDING_MODEL}")
            elif EMBEDDING_SERVICE == "huggingface":
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                model_path = resolve_model_path(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)
                Settings.embed_model = HuggingFaceEmbedding(model_name=model_path, embed_batch_size=EMBEDDING_BATCH_SIZE)
                logger.info(f"已初始化 HuggingFace 嵌入模型: {model_path}")
            else:
                logger.warning(f"未知的嵌入模型服务: {EMBEDDING_SERVICE}，跳过初始化")
        except Exception as e:
            logger.warning(f"嵌入模型初始化失败（首次请求时将自动重试）: {e}")

    if Settings._llm is None:
        from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
        from langchain_openai import ChatOpenAI
        from llama_index.llms.langchain import LangChainLLM
        try:
            langchain_llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=0.7,
                request_timeout=120.0,
            )
            Settings.llm = LangChainLLM(llm=langchain_llm)
            logger.info(f"已初始化 LLM: {LLM_MODEL}")
        except Exception as e:
            logger.warning(f"LLM 初始化失败（首次请求时将自动重试）: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_models()
    import threading
    threading.Thread(target=_warmup_agent, daemon=True).start()
    yield


def _warmup_agent():
    """后台预热 Agent，避免首次 auto 请求等待"""
    try:
        from server.agent.agent_factory import init_agent
        init_agent()
    except Exception as e:
        logger.warning(f"Agent 预热失败（首次 auto 请求时会重试）: {e}")


app = FastAPI(
    title="环卫行业智能问答系统",
    description="RAG 知识库管理与对话系统",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 模板（禁用缓存，避免 reload 时的 unhashable type 错误）
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)
templates.env.cache = None

# 注册路由
app.include_router(upload_router, prefix="/api")
app.include_router(vectorize_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """文件上传页面"""
    return templates.TemplateResponse(request, "upload.html")


@app.get("/vectorize", response_class=HTMLResponse)
async def vectorize_page(request: Request):
    """向量化构建页面"""
    return templates.TemplateResponse(request, "vectorize.html")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """RAG 对话页面"""
    return templates.TemplateResponse(request, "chat.html")


if __name__ == "__main__":
    import uvicorn
    # 不用 reload 模式，避免热重载导致 Agent 记忆丢失
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
