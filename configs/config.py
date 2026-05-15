import os
from dotenv import load_dotenv

# 获取项目根目录（configs 的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载项目根目录下的 .env 文件
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# HuggingFace 镜像端点（国内部署默认使用 hf-mirror.com，设为空字符串则直连 HF 官方）
_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
if _HF_ENDPOINT and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = _HF_ENDPOINT

# 数据路径（使用绝对路径）,须将文件名换成自己的文件名
PROJECT_EXCEL = os.path.join(PROJECT_ROOT, "data", "project", "环卫项目数据.xlsx") 
LAW_JSON = os.path.join(PROJECT_ROOT, "data", "law", "招采政策法规.json")


# LLM 配置（支持环境变量覆盖）
# Ollama 本地模型示例：
# LLM_MODEL = "glm-4-9b-chat:latest"
# LLM_API_KEY = "ollama"
# LLM_BASE_URL = "http://localhost:11434/v1"

# LLM_MODEL = "Llama-3.1-8B-Instruct"
# LLM_API_KEY = "ollama"
# LLM_BASE_URL = "http://localhost:11434/v1"

# 流式输出开关
STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "true").lower() == "true"

# DeepSeek API 示例：
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-xxx")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")


# 评估打分专用模型（可与生成模型不同，用于 LLM as Judge）
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "deepseek-chat")
EVAL_LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY", "sk-xxx")
EVAL_LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", "https://api.deepseek.com/v1")


# 嵌入模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_SERVICE = os.getenv("EMBEDDING_SERVICE", "huggingface")  # 可选: "ollama", "huggingface"
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "bge-m3"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")  # HF 模型 ID，本地不存在时自动下载
EMBEDDING_NORMALIZE = True
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
EMBEDDING_DIM = 1024  # bge-m3 维度
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# 向量库路径（使用绝对路径）
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "knowledge_base")
KB_LAW = os.path.join(VECTOR_STORE_PATH, "law")

# 法规 PDF 目录（用于扩展法律法规知识库）
LAW_PDF_DIR = os.path.join(PROJECT_ROOT, "data", "law")

# 分块配置
LAW_CHUNK_SIZE = 800
LAW_CHUNK_OVERLAP = 100
PROJECT_CHUNK_SIZE = 500
PROJECT_CHUNK_OVERLAP = 50

# 检索 top_k 配置（漏斗结构：子检索器 → Fusion 汇总 → Reranker 精选）
LAW_VECTOR_TOP_K = 30
LAW_BM25_TOP_K = 60
LAW_FUSION_TOP_K = 60       # Fusion 向 Reranker 传递的候选数

# 融合检索权重 [BM25, vector]
LAW_RETRIEVER_WEIGHTS = [0.2, 0.8]

# Reranker 模型路径（设为空字符串 '' 则跳过 rerank）
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "bge-reranker-v2-m3"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")  # HF 模型 ID

# Reranker 按知识库独立配置输出数
LAW_RERANKER_TOP_N = 5      # 法律法规精排输出数

# ========== 项目库 NL2SQL 配置 ==========
PROJECT_SQLITE_DB = os.path.join(PROJECT_ROOT, "data", "project", "sanitation_projects.db")

# SQL 路由 LLM 配置（暂用 deepseek/本地模型）
SQL_ROUTER_MODEL = os.getenv("SQL_ROUTER_MODEL", LLM_MODEL)
SQL_ROUTER_API_KEY = os.getenv("SQL_ROUTER_API_KEY", LLM_API_KEY)
SQL_ROUTER_BASE_URL = os.getenv("SQL_ROUTER_BASE_URL", LLM_BASE_URL)


def resolve_model_path(local_path: str, hf_model_name: str) -> str:
    """本地路径存在则用本地，否则返回 HF 模型 ID 触发自动下载。

    HF 下载的模型会缓存到 ~/.cache/huggingface/（Docker 中挂载为 hf_cache 卷）。
    """
    import os
    if local_path and os.path.isdir(local_path) and os.listdir(local_path):
        return local_path
    return hf_model_name
