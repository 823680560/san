import json
import logging

logging.basicConfig(level=logging.INFO)

from llama_index.core import Settings
from langchain_openai import ChatOpenAI
from llama_index.llms.langchain import LangChainLLM
from configs.config import (
    EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_SERVICE, EMBEDDING_NORMALIZE, EMBEDDING_BATCH_SIZE,
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, OLLAMA_BASE_URL, resolve_model_path
)


def init_embedding_model():
    """初始化嵌入模型（从 config 统一读取参数）"""
    if Settings._embed_model is None:
        if EMBEDDING_SERVICE == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            Settings.embed_model = OllamaEmbedding(
                model_name=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
                normalize=EMBEDDING_NORMALIZE,
                embed_batch_size=EMBEDDING_BATCH_SIZE
            )
            logging.info(f"已初始化 Ollama 嵌入模型: {EMBEDDING_MODEL}")
        elif EMBEDDING_SERVICE == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            model_path = resolve_model_path(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)
            Settings.embed_model = HuggingFaceEmbedding(model_name=model_path, embed_batch_size=EMBEDDING_BATCH_SIZE)
            logging.info(f"已初始化 HuggingFace 嵌入模型: {model_path}")
        else:
            raise ValueError(f"不支持的嵌入模型服务: {EMBEDDING_SERVICE}")


# 1. 设置 Embedding 模型和 LLM
init_embedding_model()

langchain_llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0.7,
    request_timeout=120.0
)
Settings.llm = LangChainLLM(llm=langchain_llm)


def _sanitize(text: str) -> str:
    """去除非法 Unicode 代理字符"""
    import re
    return re.sub(r'[\ud800-\udfff]', '', str(text))


def run():
    from server.agent.agent_factory import init_agent

    agent_executor = init_agent()

    while True:
        query = input("我:").strip()
        if query == 'q':
            break

        try:
            print("\n--- 🕵️ Agent 开始思考 ---")
            result = agent_executor.invoke({"input": _sanitize(query)})
            output = _sanitize(str(result.get('output', '')))
            print(f"\n🤖 最终回答: {output}\n")
            print("--- ✅ 思考结束 ---\n")

        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == '__main__':
    run()

