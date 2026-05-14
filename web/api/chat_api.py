"""
RAG 对话 API
支持选择知识库、检索、生成回答
"""
import os
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")

# 全局缓存：已加载的索引和检索引擎
_index_cache = {}
_engine_cache = {}

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """对话请求"""
    query: str
    kb_name: str = "law"
    top_k: int = 5
    session_id: str = "default"


class ChatMessage(BaseModel):
    """对话消息"""
    role: str  # "user" 或 "assistant"
    content: str


class ClearChatRequest(BaseModel):
    kb_name: str = "law"
    session_id: str = "default"


# 对话历史（所有模式的唯一数据源，按 session_id 存储）
chat_histories: dict = {}


def _format_history(entries: list, max_turns: int = 5) -> str:
    """将 [{role, content}] 转为 LLM 可读的历史字符串，最多保留最近 max_turns 轮"""
    if not entries:
        return ""
    # 只保留最近 max_turns 轮（每轮 = 1 user + 1 assistant = 2 条消息）
    recent = entries[-(max_turns * 2):]
    lines = []
    for msg in recent:
        prefix = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)


def _load_index(kb_name: str):
    """加载知识库索引（带缓存）"""
    if kb_name in _index_cache:
        return _index_cache[kb_name]

    kb_path = os.path.join(KB_DIR, kb_name)
    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail=f"知识库 {kb_name} 不存在")

    from server.rag.vector_store import load_faiss_index
    index = load_faiss_index(kb_path)
    _index_cache[kb_name] = index
    return index


def _get_search_engine(kb_name: str):
    """获取检索引擎（带缓存）"""
    cache_key = f"engine_{kb_name}"
    if cache_key in _engine_cache:
        return _engine_cache[cache_key]

    # 根据知识库名称选择检索引擎
    if kb_name == "project":
        from server.rag.project_query_engine import get_query_engine
        engine = get_query_engine()
        _engine_cache[cache_key] = engine
        return engine

    index = _load_index(kb_name)

    if kb_name == "law":
        from server.agent.tools.law_search import LawSearchEngine
        engine = LawSearchEngine(index)
    else:
        # 通用检索引擎：使用混合检索
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.retrievers import QueryFusionRetriever

        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

        class GenericSearchEngine:
            def __init__(self, retriever):
                self.retriever = retriever

            def search(self, query: str, top_k: int = 5) -> str:
                nodes = self.retriever.retrieve(query)
                if not nodes:
                    return "未找到相关信息。"
                nodes = nodes[:top_k]
                context = "\n\n".join([
                    f"[资料 {i + 1}] (相关度: {node.score:.4f})\n{node.text}"
                    for i, node in enumerate(nodes)
                ])
                return context

        # 尝试混合检索，若 BM25 不可用则退化到纯向量检索
        try:
            from server.rag.jieba_bm25 import JiebaBM25Retriever
            bm25_retriever = JiebaBM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=5)
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=5,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False
            )
        except Exception:
            logger.warning(f"BM25 混合检索初始化失败，回退到纯向量检索")
            retriever = vector_retriever

        engine = GenericSearchEngine(retriever)

    _engine_cache[cache_key] = engine
    return engine


_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
        from eval.llm_wrapper import SimpleLLM
        _llm_instance = SimpleLLM(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
        )
    return _llm_instance


def _generate_answer(query: str, context: str, history_str: str = "") -> str:
    """使用 LLM 基于检索结果生成回答"""
    llm = _get_llm()

    history_section = ""
    if history_str:
        history_section = f"对话历史：\n{history_str}\n\n"

    prompt = f"""你是一个专业的环卫行业智能问答系统。请根据以下参考资料，回答用户的问题。

要求：
1. 答案必须基于参考资料中的信息，不要编造
2. 如果参考资料不足以回答问题，请明确说明
3. 答案要全面、准确，尽量覆盖问题涉及的所有知识点
4. 不要输出"根据参考资料"、"根据文本"等前缀，直接给出答案

{history_section}参考资料：
{context}

用户问题：{query}

答案："""

    try:
        result = llm.complete(prompt)
        return result.text.strip()
    except Exception as e:
        logger.error(f"LLM 生成回答失败: {e}")
        return f"生成回答时出错: {str(e)}"


async def _agent_chat(query: str, session_id: str):
    """Agent 模式：自动识别问题类型，选择知识库"""
    from server.agent.agent_factory import run_agent_query, seed_agent_memory

    # ① 将统一历史灌入 Agent memory（保证跨模式历史连续）
    history = chat_histories.get(session_id, [])
    seed_agent_memory(session_id, history)

    # ② 执行 Agent
    result = run_agent_query(query, session_id=session_id)

    # ③ 回写到统一存储
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    chat_histories[session_id].append({"role": "user", "content": query})
    chat_histories[session_id].append({"role": "assistant", "content": result["answer"]})

    return JSONResponse({
        "success": True,
        "answer": result["answer"],
        "context": None,
        "kb_name": "auto",
        "agent_mode": True,
        "thinking_steps": result["thinking_steps"],
    })


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    发送消息并获取 RAG 回答

    kb_name="auto" → Agent 自动识别路由
    其他值 → 单知识库检索 + LLM 生成
    """
    try:
        if req.kb_name == "auto":
            return await _agent_chat(req.query, req.session_id)

        # 获取检索引擎
        engine = _get_search_engine(req.kb_name)

        # 检索（兼容 NL2SQL 引擎和传统检索引擎）
        from server.rag.project_query_engine import ProjectQueryEngine
        if isinstance(engine, ProjectQueryEngine):
            nl_result = engine.query(req.query)
            if "error" in nl_result:
                context = f"查询出错: {nl_result['error']}"
            else:
                rows = []
                for row in nl_result["results"][:10]:
                    rows.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v is not None))
                context = "\n".join(rows) if rows else "未找到相关信息。"
        else:
            context = engine.search(req.query, top_k=req.top_k)

        # 生成回答（传入统一历史，保证跨模式上下文）
        history = chat_histories.get(req.session_id, [])
        history_str = _format_history(history)
        answer = _generate_answer(req.query, context, history_str)

        # 保存对话历史到统一存储
        if req.session_id not in chat_histories:
            chat_histories[req.session_id] = []
        chat_histories[req.session_id].append({"role": "user", "content": req.query})
        chat_histories[req.session_id].append({"role": "assistant", "content": answer})

        return JSONResponse({
            "success": True,
            "answer": answer,
            "context": context,
            "kb_name": req.kb_name,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")


@router.get("/chat/history")
async def get_chat_history(session_id: str = "default"):
    """获取对话历史（所有模式统一从 chat_histories 读取）"""
    history = chat_histories.get(session_id, [])
    return JSONResponse({"success": True, "history": history})


@router.post("/chat/clear")
async def clear_chat_history(req: ClearChatRequest):
    """清空对话历史（同时清空 chat_histories 和 Agent memory）"""
    chat_histories[req.session_id] = []
    from server.agent.agent_factory import clear_agent_memory
    clear_agent_memory(req.session_id)
    return JSONResponse({"success": True, "message": "对话历史已清空"})
