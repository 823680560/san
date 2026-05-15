"""
共享 Agent 工厂 — 供 main.py (CLI) 和 web/api/chat_api.py (Web) 共用。
"""
import logging
import re
import threading

from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from configs.prompt_config import AGENT_PREFIX
from server.rag.vector_store import check_load_law_kb
from server.agent.tools.law_search import get_law_search_tool
from server.agent.tools.project_search_db import get_project_search_db_tool
from server.agent.tools.search_internet import search_internet

logger = logging.getLogger(__name__)

_agent_executors = {}
_agent_lock = threading.Lock()
_llm = None
_tools = None


class _SafeMemory(ConversationBufferWindowMemory):
    """过滤 LLM 输出中的 Unicode 非法代理字符，防止序列化报错"""
    def save_context(self, inputs, outputs):
        def _clean(d):
            return {k: re.sub(r'[\ud800-\udfff]', '', str(v)) if isinstance(v, str) else v
                    for k, v in d.items()}
        super().save_context(_clean(inputs), _clean(outputs))


def _get_shared_resources():
    """懒加载 LLM 和 tools（全局共享，不重复创建）"""
    global _llm, _tools

    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.7,
            request_timeout=120.0,
            streaming=True,
        )

    if _tools is None:
        law_index = check_load_law_kb()
        _tools = [
            get_project_search_db_tool(),
            get_law_search_tool(law_index),
            search_internet,
        ]

    return _llm, _tools


def _create_agent():
    """创建新的 Agent 实例（memory 独立，LLM/tools 共享）"""
    llm, tools = _get_shared_resources()

    memory = _SafeMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            'prefix': AGENT_PREFIX,
            'suffix': 'Begin!\n\n{chat_history}\nQuestion: {input}\nThought:{agent_scratchpad}',
        },
        max_iterations=5,
        memory=memory,
        return_intermediate_steps=True,
    )

    return agent


def init_agent(session_id: str = "default"):
    """获取或创建 session 级 Agent（线程安全）"""
    if session_id in _agent_executors:
        return _agent_executors[session_id]

    with _agent_lock:
        if session_id in _agent_executors:
            return _agent_executors[session_id]

        logger.info(f"正在为 session={session_id} 初始化 Agent...")
        _agent_executors[session_id] = _create_agent()
        logger.info(f"Agent 初始化完成 (session={session_id})")
        return _agent_executors[session_id]


def run_agent_query(query: str, session_id: str = "default") -> dict:
    """执行 Agent 查询，返回 {answer, thinking_steps, tool_calls_count}"""
    from server.agent.tools.search_internet import clear_search_cache
    clear_search_cache()

    agent = init_agent(session_id)
    raw = agent.invoke({"input": query})

    thinking_steps = []
    tool_counts = {}

    for action, obs in raw.get("intermediate_steps", []):
        tool_name = action.tool
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        thinking_steps.append({
            "step": len(thinking_steps) + 1,
            "tool": tool_name,
            "tool_input": str(action.tool_input),
            "observation_preview": str(obs)[:2000],
        })

    return {
        "answer": raw.get("output", "Agent 未能生成回答"),
        "thinking_steps": thinking_steps,
        "tool_calls_count": tool_counts,
    }


def run_agent_query_streaming(query: str, session_id: str = "default", callbacks: list = None) -> dict:
    """执行 Agent 查询（带回调支持，用于流式输出）"""
    from server.agent.tools.search_internet import clear_search_cache
    clear_search_cache()

    agent = init_agent(session_id)
    config = {"callbacks": callbacks} if callbacks else None
    raw = agent.invoke({"input": query}, config=config)

    thinking_steps = []
    tool_counts = {}

    for action, obs in raw.get("intermediate_steps", []):
        tool_name = action.tool
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        thinking_steps.append({
            "step": len(thinking_steps) + 1,
            "tool": tool_name,
            "tool_input": str(action.tool_input),
            "observation_preview": str(obs)[:2000],
        })

    return {
        "answer": raw.get("output", "Agent 未能生成回答"),
        "thinking_steps": thinking_steps,
        "tool_calls_count": tool_counts,
    }


def get_agent_history(session_id: str) -> list:
    """读取 session 的对话历史，返回 [{"role": "user", "content": ...}, ...]"""
    agent = _agent_executors.get(session_id)
    if not agent:
        return []
    messages = agent.memory.chat_memory.messages
    result = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        result.append({"role": role, "content": msg.content})
    return result


def seed_agent_memory(session_id: str, history_entries: list):
    """将外部历史灌入 Agent memory（先清空后加载）"""
    agent = _agent_executors.get(session_id)
    if not agent:
        return
    agent.memory.chat_memory.clear()
    for i in range(0, len(history_entries), 2):
        user_msg = history_entries[i]
        ai_msg = history_entries[i + 1] if i + 1 < len(history_entries) else None
        if user_msg.get("role") == "user" and ai_msg and ai_msg.get("role") == "assistant":
            agent.memory.save_context(
                {"input": user_msg["content"]},
                {"output": ai_msg["content"]},
            )


def clear_agent_memory(session_id: str):
    """清空指定 session 的 Agent"""
    if session_id in _agent_executors:
        del _agent_executors[session_id]
        logger.info(f"已清空 session={session_id} 的 Agent memory")
