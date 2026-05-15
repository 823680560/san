"""
SSE 流式对话 API
支持逐 token 推送回答，Agent 模式下实时推送思考步骤
"""
import asyncio
import json
import logging
import queue
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_classic.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from configs.config import STREAMING_ENABLED

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# ============================================================
# SSE 工具
# ============================================================

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============================================================
# Agent 回调处理器（线程安全）
# ============================================================

class _StreamingCallbackHandler(BaseCallbackHandler):
    """Agent 流式回调处理器，将 LLM 输出和工具调用推入线程安全队列"""

    def __init__(self, event_queue: queue.Queue):
        super().__init__()
        self._queue = event_queue
        self._step_counter = 0
        self._token_buffer = ""
        self._in_final_answer = False
        self._final_answer_started = False

    def _emit(self, event: dict):
        self._queue.put(event)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._token_buffer = ""
        self._in_final_answer = False
        self._final_answer_started = False

    def on_llm_new_token(self, token: str, **kwargs):
        if self._in_final_answer:
            if token:
                if not self._final_answer_started:
                    token = token.lstrip()
                    self._final_answer_started = True
                if token:
                    self._emit({"type": "token", "data": {"text": token}})
            return

        self._token_buffer += token
        marker = "Final Answer:"

        # 按行解析：每个完整行检查是否为 Thought，同时检查 Final Answer
        while '\n' in self._token_buffer:
            line, self._token_buffer = self._token_buffer.split('\n', 1)
            line_stripped = line.strip()
            if line_stripped.startswith('Thought:'):
                thought_text = line_stripped[len('Thought:'):].strip()
                if thought_text:
                    self._emit({
                        "type": "thought",
                        "data": {
                            "step": self._step_counter + 1,
                            "text": thought_text,
                        }
                    })
            elif marker in line_stripped:
                idx = line_stripped.index(marker) + len(marker)
                remaining = line_stripped[idx:]
                self._in_final_answer = True
                self._token_buffer = ""
                if remaining.strip():
                    self._final_answer_started = True
                    self._emit({"type": "token", "data": {"text": remaining.lstrip()}})
                return

        # 检查剩余 buffer 是否包含 Final Answer:（可能跨 token 未形成完整行）
        if marker in self._token_buffer:
            idx = self._token_buffer.index(marker) + len(marker)
            remaining = self._token_buffer[idx:]
            self._in_final_answer = True
            self._token_buffer = ""
            if remaining.strip():
                self._final_answer_started = True
                self._emit({"type": "token", "data": {"text": remaining.lstrip()}})

    def on_agent_action(self, action, **kwargs):
        self._step_counter += 1
        logger.info(f"Agent step {self._step_counter}: {action.tool}({str(action.tool_input)[:100]})")
        self._emit({
            "type": "thinking_step",
            "data": {
                "step": self._step_counter,
                "tool": action.tool,
                "tool_input": str(action.tool_input),
            }
        })

    def on_tool_end(self, output, **kwargs):
        self._emit({
            "type": "tool_result",
            "data": {
                "step": self._step_counter,
                "observation_preview": str(output)[:2000],
            }
        })

    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM error in agent: {error}")
        self._emit({"type": "error", "data": {"message": str(error)}})

    def on_tool_error(self, error, **kwargs):
        self._emit({"type": "error", "data": {"message": str(error)}})


# ============================================================
# 非 Agent 模式 SSE 生成器
# ============================================================

async def _kb_sse_generator(query: str, kb_name: str, top_k: int, session_id: str):
    """知识库检索 + 流式 LLM 生成"""
    from web.api.chat_api import _get_llm, _get_search_engine, _format_history, chat_histories
    from configs.prompt_config import chat_rag_prompt
    from server.rag.project_query_engine import ProjectQueryEngine

    try:
        # Phase 1: 检索
        yield _sse_event("status", {"status": "retrieving", "message": "正在检索..."})
        await asyncio.sleep(0)

        engine = _get_search_engine(kb_name)
        if isinstance(engine, ProjectQueryEngine):
            nl_result = engine.query(query)
            if "error" in nl_result:
                yield _sse_event("error", {"message": f"查询出错: {nl_result['error']}"})
                return
            rows = []
            for row in nl_result["results"][:10]:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v is not None))
            context = "\n".join(rows) if rows else "未找到相关信息。"
        else:
            context = engine.search(query, top_k=top_k)

        # Phase 2: 发送检索结果 + 构建 prompt
        yield _sse_event("context", {"text": context})

        history = chat_histories.get(session_id, [])
        history_str = _format_history(history)
        prompt = chat_rag_prompt(query, context, history_str)

        # Phase 3: 流式生成
        yield _sse_event("status", {"status": "generating", "message": "正在生成回答..."})

        llm = _get_llm()
        full_answer = ""

        async for chunk in iterate_in_threadpool(llm.stream(prompt)):
            if chunk:
                full_answer += chunk
                yield _sse_event("token", {"text": chunk})

        # Phase 4: 保存历史
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].append({"role": "user", "content": query})
        chat_histories[session_id].append({"role": "assistant", "content": full_answer})

        yield _sse_event("done", {"status": "complete"})

    except Exception as e:
        logger.error(f"流式生成失败: {e}", exc_info=True)
        yield _sse_event("error", {"message": str(e)})


# ============================================================
# Agent 模式 SSE 生成器
# ============================================================

async def _agent_sse_generator(query: str, session_id: str):
    """后台线程执行 Agent + 回调队列推送 SSE 事件"""
    from web.api.chat_api import chat_histories
    from server.agent.agent_factory import run_agent_query_streaming, seed_agent_memory

    loop = asyncio.get_running_loop()
    event_queue = queue.Queue()
    final_answer = [""]
    error_msg = [None]
    done_flag = threading.Event()

    callback = _StreamingCallbackHandler(event_queue)

    def _run_in_thread():
        try:
            history = chat_histories.get(session_id, [])
            seed_agent_memory(session_id, history)
            result = run_agent_query_streaming(query, session_id, callbacks=[callback])
            final_answer[0] = result["answer"]
        except Exception as e:
            error_msg[0] = str(e)
        finally:
            done_flag.set()

    yield _sse_event("status", {"status": "retrieving", "message": "Agent 正在思考..."})
    await asyncio.sleep(0)

    loop.run_in_executor(None, _run_in_thread)

    while not done_flag.is_set():
        had_events = False
        while True:
            try:
                event = event_queue.get_nowait()
                had_events = True
                yield _sse_event(event["type"], event["data"])
            except queue.Empty:
                break
        if not had_events:
            await asyncio.sleep(0.05)

    while True:
        try:
            event = event_queue.get_nowait()
            yield _sse_event(event["type"], event["data"])
        except queue.Empty:
            break

    if error_msg[0]:
        yield _sse_event("error", {"message": error_msg[0]})
        return

    if session_id not in chat_histories:
        chat_histories[session_id] = []
    chat_histories[session_id].append({"role": "user", "content": query})
    chat_histories[session_id].append({"role": "assistant", "content": final_answer[0]})

    yield _sse_event("done", {"status": "complete"})


# ============================================================
# 请求模型 & 端点
# ============================================================

class ChatStreamRequest(BaseModel):
    query: str
    kb_name: str = "law"
    top_k: int = 5
    session_id: str = "default"


@router.post("/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """
    POST /api/chat/stream — SSE 流式对话端点

    事件类型: status, thinking_step, tool_result, token, done, error
    """
    if not STREAMING_ENABLED:
        raise HTTPException(status_code=503, detail="流式输出未启用")

    if req.kb_name == "auto":
        generator = _agent_sse_generator(req.query, req.session_id)
    else:
        generator = _kb_sse_generator(req.query, req.kb_name, req.top_k, req.session_id)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
