# -*- coding: utf-8 -*-
"""
Agent适配器模块（支持知识库路由）
==============================
功能：
1. 封装 Agent 的检索逻辑，供评估系统调用
2. 支持按知识库名称路由到对应的检索工具
3. 保持与生产环境一致的检索逻辑（使用相同的索引和检索器）
4. 支持模拟 Agent 调用流程，返回检索结果和生成回答
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from server.rag.jieba_bm25 import JiebaBM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from llama_index.core import Settings
from server.rag.vector_store import check_load_law_kb
from server.rag.project_query_engine import get_query_engine as get_project_engine
from configs.config import (
    EMBEDDING_MODEL, EMBEDDING_SERVICE, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_NORMALIZE, EMBEDDING_BATCH_SIZE,
    LAW_BM25_TOP_K, LAW_FUSION_TOP_K,
    LAW_RETRIEVER_WEIGHTS,
    RERANKER_MODEL_PATH, RERANKER_MODEL_NAME, resolve_model_path,
)
from configs.prompt_config import rag_answer_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果的数据类"""
    query: str
    nodes: List = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    metadatas: List[Dict] = field(default_factory=list)
    sql: str = ""         # 生成的 SQL（仅 project NL2SQL）
    sql_error: str = ""   # SQL 执行错误（空字符串表示成功）
    row_count: int = 0    # SQL 返回的行数（仅 project）
    results: List[Dict] = field(default_factory=list)  # 实际执行结果（仅 project NL2SQL）


class AgentAdapter:
    """
    Agent 适配器
    封装 Agent 的检索逻辑，供评估系统调用。
    支持按知识库路由到对应的检索工具。
    """

    # 类级别缓存，避免重复加载索引
    _index_cache: Dict[str, VectorStoreIndex] = {}
    _engine_cache: Dict[str, object] = {}

    def __init__(self, llm=None):
        """
        初始化 Agent 适配器

        Args:
            llm: LLM 实例，用于生成回答（可选）
        """
        self.llm = llm
        self._reranker = None
        self._load_indexes()

    def _load_indexes(self):
        """加载所有知识库的索引（project 使用 NL2SQL，无需向量索引）"""
        # 确保嵌入模型已初始化（避免加载索引时使用默认的 OpenAI 嵌入模型）
        self._ensure_embed_model()

        if "law" not in self._index_cache:
            logger.info("加载法律知识库索引...")
            self._index_cache["law"] = check_load_law_kb()
            logger.info("法律知识库索引加载完成")

    @staticmethod
    def _ensure_embed_model():
        """确保 Settings.embed_model 和 Settings.llm 已初始化，避免加载索引或创建检索器时使用默认的 OpenAI 模型"""
        # 初始化嵌入模型
        if Settings._embed_model is None:
            logger.info(f"初始化嵌入模型: {EMBEDDING_MODEL} (service={EMBEDDING_SERVICE})")
            if EMBEDDING_SERVICE == "ollama":
                from llama_index.embeddings.ollama import OllamaEmbedding
                Settings.embed_model = OllamaEmbedding(
                    model_name=EMBEDDING_MODEL,
                    normalize=EMBEDDING_NORMALIZE,
                    embed_batch_size=EMBEDDING_BATCH_SIZE
                )
            elif EMBEDDING_SERVICE == "huggingface":
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                model_path = resolve_model_path(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)
                Settings.embed_model = HuggingFaceEmbedding(model_name=model_path, embed_batch_size=EMBEDDING_BATCH_SIZE)
            else:
                raise ValueError(f"不支持的嵌入模型服务: {EMBEDDING_SERVICE}")
            logger.info(f"嵌入模型初始化完成: {EMBEDDING_MODEL}")

        # 初始化 LLM（避免 QueryFusionRetriever 等组件访问 Settings.llm 时触发默认 OpenAI 加载）
        if Settings._llm is None:
            from llama_index.core.llms.mock import MockLLM
            logger.info("初始化 MockLLM（避免 Settings.llm 触发默认 OpenAI 加载）")
            Settings.llm = MockLLM()

    def get_index(self, kb_name: str):
        """
        获取指定知识库的索引

        Args:
            kb_name: 知识库名称 ("law" 或 "project")

        Returns:
            VectorStoreIndex 实例（project 返回 None）
        """
        return self._index_cache.get(kb_name)

    def get_retriever(self, kb_name: str, similarity_top_k: int = 5):
        """
        获取指定知识库的检索器（与生产环境一致）

        Args:
            kb_name: 知识库名称
            similarity_top_k: 检索返回的 top-k 结果数

        Returns:
            检索器实例（project 返回 ProjectQueryEngine）
        """
        if kb_name == "project":
            if "project_engine" not in self._engine_cache:
                self._engine_cache["project_engine"] = get_project_engine()
            return self._engine_cache["project_engine"]

        index = self.get_index(kb_name)
        if index is None:
            raise ValueError(f"未知的知识库: {kb_name}")

        if kb_name == "law":
            # 子检索器需要更多候选供 fusion 挑选；BM25 按 kb_name 缓存（建索引慢）
            bm25_top_k = max(LAW_BM25_TOP_K, similarity_top_k)
            bm25_cache = f"bm25_{kb_name}"
            if bm25_cache not in self._engine_cache:
                self._engine_cache[bm25_cache] = JiebaBM25Retriever.from_defaults(
                    docstore=index.docstore, similarity_top_k=bm25_top_k,
                )
            retriever = QueryFusionRetriever(
                [self._engine_cache[bm25_cache],
                 VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)],
                similarity_top_k=similarity_top_k,
                num_queries=1, mode="relative_score",
                retriever_weights=LAW_RETRIEVER_WEIGHTS, use_async=False,
            )
        else:
            raise ValueError(f"未知的知识库: {kb_name}")

        return retriever

    def _get_reranker(self):
        """延迟初始化 reranker 模型"""
        if self._reranker is not None:
            return self._reranker
        if not RERANKER_MODEL_PATH and not RERANKER_MODEL_NAME:
            return None
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            model_path = resolve_model_path(RERANKER_MODEL_PATH, RERANKER_MODEL_NAME)
            logger.info(f"加载 reranker 模型: {model_path}")
            self._reranker = SentenceTransformerRerank(
                top_n=LAW_FUSION_TOP_K, model=model_path, device="cuda",
            )
            # 关掉 CrossEncoder 的批处理进度条（不刷屏）
            import logging as _logging
            _logging.getLogger("transformers").setLevel(_logging.WARNING)
            _logging.getLogger("sentence_transformers").setLevel(_logging.WARNING)
            logger.info("reranker 模型加载完成")
        except Exception as e:
            logger.warning(f"reranker 加载失败（跳过）: {e}")
            self._reranker = False  # 标记失败，避免重复尝试
        return self._reranker if self._reranker else None

    def retrieve(
        self,
        query: str,
        kb_name: str,
        similarity_top_k: int = 5,
    ) -> RetrievalResult:
        """
        执行检索操作

        Args:
            query: 查询字符串
            kb_name: 知识库名称
            similarity_top_k: 返回的 top-k 结果数（用于 law；project 固定返回）

        Returns:
            RetrievalResult 实例
        """
        if kb_name == "project":
            engine = self.get_retriever("project")
            nl_result = engine.query(query)
            results_data = nl_result.get("results", [])
            result = RetrievalResult(
                query=query, sql=nl_result.get("sql", ""),
                sql_error=nl_result.get("error", ""),
                row_count=len(results_data),
                results=results_data,
            )
            if "error" not in nl_result:
                # 将 NL2SQL 结果包装为 RetrievalResult
                for row in results_data[:similarity_top_k]:
                    text = " | ".join(f"{k}: {v}" for k, v in row.items())
                    result.contexts.append(text)
                    result.scores.append(1.0)
            return result

        # 有 reranker 时，融合检索多返回一些候选，让 reranker 从中选最优的
        candidate_k = max(similarity_top_k * 3, 20) if self._get_reranker() else similarity_top_k
        retriever = self.get_retriever(kb_name, candidate_k)
        nodes = retriever.retrieve(query)

        # 可选：reranker 重排序（top_n 设大一点，截断由 similarity_top_k 控制）
        reranker = self._get_reranker()
        if reranker is not None:
            nodes = reranker.postprocess_nodes(nodes, query_str=query)[:similarity_top_k]

        result = RetrievalResult(query=query)
        for node in nodes:
            result.nodes.append(node)
            result.contexts.append(node.text)
            result.scores.append(float(node.score) if hasattr(node, 'score') and node.score else 0.0)
            result.node_ids.append(node.node_id)
            result.metadatas.append(dict(node.metadata) if hasattr(node, 'metadata') else {})

        return result

    def retrieve_with_context(
        self,
        query: str,
        kb_name: str,
        similarity_top_k: int = 5,
    ) -> str:
        """
        执行检索并返回格式化的上下文文本（与生产环境工具函数一致）

        Args:
            query: 查询字符串
            kb_name: 知识库名称
            similarity_top_k: 返回的 top-k 结果数

        Returns:
            格式化的上下文文本
        """
        if kb_name == "project":
            result = self.retrieve(query, kb_name, similarity_top_k)
            if not result.contexts:
                return "未在项目库中找到相关信息。"
            return "\n\n".join(
                f"[结果 {i + 1}]\n{ctx}" for i, ctx in enumerate(result.contexts)
            )

        result = self.retrieve(query, kb_name, similarity_top_k)

        if not result.nodes:
            if kb_name == "law":
                return "未在法律法规库中找到相关信息。"
            else:
                return "未在招投标项目库中找到相关信息。"

        if kb_name == "law":
            context_parts = []
            for i, node in enumerate(result.nodes):
                context_parts.append(
                    f"[资料 {i + 1}] (相关度: {result.scores[i]:.4f})\n"
                    f"标题: {node.metadata.get('title', '未知')}\n"
                    f"发布时间: {node.metadata.get('publish_time', '未知')}\n"
                    f"施行时间: {node.metadata.get('effective_time', '未知')}\n"
                    f"来源: {node.metadata.get('source_website', '未知')}\n"
                    f"内容: {node.text}"
                )
        else:
            context_parts = []
            for i, node in enumerate(result.nodes):
                context_parts.append(
                    f"[资料 {i + 1}] (相关度: {result.scores[i]:.4f})\n"
                    f"项目名称: {node.metadata.get('project_name', '未知')}\n"
                    f"招标单位: {node.metadata.get('title', '未知')} | "
                    f"类别: {node.metadata.get('category', '未知')}\n"
                    f"预算金额: {node.metadata.get('budget', '未知')}\n"
                    f"所在地区: {node.metadata.get('region', '未知')}\n"
                    f"内容: {node.text}"
                )

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        context: str,
        kb_name: str,
    ) -> str:
        """
        基于检索上下文生成回答（模拟 Agent 的生成过程）

        Args:
            query: 用户查询
            context: 检索到的上下文文本
            kb_name: 知识库名称

        Returns:
            生成的回答文本
        """
        if self.llm is None:
            logger.warning("未设置 LLM，无法生成回答")
            return ""

        prompt = rag_answer_prompt(query, context)

        try:
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return f"生成回答时出错: {e}"

    def query(
        self,
        query: str,
        kb_name: str,
        similarity_top_k: int = 5,
    ) -> Tuple[str, RetrievalResult]:
        """
        完整的 RAG 查询流程：检索 + 生成

        Args:
            query: 用户查询
            kb_name: 知识库名称
            similarity_top_k: top-k 参数

        Returns:
            (answer, retrieval_result) 元组
        """
        if kb_name == "project":
            engine = self.get_retriever("project")
            nl_result = engine.query(query)
            context = ""
            if "error" not in nl_result:
                rows = []
                for row in nl_result["results"][:similarity_top_k]:
                    rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
                context = "\n".join(rows)

            results_data = nl_result.get("results", [])
            retrieval_result = RetrievalResult(
                query=query, sql=nl_result.get("sql", ""),
                sql_error=nl_result.get("error", ""),
                row_count=len(results_data),
                results=results_data,
            )
            if context:
                retrieval_result.contexts = [context]

            answer = self.generate_answer(query, context, kb_name)
            return answer, retrieval_result

        # 1. 检索
        context = self.retrieve_with_context(query, kb_name, similarity_top_k)
        retrieval_result = self.retrieve(query, kb_name, similarity_top_k)

        # 2. 生成
        answer = self.generate_answer(query, context, kb_name)

        return answer, retrieval_result

    def get_available_knowledge_bases(self) -> List[str]:
        """获取所有可用的知识库名称列表"""
        return list(self._index_cache.keys())


