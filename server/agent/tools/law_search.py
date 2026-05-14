import datetime
import re

from langchain.tools import tool
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from server.rag.jieba_bm25 import JiebaBM25Retriever
from server.rag.reranker import BGEReranker
from configs.config import (
    LAW_VECTOR_TOP_K, LAW_BM25_TOP_K, LAW_FUSION_TOP_K,
    LAW_RETRIEVER_WEIGHTS,
    RERANKER_MODEL_PATH, RERANKER_MODEL_NAME, LAW_RERANKER_TOP_N,
)
from typing import Optional


def _sanitize_text(text: str) -> str:
    """清洗文本中的非法 Unicode 代理字符（OCR 扫描件常见问题）"""
    text = re.sub(r'[\ud800-\udfff]', '', str(text))
    return text

# 定义全局缓存列表, 记录每次检索的日志，用于检索评估
RETRIEVAL_LOG = []

class LawSearchEngine:
    """
    法律检索引擎封装类。
    负责管理 LlamaIndex 的索引实例和混合检索逻辑。
    """
    def __init__(self, index):
        self.index = index
        self.reranker = BGEReranker(model_path=RERANKER_MODEL_PATH, hf_model_name=RERANKER_MODEL_NAME, top_n=LAW_RERANKER_TOP_N)
        self.retriever = self._init_retriever()
        self._cache = {}

    def _init_retriever(self):
        vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=LAW_VECTOR_TOP_K)
        bm25_retriever = JiebaBM25Retriever.from_defaults(docstore=self.index.docstore, similarity_top_k=LAW_BM25_TOP_K)

        return QueryFusionRetriever(
            [bm25_retriever, vector_retriever],
            similarity_top_k=LAW_FUSION_TOP_K,
            num_queries=1,
            mode="relative_score",
            retriever_weights=LAW_RETRIEVER_WEIGHTS,
            use_async=False
        )

    def search(self, query: str, top_k: Optional[int] = None) -> str:
        """执行检索并返回格式化字符串（相同查询自动复用缓存）"""
        top_k = top_k if top_k is not None else LAW_RERANKER_TOP_N
        cache_key = f"{query.strip()}::top_k={top_k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            nodes = self.retriever.retrieve(query)
            nodes = self.reranker.rerank(query, nodes, top_k=top_k)

            log_results = []
            for i, node in enumerate(nodes):
                log_results.append({
                    "rank": i + 1,
                    "node_id": node.node_id,
                    "score": float(node.score), 
                    "metadata": {k: str(v) for k, v in node.metadata.items()}, 
                    "text": node.text[:200]
                })

            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "tool": "law_search",
                "results": log_results
            }
            RETRIEVAL_LOG.append(log_entry)

            if not nodes:
                self._cache[cache_key] = "未在法律法规库中找到相关信息。"
                return self._cache[cache_key]

            context = "\n\n".join([
                f"[资料 {i + 1}] (相关度: {node.score:.4f})\n"
                f"标题: {_sanitize_text(str(node.metadata.get('title', '')))}\n"
                f"发布时间: {_sanitize_text(str(node.metadata.get('publish_time', '')))}\n"
                f"施行时间: {_sanitize_text(str(node.metadata.get('effective_time', '')))}\n"
                f"来源: {_sanitize_text(str(node.metadata.get('source_website', '')))}\n"
                f"内容: {_sanitize_text(node.text)}"
                for i, node in enumerate(nodes)
            ])
            self._cache[cache_key] = context
            return context
        except Exception as e:
            return f"法律搜索工具执行出错: {str(e)}"

# 全局变量，用于存储初始化后的工具函数
_law_search_tool_func: Optional[callable] = None

def get_law_search_tool(index):
    """
    工厂函数：接收 LlamaIndex 索引，返回 LangChain 兼容的工具函数。
    利用单例模式确保工具只初始化一次。
    """
    global _law_search_tool_func
    
    if _law_search_tool_func is None:
        engine = LawSearchEngine(index)
        
        @tool
        def law_search(query: str) -> str:
            """
            搜索招标投标法、政府采购法等相关法律法规。
            当用户询问关于招标政策、专家条件、预算规定等细节时调用此工具。
            
            Args:
                query: 检索关键字或自然语言问题。
            """
            return engine.search(query)
            
        _law_search_tool_func = law_search
        
    return _law_search_tool_func
