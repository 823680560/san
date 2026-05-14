"""bge-reranker-v2-m3 重排序器"""

import logging
from typing import List, Optional

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class BGEReranker:
    """基于 bge-reranker-v2-m3 的重排序器。

    对混合检索（向量+BM25）的候选结果做 cross-encoder 重新打分排序。
    若 model_path 为空字符串，则跳过重排序，退化为直接取 top_n。
    """

    def __init__(self, model_path: str = "", hf_model_name: str = "", top_n: int = 10, use_fp16: bool = False):
        self.top_n = top_n
        self._model = None

        if not model_path:
            logger.warning("Reranker 模型路径为空，跳过重排序")
            return

        from configs.config import resolve_model_path

        actual_path = resolve_model_path(model_path, hf_model_name)

        try:
            import torch
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(
                actual_path, trust_remote_code=True, device=device,
            )
            logger.info(f"已加载 CrossEncoder 重排序模型: {actual_path} (device={device})")
        except ImportError:
            logger.error(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"加载 CrossEncoder 模型失败: {e}")
            self._model = None

    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: Optional[int] = None) -> List[NodeWithScore]:
        """对检索结果做重排序，返回 top_n。

        Args:
            query: 查询字符串。
            nodes: 待重排序的候选节点列表。
            top_k: 可选，覆盖 self.top_n 控制最终返回数。
        """
        top_n = top_k if top_k is not None else self.top_n

        if not nodes:
            return nodes

        if self._model is None:
            return nodes[:top_n]

        pairs = [(query, node.text) for node in nodes]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for node, score in zip(nodes, scores):
            node.score = float(score)

        nodes.sort(key=lambda x: x.score, reverse=True)
        return nodes[:top_n]
