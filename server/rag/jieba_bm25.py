"""Jieba-based BM25 retriever for Chinese text segmentation."""

import logging
import re
from typing import Callable, Dict, List, Optional, cast

import jieba
import bm25s
import numpy as np
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    MetadataMode,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
    build_metadata_filter_fn,
)

logger = logging.getLogger(__name__)


def jieba_tokenize(text: str) -> List[str]:
    """Tokenize Chinese text using jieba."""
    # Strip illegal surrogate characters (common in OCR'd PDFs)
    text = re.sub(r'[\ud800-\udfff]', '', text) # 清理不完整特殊字符
    return [t.strip() for t in jieba.cut(text) if t.strip()]


class JiebaBM25Retriever(BaseRetriever):
    """
    一个使用 jieba 进行中文分词的 BM25 检索器。

    作为 llama-index-retrievers-bm25 中 BM25Retriever 的即插即用替代方案，它将默认的基于正则表达式的分词器替换为 jieba 分词，从而实现对中文关键词的正确匹配。
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        filters: Optional[MetadataFilters] = None,
        corpus_weight_mask: Optional[List[int]] = None,
    ) -> None:
        self.similarity_top_k = similarity_top_k

        if nodes is None:
            raise ValueError("Please pass nodes.")

        self.corpus = [
            node_to_metadata_dict(node) | {"node_id": node.node_id}
            for node in nodes
        ]

        corpus_tokens = [
            jieba_tokenize(node.get_content(metadata_mode=MetadataMode.EMBED))
            for node in nodes
        ]
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens, show_progress=verbose)

        if (
            self.bm25.scores.get("num_docs")
            and int(self.bm25.scores["num_docs"]) < self.similarity_top_k
        ):
            if int(self.bm25.scores["num_docs"]) == 0:
                raise ValueError(
                    "No nodes added to the retriever kindly add more data."
                )
            logger.warning(
                "Overriding similarity_top_k to %d (corpus size)",
                int(self.bm25.scores["num_docs"]),
            )
            self.similarity_top_k = int(self.bm25.scores["num_docs"])

        self.corpus_weight_mask = corpus_weight_mask or None
        if filters and self.corpus:
            _corpus_dict = {
                corpus_token["node_id"]: corpus_token
                for corpus_token in self.corpus
            }
            _query_filter_fn = build_metadata_filter_fn(
                lambda node_id: _corpus_dict[node_id], filters
            )
            self.corpus_weight_mask = [
                int(_query_filter_fn(corpus_token["node_id"]))
                for corpus_token in self.corpus
            ]

            if not any(self.corpus_weight_mask):
                raise ValueError(
                    "All nodes were filtered out by the metadata filters. "
                    "Please adjust your filters or add more data."
                )

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        filters: Optional[MetadataFilters] = None,
    ) -> "JiebaBM25Retriever":
        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert nodes is not None, (
            "Please pass exactly one of index, nodes, or docstore."
        )

        return cls(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            filters=filters,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = [jieba_tokenize(query)]

        indexes, scores = self.bm25.retrieve(
            tokenized_query,
            k=self.similarity_top_k,
            show_progress=self._verbose,
            weight_mask=np.array(self.corpus_weight_mask)
            if self.corpus_weight_mask
            else None,
        )

        # batched, but only one query
        indexes = indexes[0]
        scores = scores[0]

        nodes: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes
