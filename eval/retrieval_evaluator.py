# -*- coding: utf-8 -*-
"""
检索质量评估模块（支持多知识库）
==============================
功能：
1. 实现自定义指标：Hit Rate、MRR、Precision、Recall、NDCG
2. 支持在指定 top_k 值下评估
3. 支持多知识库独立评估和汇总对比
4. 输出详细的检索指标报告
"""

import json
import os
import logging
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from eval.agent_adapter import AgentAdapter, RetrievalResult
from configs.config import EMBEDDING_MODEL
from utils import get_model_short_name, get_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """检索指标的数据类"""
    hit_rate: float = 0.0
    mrr: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    ndcg: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4),
        }


class RetrievalEvaluator:
    """
    检索质量评估器
    评估检索系统在多个指标上的表现。
    """

    def __init__(self, agent_adapter: AgentAdapter, llm=None):
        """
        初始化检索评估器

        Args:
            agent_adapter: AgentAdapter 实例
            llm: LLM 实例（保留参数兼容）
        """
        self.agent_adapter = agent_adapter
        self.llm = llm

    def evaluate_knowledge_base(
        self,
        kb_name: str,
        dataset: Dict,
        top_k: int = 5,
    ) -> Dict:
        """
        评估单个知识库的检索质量

        Args:
            kb_name: 知识库名称
            dataset: 评测集字典
            top_k: top-k 值（默认5）

        Returns:
            评估结果字典
        """
        queries = dataset.get("queries", [])
        if not queries:
            logger.warning(f"知识库 {kb_name} 的评测集为空")
            return {}

        logger.info(f"开始评估知识库 {kb_name}，共 {len(queries)} 个查询，top_k={top_k}")

        metrics, details = self._evaluate_at_k(kb_name, queries, top_k)

        return {
            "knowledge_base": kb_name,
            "dataset_size": len(queries),
            "top_k": top_k,
            "metrics": metrics.to_dict(),
            "details": details,
        }

    def _evaluate_at_k(
        self,
        kb_name: str,
        queries: List[Dict],
        top_k: int,
    ) -> Tuple[RetrievalMetrics, List[Dict]]:
        """
        在指定的 top_k 下执行评估

        Args:
            kb_name: 知识库名称
            queries: 查询列表
            top_k: top-k 值

        Returns:
            (RetrievalMetrics, detailed_results) 元组
        """
        all_metrics = []
        detailed_results = []

        for q_data in tqdm(queries, desc=f"  Evaluating {kb_name} @{top_k}"):
            query = q_data["query"]
            ground_truth = q_data.get("ground_truth_answer", "")
            qid = q_data.get("qid", 0)
            # 获取 ground truth 节点的 node_id 列表（用于精确匹配）
            source_node_ids = q_data.get("source_node_ids", [])

            # 执行检索
            result = self.agent_adapter.retrieve(query, kb_name, similarity_top_k=top_k)

            # 计算指标（使用 node_id 精确匹配）
            metrics = self._compute_metrics(result, ground_truth, top_k, source_node_ids)

            all_metrics.append(metrics)

            # 记录详细信息
            detailed_results.append({
                "qid": qid,
                "query": query,
                "ground_truth": ground_truth[:200] if ground_truth else "",
                "source_node_ids": source_node_ids,
                "retrieved_node_ids": result.node_ids,
                "retrieved_scores": result.scores,
                "is_hit": metrics["is_hit"],
                "reciprocal_rank": metrics["reciprocal_rank"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ndcg": metrics["ndcg"],
                "relevant_count": metrics["relevant_count"],
            })


        # 汇总指标
        avg_metrics = self._average_metrics(all_metrics)

        return avg_metrics, detailed_results

    def _compute_metrics(
        self,
        result: RetrievalResult,
        ground_truth: str,
        top_k: int,
        source_node_ids: str = "",
    ) -> Dict:
        """
        计算单个查询的各项检索指标

        使用 node_id 精确匹配来判断相关性（而非文本重叠），
        确保评估结果的准确性和可复现性。

        Args:
            result: 检索结果
            ground_truth: 参考答案（保留参数兼容）
            top_k: top-k 值
            source_node_ids: 源节点 ID 列表（ground truth 节点），用于精确匹配。
                             可以是字符串（单个节点，兼容旧数据集）或列表（多个节点）。

        Returns:
            指标字典
        """
        # 统一转为列表
        if isinstance(source_node_ids, str):
            node_id_list = [source_node_ids] if source_node_ids else []
        else:
            node_id_list = source_node_ids

        # 判断相关文档：通过 node_id 精确匹配
        relevant_indices = []
        if node_id_list:
            for i, node_id in enumerate(result.node_ids):
                if node_id in node_id_list:
                    relevant_indices.append(i)
        else:
            # 兼容旧数据集：如果没有 source_node_id，使用文本重叠判断
            for i, context in enumerate(result.contexts):
                if self._is_relevant(ground_truth, context):
                    relevant_indices.append(i)


        total_relevant = len(relevant_indices)
        retrieved_count = len(result.nodes)

        # Hit Rate: 是否有相关文档被检索到
        is_hit = total_relevant > 0

        # MRR (Mean Reciprocal Rank): 第一个相关文档的排名的倒数
        reciprocal_rank = 0.0
        if relevant_indices:
            first_relevant_rank = relevant_indices[0] + 1  # 1-based rank
            reciprocal_rank = 1.0 / first_relevant_rank

        # Precision: 检索结果中相关文档的比例
        precision = total_relevant / max(retrieved_count, 1)

        # Recall: 相关文档中被检索到的比例
        # 分母为总相关文档数（source_node_ids 的长度），分子为命中的相关文档数
        total_ground_truth = len(node_id_list)
        recall = total_relevant / max(total_ground_truth, 1) if total_ground_truth > 0 else 0.0

        # NDCG (Normalized Discounted Cumulative Gain)
        ndcg = self._compute_ndcg(relevant_indices, retrieved_count, top_k, total_ground_truth)

        return {
            "is_hit": is_hit,
            "reciprocal_rank": reciprocal_rank,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "relevant_count": total_relevant,
        }

    def _is_relevant(self, ground_truth: str, retrieved_text: str) -> bool:
        """
        判断检索到的文本是否与参考答案相关

        Args:
            ground_truth: 参考答案
            retrieved_text: 检索到的文本

        Returns:
            是否相关
        """
        if not ground_truth or not retrieved_text:
            return False

        # 简化方法：检查是否有较长的公共子串
        gt_sentences = set(s.strip() for s in ground_truth.split("。") if len(s.strip()) > 10)
        ret_sentences = set(s.strip() for s in retrieved_text.split("。") if len(s.strip()) > 10)

        # 检查是否有公共句子
        common = gt_sentences & ret_sentences
        if common:
            return True

        # 检查是否有较长的公共子串（>20字符）
        gt_chars = set(ground_truth[i:i+20] for i in range(len(ground_truth)-19))
        ret_chars = set(retrieved_text[i:i+20] for i in range(len(retrieved_text)-19))
        if gt_chars & ret_chars:
            return True

        return False

    def _compute_ndcg(
        self,
        relevant_indices: List[int],
        retrieved_count: int,
        top_k: int,
        total_ground_truth: int = 0,
    ) -> float:
        """
        计算 NDCG (Normalized Discounted Cumulative Gain)
        使用二元相关性（0/1）

        Args:
            relevant_indices: 相关文档的索引列表
            retrieved_count: 检索到的文档数
            top_k: top-k 值
            total_ground_truth: 总相关文档数（source_node_ids 的长度），用于 IDCG 计算

        Returns:
            NDCG 值
        """
        k = min(top_k, retrieved_count)
        if k == 0:
            return 0.0

        # DCG: 对每个位置计算折损累积增益
        dcg = 0.0
        for i in range(k):
            if i in relevant_indices:
                if i == 0:
                    dcg += 1.0
                else:
                    dcg += 1.0 / math.log2(i + 1)

        # IDCG: 理想情况下的 DCG
        # 基于总相关文档数（total_ground_truth）计算理想排序，
        # 而非基于已命中的相关文档数（len(relevant_indices)）
        ideal_relevant = min(total_ground_truth, k)
        idcg = 0.0
        for i in range(ideal_relevant):
            if i == 0:
                idcg += 1.0
            else:
                idcg += 1.0 / math.log2(i + 1)

        return dcg / max(idcg, 1e-10)

    def _average_metrics(self, all_metrics: List[Dict]) -> RetrievalMetrics:
        """
        计算所有查询的平均指标

        Args:
            all_metrics: 所有查询的指标列表

        Returns:
            RetrievalMetrics 实例
        """
        n = len(all_metrics)
        if n == 0:
            return RetrievalMetrics()

        metrics = RetrievalMetrics()
        metrics.hit_rate = sum(1 for m in all_metrics if m["is_hit"]) / n
        metrics.mrr = sum(m["reciprocal_rank"] for m in all_metrics) / n
        metrics.precision = sum(m["precision"] for m in all_metrics) / n
        metrics.recall = sum(m["recall"] for m in all_metrics) / n
        metrics.ndcg = sum(m["ndcg"] for m in all_metrics) / n

        return metrics

    def evaluate_all_knowledge_bases(
        self,
        datasets: Dict[str, Dict],
        top_k: int = 5,
        output_dir: str = "./eval/reports",
    ) -> Dict[str, Dict]:
        """
        评估所有知识库

        Args:
            datasets: {kb_name: dataset_dict} 格式的评测集字典
            top_k: top-k 值（默认5），支持 int 或 list（如 [1, 3, 5, 10]）
            output_dir: 报告输出目录

        Returns:
            {kb_name: evaluation_result} 格式的评估结果字典
        """
        # 统一转为列表
        top_k_list = [top_k] if isinstance(top_k, int) else top_k

        all_results = {}
        for kb_name, dataset in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"开始评估知识库: {kb_name}")
            logger.info(f"{'='*60}")

            # 对每个 top_k 值分别评估
            kb_results = {}
            for k in top_k_list:
                logger.info(f"  top_k={k} 评估中...")
                result = self.evaluate_knowledge_base(kb_name, dataset, k)
                kb_results[f"top_{k}"] = result

            # 合并结果：保留所有 top_k 的结果
            combined_result = {
                "knowledge_base": kb_name,
                "embedding_model": EMBEDDING_MODEL,
                "dataset_size": len(dataset.get("queries", [])),
                "top_k_results": kb_results,
                "top_k_list": top_k_list,
            }
            all_results[kb_name] = combined_result

            # 保存详细结果
            if output_dir:
                kb_output_dir = os.path.join(output_dir, kb_name)
                os.makedirs(kb_output_dir, exist_ok=True)

                model_short = get_model_short_name(EMBEDDING_MODEL)
                timestamp = get_timestamp()
                result_path = os.path.join(kb_output_dir, f"retrieval_eval_results_{model_short}_{timestamp}.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(combined_result, f, ensure_ascii=False, indent=2)
                logger.info(f"检索评估结果已保存到 {result_path}")

        # 生成汇总对比报告
        self._generate_comparison_report(all_results, output_dir)

        return all_results

    def _generate_comparison_report(
        self,
        all_results: Dict[str, Dict],
        output_dir: str,
    ):
        """
        生成多知识库检索质量汇总对比报告（支持多 top_k 对比）

        Args:
            all_results: 所有知识库的评估结果
            output_dir: 输出目录
        """
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        report_lines = [
            "# 多知识库检索质量评估汇总报告\n",
            f"**嵌入模型**: {EMBEDDING_MODEL}\n",
            f"**评估时间**: {get_timestamp()}\n",
            f"\n## 评估概览\n",
        ]

        for kb_name, result in all_results.items():
            top_k_list = result.get("top_k_list", [5])
            report_lines.append(
                f"- **{kb_name}**: {result.get('dataset_size', 0)} 个查询，"
                f"Top-K 列表: {top_k_list}"
            )

        report_lines.append("\n## 检索质量对比\n")

        # 表头
        header = "| 知识库 | Top-K | Hit Rate | MRR | Precision | Recall | NDCG |"
        separator = "|--------|-------|----------|-----|-----------|--------|------|"
        report_lines.append(header)
        report_lines.append(separator)

        for kb_name, result in all_results.items():
            top_k_results = result.get("top_k_results", {})
            for k_key in sorted(top_k_results.keys(), key=lambda x: int(x.split('_')[1])):
                k_result = top_k_results[k_key]
                metrics = k_result.get("metrics", {})
                top_k = k_result.get("top_k", 5)
                report_lines.append(
                    f"| {kb_name} | {top_k} "
                    f"| {metrics.get('hit_rate', 0):.4f} "
                    f"| {metrics.get('mrr', 0):.4f} "
                    f"| {metrics.get('precision', 0):.4f} "
                    f"| {metrics.get('recall', 0):.4f} "
                    f"| {metrics.get('ndcg', 0):.4f} |"
                )

        report_lines.append("\n## 优化建议\n")
        for kb_name, result in all_results.items():
            top_k_results = result.get("top_k_results", {})
            # 取最大的 top_k 对应的 hit_rate 做建议
            max_k_key = sorted(top_k_results.keys())[-1] if top_k_results else "top_5"
            max_result = top_k_results.get(max_k_key, {})
            metrics = max_result.get("metrics", {})
            hit_rate = metrics.get("hit_rate", 0)

            if hit_rate < 0.7:
                suggestion = "召回率偏低，建议调整分块策略或增加 top_k"
            elif hit_rate < 0.9:
                suggestion = "召回率中等，可考虑优化检索参数"
            else:
                suggestion = "召回率良好，当前策略可维持"

            report_lines.append(f"- **{kb_name}**: Hit Rate={hit_rate:.4f}（top_k={max_result.get('top_k', 5)}）。{suggestion}")

        report = "\n".join(report_lines)

        model_short = get_model_short_name(EMBEDDING_MODEL)
        timestamp = get_timestamp()
        report_path = os.path.join(summary_dir, f"retrieval_comparison_report_{model_short}_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"检索质量汇总报告已保存到 {report_path}")


