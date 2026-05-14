# -*- coding: utf-8 -*-
"""
生成质量评估模块（支持双模式对比）
==============================
功能：
1. 使用 LLM 调用自定义评估 prompt 评估生成质量
   - Correctness - 评估答案正确性
   - Answer Relevancy - 评估回答相关性
   - Faithfulness - 评估忠实度
2. 两种模式评估：
   - 模式A (Agent RAG)：通过 Agent 调用检索工具，基于检索结果生成回答
   - 模式B (LLM Only)：直接调用大语言模型，不提供任何检索上下文
3. 计算对比指标（Δ = RAG_score - LLM_only_score）
4. 支持多知识库独立评估和汇总对比
"""

import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, EVAL_LLM_MODEL, EVAL_LLM_API_KEY, EVAL_LLM_BASE_URL
from configs.prompt_config import (
    correctness_eval_prompt,
    relevancy_eval_prompt,
    faithfulness_eval_prompt,
    recall_eval_prompt,
)
from utils import get_model_short_name, get_timestamp
from eval.agent_adapter import AgentAdapter
from eval.llm_adapter import LLMOnlyAdapter
from eval.llm_wrapper import SimpleLLM
from eval.sql_evaluator import SqlEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """生成质量指标的数据类"""
    correctness: float = 0.0
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    recall: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "correctness": round(self.correctness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "faithfulness": round(self.faithfulness, 4),
            "recall": round(self.recall, 4),
        }

    def overall_score(self) -> float:
        """综合得分（四项指标的均值）"""
        return round((self.correctness + self.answer_relevancy + self.faithfulness + self.recall) / 4, 4)


@dataclass
class ComparisonResult:
    """双模式对比结果的数据类"""
    rag_metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    llm_only_metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    delta_correctness: float = 0.0
    delta_answer_relevancy: float = 0.0
    delta_faithfulness: float = 0.0
    delta_recall: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "rag": self.rag_metrics.to_dict(),
            "llm_only": self.llm_only_metrics.to_dict(),
            "delta": {
                "correctness": round(self.delta_correctness, 4),
                "answer_relevancy": round(self.delta_answer_relevancy, 4),
                "faithfulness": round(self.delta_faithfulness, 4),
                "recall": round(self.delta_recall, 4),
            },
            "rag_overall_score": self.rag_metrics.overall_score(),
            "llm_only_overall_score": self.llm_only_metrics.overall_score(),
            "improvement_pct": self._improvement_percentage(),
        }

    def _improvement_percentage(self) -> float:
        """计算 RAG 相比 LLM Only 的提升百分比"""
        llm_score = self.llm_only_metrics.overall_score()
        if llm_score == 0:
            return 0.0
        return round((self.rag_metrics.overall_score() - llm_score) / llm_score * 100, 2)


class GenerationEvaluator:
    """
    生成质量评估器
    评估 RAG 系统和纯 LLM 的生成质量，并进行对比。
    使用自定义 prompt 评估，兼容 deepseek 等非 OpenAI 模型。
    """

    def __init__(self, llm=None, judge_llm=None):
        """
        初始化生成质量评估器

        Args:
            llm: LLM 实例，用于生成回答（RAG 和 LLM Only 共用）
            judge_llm: LLM 实例，用于打分评估（LLM as Judge）。若为 None 则与 llm 相同
        """
        self.llm = llm or self._create_llm(LLM_MODEL, LLM_API_KEY, LLM_BASE_URL)
        self.judge_llm = judge_llm or self._create_llm(EVAL_LLM_MODEL, EVAL_LLM_API_KEY, EVAL_LLM_BASE_URL)

        if self.judge_llm is self.llm:
            logger.info(f"生成和评估使用同一模型: {LLM_MODEL}")
        else:
            logger.info(f"生成模型: {LLM_MODEL}，评估打分模型: {EVAL_LLM_MODEL}")

        # 初始化适配器（生成回答用 self.llm）
        self.agent_adapter = AgentAdapter(llm=self.llm)
        self.llm_only_adapter = LLMOnlyAdapter(llm=self.llm)

        # SQL 评估器（仅用于 project NL2SQL，共享 judge_llm 打分）
        self.sql_evaluator = SqlEvaluator()

    @staticmethod
    def _create_llm(model: str, api_key: str, base_url: str) -> SimpleLLM:
        """创建 LLM 实例"""
        logger.info(f"初始化 LLM，模型: {model}")
        return SimpleLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )

    def evaluate(
        self,
        kb_name: str,
        dataset: Dict,
        similarity_top_k: int = 5,
    ) -> Dict:
        """
        对指定知识库执行生成质量评估（双模式对比）

        Args:
            kb_name: 知识库名称
            dataset: 评测集字典
            similarity_top_k: 检索的 top-k 参数

        Returns:
            评估结果字典
        """
        queries = dataset.get("queries", [])
        if not queries:
            logger.warning(f"知识库 {kb_name} 的评测集为空")
            return {}

        logger.info(f"开始生成质量评估 - 知识库: {kb_name}, 查询数: {len(queries)}")

        # 存储每个查询的评估结果
        rag_results = []
        llm_only_results = []

        for q_data in tqdm(queries, desc=f"  Evaluating generation for {kb_name}"):
            query = q_data["query"]
            ground_truth = q_data.get("ground_truth_answer", "")
            qid = q_data.get("qid", 0)

            # ---- 模式A: Agent RAG ----
            rag_answer, retrieval_result = self.agent_adapter.query(
                query, kb_name, similarity_top_k
            )
            rag_context = "\n\n".join(retrieval_result.contexts) if retrieval_result.contexts else ""

            # ---- 模式B: LLM Only ----
            llm_only_answer = self.llm_only_adapter.generate_answer(query)

            # ---- 评估 RAG 回答 ----
            rag_metrics = self._evaluate_single(
                query=query,
                response=rag_answer,
                contexts=[rag_context],
                reference=ground_truth,
            )

            # ---- 评估 LLM Only 回答 ----
            llm_only_metrics = self._evaluate_single(
                query=query,
                response=llm_only_answer,
                contexts=[],  # 无上下文
                reference=ground_truth,
            )

            # ---- SQL 评估（仅 project） ----
            sql_metrics = {}
            if kb_name == "project":
                # 重新执行期望 SQL，避免预存结果过期导致评估失真
                fresh_expected = self._execute_expected_sql(q_data.get("expected_sql", ""))
                if fresh_expected is not None:
                    logger.info(f"QID {q_data.get('qid', '?')}: 使用新执行结果 ({len(fresh_expected)} 行)")
                expected_results = fresh_expected if fresh_expected is not None else q_data.get("expected_results", None)
                sql_metrics = self.sql_evaluator.evaluate_single(
                    query=query,
                    sql=retrieval_result.sql,
                    sql_error=retrieval_result.sql_error,
                    row_count=retrieval_result.row_count,
                    expected_sql=q_data.get("expected_sql", ""),
                    expected_results=expected_results,
                    generated_results=retrieval_result.results,
                )

            # 记录详细结果
            rag_entry = {
                "qid": qid,
                "query": query,
                "sql": retrieval_result.sql if kb_name == "project" else "",
                "sql_error": retrieval_result.sql_error if kb_name == "project" else "",
                "expected_sql": q_data.get("expected_sql", "") if kb_name == "project" else "",
                "expected_results_count": len(retrieval_result.results) if kb_name == "project" else 0,
                "context": rag_context,
                "response": rag_answer,
                "metrics": rag_metrics.to_dict(),
            }
            if sql_metrics:
                rag_entry["sql_metrics"] = sql_metrics
            rag_results.append(rag_entry)

            llm_only_results.append({
                "qid": qid,
                "query": query,
                "response": llm_only_answer,
                "metrics": llm_only_metrics.to_dict(),
            })

        # 汇总指标
        comparison = self._aggregate_comparison(rag_results, llm_only_results)

        # 汇总 SQL 评估指标
        sql_metrics_summary = {}
        if kb_name == "project":
            sql_metrics_list = [
                r.get("sql_metrics", {}) for r in rag_results if "sql_metrics" in r
            ]
            if sql_metrics_list:
                sql_metrics_summary = SqlEvaluator.aggregate_metrics(sql_metrics_list)

        result = {
            "knowledge_base": kb_name,
            "llm_model": LLM_MODEL,
            "dataset_size": len(queries),
            "comparison": comparison.to_dict(),
            "sql_metrics": sql_metrics_summary,
            "detailed_results": {
                "rag": rag_results,
                "llm_only": llm_only_results,
            },
        }

        return result

    def _evaluate_single(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: str,
    ) -> GenerationMetrics:
        """
        评估单个回答的质量（使用 LLM 调用自定义 prompt）

        Args:
            query: 用户问题
            response: 生成的回答
            contexts: 检索到的上下文列表
            reference: 参考答案

        Returns:
            GenerationMetrics 实例
        """
        metrics = GenerationMetrics()

        try:
            # 1. Correctness: 与参考答案的匹配程度
            if reference:
                metrics.correctness = self._evaluate_correctness(query, response, reference)
            else:
                metrics.correctness = 0.0
        except Exception as e:
            logger.warning(f"Correctness 评估失败: {e}")
            metrics.correctness = 0.0

        try:
            # 2. Answer Relevancy: 回答与问题的相关程度
            metrics.answer_relevancy = self._evaluate_relevancy(query, response)
        except Exception as e:
            logger.warning(f"Relevancy 评估失败: {e}")
            metrics.answer_relevancy = 0.0

        try:
            # 3. Faithfulness: 回答是否忠实于上下文
            if contexts and any(c.strip() for c in contexts):
                metrics.faithfulness = self._evaluate_faithfulness(query, response, contexts)
            else:
                # 无上下文时，faithfulness 基于模型自身知识
                metrics.faithfulness = 0.5  # 中性值
        except Exception as e:
            logger.warning(f"Faithfulness 评估失败: {e}")
            metrics.faithfulness = 0.0

        try:
            # 4. Recall: 回答对参考答案信息点的覆盖程度
            if reference:
                metrics.recall = self._evaluate_recall(query, response, reference)
            else:
                metrics.recall = 0.0
        except Exception as e:
            logger.warning(f"Recall 评估失败: {e}")
            metrics.recall = 0.0

        return metrics

    def _evaluate_correctness(self, query: str, response: str, reference: str) -> float:
        """
        评估回答的正确性（与参考答案对比）

        Args:
            query: 用户问题
            response: 生成的回答
            reference: 参考答案

        Returns:
            0.0 ~ 1.0 的分数
        """
        prompt = correctness_eval_prompt(query, response, reference)
        try:
            result = self.judge_llm.complete(prompt)
            score_text = result.text.strip()
            # 提取数字
            score = self._extract_score(score_text)
            return score / 100.0
        except Exception as e:
            logger.warning(f"Correctness LLM 调用失败: {e}")
            return 0.0

    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """
        评估回答与问题的相关性

        Args:
            query: 用户问题
            response: 生成的回答

        Returns:
            0.0 ~ 1.0 的分数
        """
        prompt = relevancy_eval_prompt(query, response)
        try:
            result = self.judge_llm.complete(prompt)
            score_text = result.text.strip()
            score = self._extract_score(score_text)
            return score / 100.0
        except Exception as e:
            logger.warning(f"Relevancy LLM 调用失败: {e}")
            return 0.0

    def _execute_expected_sql(self, sql: str) -> Optional[List[Dict]]:
        """重新执行期望 SQL，返回最新结果。失败时返回 None 触发预存结果兜底。"""
        if not sql or not sql.strip():
            return None
        try:
            import sqlite3
            from configs.config import PROJECT_SQLITE_DB
            conn = sqlite3.connect(PROJECT_SQLITE_DB)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()
            conn.close()
            results = [dict(r) for r in rows]
            logger.info(f"重新执行期望 SQL 成功，返回 {len(results)} 行")
            return results
        except Exception as e:
            logger.warning(f"重新执行期望 SQL 失败，使用预存结果兜底: {e}")
            return None

    def _evaluate_faithfulness(self, query: str, response: str, contexts: List[str]) -> float:
        """
        评估回答是否忠实于提供的上下文（无幻觉）

        Args:
            query: 用户问题
            response: 生成的回答
            contexts: 提供的上下文列表

        Returns:
            0.0 ~ 1.0 的分数
        """
        prompt = faithfulness_eval_prompt(query, response, "\n\n".join(contexts))
        try:
            result = self.judge_llm.complete(prompt)
            score_text = result.text.strip()
            score = self._extract_score(score_text)
            return score / 100.0
        except Exception as e:
            logger.warning(f"Faithfulness LLM 调用失败: {e}")
            return 0.0

    def _evaluate_recall(self, query: str, response: str, reference: str) -> float:
        """
        评估回答对参考答案中关键信息点的覆盖程度（生成召回率）

        关注：参考答案中的关键事实/要点，回答是否都提到了。
        与 Correctness 的区别：Correctness 混入准确性判断（答错扣分），
        Recall 只关注「该说的都说了没有」，不因多说了而扣分。

        Args:
            query: 用户问题
            response: 生成的回答
            reference: 参考答案

        Returns:
            0.0 ~ 1.0 的分数
        """
        prompt = recall_eval_prompt(query, response, reference)
        try:
            result = self.judge_llm.complete(prompt)
            score_text = result.text.strip()
            score = self._extract_score(score_text)
            return score / 100.0
        except Exception as e:
            logger.warning(f"Recall LLM 调用失败: {e}")
            return 0.0

    def _extract_score(self, text: str) -> float:
        """
        从 LLM 输出中提取分数

        Args:
            text: LLM 输出的文本

        Returns:
            提取的分数（0-100）
        """
        import re
        # 尝试匹配数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            score = float(numbers[0])
            # 确保在 0-100 范围内
            return max(0.0, min(100.0, score))
        return 0.0

    def _aggregate_comparison(
        self,
        rag_results: List[Dict],
        llm_only_results: List[Dict],
    ) -> ComparisonResult:
        """
        汇总所有查询的评估结果，计算对比指标

        Args:
            rag_results: RAG 模式的评估结果列表
            llm_only_results: LLM Only 模式的评估结果列表

        Returns:
            ComparisonResult 实例
        """
        n = len(rag_results)
        if n == 0:
            return ComparisonResult()

        # 计算平均指标
        rag_metrics = GenerationMetrics()
        llm_only_metrics = GenerationMetrics()

        for i in range(n):
            rag_m = rag_results[i]["metrics"]
            llm_m = llm_only_results[i]["metrics"]

            rag_metrics.correctness += rag_m.get("correctness", 0)
            rag_metrics.answer_relevancy += rag_m.get("answer_relevancy", 0)
            rag_metrics.faithfulness += rag_m.get("faithfulness", 0)
            rag_metrics.recall += rag_m.get("recall", 0)

            llm_only_metrics.correctness += llm_m.get("correctness", 0)
            llm_only_metrics.answer_relevancy += llm_m.get("answer_relevancy", 0)
            llm_only_metrics.faithfulness += llm_m.get("faithfulness", 0)
            llm_only_metrics.recall += llm_m.get("recall", 0)

        rag_metrics.correctness /= n
        rag_metrics.answer_relevancy /= n
        rag_metrics.faithfulness /= n
        rag_metrics.recall /= n

        llm_only_metrics.correctness /= n
        llm_only_metrics.answer_relevancy /= n
        llm_only_metrics.faithfulness /= n
        llm_only_metrics.recall /= n

        # 计算差值
        comparison = ComparisonResult(
            rag_metrics=rag_metrics,
            llm_only_metrics=llm_only_metrics,
            delta_correctness=rag_metrics.correctness - llm_only_metrics.correctness,
            delta_answer_relevancy=rag_metrics.answer_relevancy - llm_only_metrics.answer_relevancy,
            delta_faithfulness=rag_metrics.faithfulness - llm_only_metrics.faithfulness,
            delta_recall=rag_metrics.recall - llm_only_metrics.recall,
        )

        return comparison

    def evaluate_all_knowledge_bases(
        self,
        datasets: Dict[str, Dict],
        similarity_top_k: int = 5,
        output_dir: str = "./eval/reports",
    ) -> Dict[str, Dict]:
        """
        评估所有知识库的生成质量

        Args:
            datasets: {kb_name: dataset_dict} 格式的评测集字典
            similarity_top_k: 检索的 top-k 参数
            output_dir: 报告输出目录

        Returns:
            {kb_name: evaluation_result} 格式的评估结果字典
        """
        all_results = {}
        for kb_name, dataset in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"开始生成质量评估 - 知识库: {kb_name}")
            logger.info(f"{'='*60}")

            result = self.evaluate(kb_name, dataset, similarity_top_k)
            all_results[kb_name] = result

            # 保存详细结果
            if output_dir:
                kb_output_dir = os.path.join(output_dir, kb_name)
                os.makedirs(kb_output_dir, exist_ok=True)

                model_short = get_model_short_name(LLM_MODEL)
                timestamp = get_timestamp()
                result_path = os.path.join(kb_output_dir, f"generation_eval_results_{model_short}_{timestamp}.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"生成评估结果已保存到 {result_path}")

        # 生成汇总对比报告
        self._generate_comparison_report(all_results, output_dir)

        return all_results

    def _generate_comparison_report(
        self,
        all_results: Dict[str, Dict],
        output_dir: str,
    ):
        """
        生成多知识库生成质量汇总对比报告

        Args:
            all_results: 所有知识库的评估结果
            output_dir: 输出目录
        """
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        report_lines = [
            "# 多知识库生成质量评估汇总报告（RAG vs LLM Only）\n",
            f"**生成模型**: {LLM_MODEL}\n",
            f"**打分模型**: {self.judge_llm.model}\n",
            f"**评估时间**: {get_timestamp()}\n",
            f"\n## 评估概览\n",
        ]

        for kb_name, result in all_results.items():
            report_lines.append(f"- **{kb_name}**: {result.get('dataset_size', 0)} 个查询")

        report_lines.append("\n## 生成质量对比\n")
        report_lines.append(
            "| 知识库 | 模式 | Correctness | Relevancy | Faithfulness | Recall | 综合得分 |"
        )
        report_lines.append(
            "|--------|------|-------------|-----------|--------------|--------|----------|"
        )

        for kb_name, result in all_results.items():
            comparison = result.get("comparison", {})
            rag = comparison.get("rag", {})
            llm_only = comparison.get("llm_only", {})
            delta = comparison.get("delta", {})
            rag_overall = comparison.get("rag_overall_score", 0)
            llm_only_overall = comparison.get("llm_only_overall_score", 0)
            improvement = comparison.get("improvement_pct", 0)

            # RAG 行
            report_lines.append(
                f"| {kb_name} | RAG "
                f"| {rag.get('correctness', 0):.4f} "
                f"| {rag.get('answer_relevancy', 0):.4f} "
                f"| {rag.get('faithfulness', 0):.4f} "
                f"| {rag.get('recall', 0):.4f} "
                f"| {rag_overall:.4f} |"
            )

            # LLM Only 行
            report_lines.append(
                f"| {kb_name} | LLM "
                f"| {llm_only.get('correctness', 0):.4f} "
                f"| {llm_only.get('answer_relevancy', 0):.4f} "
                f"| {llm_only.get('faithfulness', 0):.4f} "
                f"| {llm_only.get('recall', 0):.4f} "
                f"| {llm_only_overall:.4f} |"
            )

            # 差值行
            delta_c = delta.get("correctness", 0)
            delta_r = delta.get("answer_relevancy", 0)
            delta_f = delta.get("faithfulness", 0)
            delta_rec = delta.get("recall", 0)
            delta_str_c = f"+{delta_c:.4f}" if delta_c >= 0 else f"{delta_c:.4f}"
            delta_str_r = f"+{delta_r:.4f}" if delta_r >= 0 else f"{delta_r:.4f}"
            delta_str_f = f"+{delta_f:.4f}" if delta_f >= 0 else f"{delta_f:.4f}"
            delta_str_rec = f"+{delta_rec:.4f}" if delta_rec >= 0 else f"{delta_rec:.4f}"

            report_lines.append(
                f"| {kb_name} | 差值 "
                f"| **{delta_str_c}** "
                f"| **{delta_str_r}** "
                f"| **{delta_str_f}** "
                f"| **{delta_str_rec}** "
                f"| **{improvement:+.2f}%** |"
            )

        # RAG 提升效果总结
        report_lines.append("\n## RAG 提升效果总结\n")
        all_correctness_deltas = []
        all_faithfulness_deltas = []
        all_recall_deltas = []
        all_improvements = []

        for kb_name, result in all_results.items():
            comparison = result.get("comparison", {})
            delta = comparison.get("delta", {})
            improvement = comparison.get("improvement_pct", 0)

            all_correctness_deltas.append(delta.get("correctness", 0))
            all_faithfulness_deltas.append(delta.get("faithfulness", 0))
            all_recall_deltas.append(delta.get("recall", 0))
            all_improvements.append(improvement)

        if all_correctness_deltas:
            avg_correctness_delta = np.mean(all_correctness_deltas)
            avg_faithfulness_delta = np.mean(all_faithfulness_deltas)
            avg_recall_delta = np.mean(all_recall_deltas)
            best_kb = max(all_results.keys(), key=lambda k: all_results[k].get("comparison", {}).get("improvement_pct", 0))
            best_improvement = max(all_improvements)

            report_lines.append(f"- 平均 Correctness 提升: {avg_correctness_delta:+.4f}")
            report_lines.append(f"- 平均 Faithfulness 提升: {avg_faithfulness_delta:+.4f} (RAG 在减少幻觉方面优势明显)")
            report_lines.append(f"- 平均 Recall 提升: {avg_recall_delta:+.4f} (RAG 对信息覆盖完整性的提升)")
            report_lines.append(f"- 最佳提升知识库: {best_kb} ({best_improvement:+.2f}%)")

        # SQL 评估指标
        has_sql_metrics = any(
            result.get("sql_metrics") for result in all_results.values()
        )
        if has_sql_metrics:
            report_lines.append("\n---")
            report_lines.append("## SQL 生成质量评估\n")
            report_lines.append(
                "| 知识库 | VE | EX | CM | SIM | 综合得分 |"
            )
            report_lines.append(
                "|--------|----|----|----|-----|---------|"
            )
            for kb_name, result in all_results.items():
                sql_m = result.get("sql_metrics", {})
                if sql_m:
                    report_lines.append(
                        f"| {kb_name} "
                        f"| {sql_m.get('ve', 0):.4f} "
                        f"| {sql_m.get('ex', 0):.4f} "
                        f"| {sql_m.get('cm', 0):.4f} "
                        f"| {sql_m.get('sim', 0):.4f} "
                        f"| {sql_m.get('overall', 0):.4f} |"
                    )
            report_lines.append(
                "\nCM 分项（组件匹配率）：\n"
            )
            for kb_name, result in all_results.items():
                sql_m = result.get("sql_metrics", {})
                if sql_m:
                    report_lines.append(
                        f"- **{kb_name}**: SELECT={sql_m.get('cm_select', 0):.2f}  "
                        f"WHERE={sql_m.get('cm_where', 0):.2f}  "
                        f"GROUP={sql_m.get('cm_group', 0):.2f}  "
                        f"ORDER={sql_m.get('cm_order', 0):.2f}  "
                        f"KEYWORDS={sql_m.get('cm_keywords', 0):.2f}"
                    )
            report_lines.append(
                "\n指标说明：\n"
                "- **VE** (Valid Execution): SQL 能否无错解析并执行\n"
                "- **EX** (Execution Accuracy): 执行结果与标准结果是否一致（忽略行序）\n"
                "- **CM** (Component Match): SELECT / WHERE / GROUP / ORDER / KEYWORDS 五组件加权匹配率\n"
                "- **SIM** (SQL Similarity): 叶子节点（表名/列名/字面值/算子）Jaccard 相似度\n"
            )

            # 逐条 SQL 详情（仅 project）
            for kb_name, result in all_results.items():
                if kb_name != "project":
                    continue
                rag_details = result.get("detailed_results", {}).get("rag", [])
                if not rag_details:
                    continue

                report_lines.append("\n### SQL 详情（逐条）\n")
                report_lines.append(
                    "| QID | 问题 | 生成 SQL | 期望 SQL | 行数 | VE | EX | CM | SIM |"
                )
                report_lines.append(
                    "|-----|------|---------|---------|------|----|----|----|-----|"
                )
                for r in rag_details:
                    sql = r.get("sql", "")
                    expected_sql = r.get("expected_sql", "")
                    sm = r.get("sql_metrics", {})
                    report_lines.append(
                        f"| {r['qid']} "
                        f"| {r['query']} "
                        f"| `{sql}` "
                        f"| `{expected_sql}` "
                        f"| {r.get('expected_results_count', 0)} "
                        f"| {sm.get('ve', 0):.2f} "
                        f"| {sm.get('ex', 0):.2f} "
                        f"| {sm.get('cm', 0):.2f} "
                        f"| {sm.get('sim', 0):.4f} |"
                    )
                report_lines.append("")

        # 优化建议
        report_lines.append("\n## 优化建议（按知识库）\n")
        for kb_name, result in all_results.items():
            comparison = result.get("comparison", {})
            improvement = comparison.get("improvement_pct", 0)
            rag_overall = comparison.get("rag_overall_score", 0)
            llm_only_overall = comparison.get("llm_only_overall_score", 0)

            if improvement < 5:
                suggestion = "RAG 提升效果不明显，建议检查检索质量或调整分块策略"
            elif improvement < 15:
                suggestion = "RAG 有一定提升，可考虑优化检索参数进一步提升"
            else:
                suggestion = "RAG 提升效果显著，可考虑推广当前检索策略"

            report_lines.append(
                f"- **{kb_name}**: RAG 综合得分 {rag_overall:.4f} vs LLM Only {llm_only_overall:.4f} "
                f"(提升 {improvement:+.2f}%)。{suggestion}"
            )

        report = "\n".join(report_lines)

        model_short = get_model_short_name(LLM_MODEL)
        timestamp = get_timestamp()
        report_path = os.path.join(summary_dir, f"generation_comparison_report_{model_short}_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"生成质量汇总报告已保存到 {report_path}")


