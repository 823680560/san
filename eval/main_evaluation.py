# -*- coding: utf-8 -*-
"""
主评估流程（遍历知识库 + 双模式对比）
==============================
功能：
1. 加载配置和评测集
2. 遍历所有知识库执行检索评估和生成评估
3. 生成各知识库独立报告和汇总对比报告
4. 支持命令行参数配置
"""

import json
import os
import sys
import logging
import argparse
import time
from typing import Dict, List, Optional
from datetime import datetime

import yaml

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 确保 reports 目录存在
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

from llama_index.core import Settings

from configs.config import (
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, EVAL_LLM_MODEL, EVAL_LLM_API_KEY, EVAL_LLM_BASE_URL,
    EMBEDDING_MODEL, EMBEDDING_SERVICE, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_NORMALIZE, EMBEDDING_BATCH_SIZE, resolve_model_path,
)
from utils import get_model_short_name, get_timestamp
from eval.dataset_builder import DatasetBuilder, SQLBasedDatasetBuilder
from eval.agent_adapter import AgentAdapter
from eval.retrieval_evaluator import RetrievalEvaluator
from eval.generation_evaluator import GenerationEvaluator
from eval.llm_wrapper import SimpleLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(REPORTS_DIR, "evaluation.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    评估流水线
    协调整个评估流程：数据集构建 -> 检索评估 -> 生成评估 -> 报告生成
    """

    def __init__(self, config_path: str = None):
        """
        初始化评估流水线

        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "eval", "knowledge_base_config.yaml")
        self.config = self._load_config(config_path)
        self._setup_llm_and_embedding()

        # 初始化各模块
        self.dataset_builder = DatasetBuilder(llm=self.llm)
        self.agent_adapter = AgentAdapter(llm=self.llm)
        self.retrieval_evaluator = RetrievalEvaluator(self.agent_adapter, llm=self.llm)
        self.generation_evaluator = GenerationEvaluator(llm=self.llm, judge_llm=self.judge_llm)

    def _load_config(self, config_path: str) -> Dict:
        """
        加载 YAML 配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        if not os.path.exists(config_path):
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return self._default_config()

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config

    def _default_config(self) -> Dict:
        """返回默认配置"""
        return {
            "retrieval": {"top_k": [1, 3, 5, 10]},
            "generation": {"compare_modes": True},
            "dataset": {"questions_per_chunk": 1, "test_size": 20},
            "knowledge_bases": [
                {"name": "law", "enabled": True},
                {"name": "project", "enabled": True},
            ],
        }

    def _setup_llm_and_embedding(self):
        """设置 LLM 和 Embedding 模型"""
        # 设置 Embedding 模型（与 agent_adapter / dataset_builder 一致的配置）
        if Settings._embed_model is None:
            logger.info(f"初始化嵌入模型: {EMBEDDING_MODEL} (service={EMBEDDING_SERVICE})")
            if EMBEDDING_SERVICE == "ollama":
                from llama_index.embeddings.ollama import OllamaEmbedding
                Settings.embed_model = OllamaEmbedding(
                    model_name=EMBEDDING_MODEL,
                    normalize=EMBEDDING_NORMALIZE,
                    embed_batch_size=EMBEDDING_BATCH_SIZE,
                )
            elif EMBEDDING_SERVICE == "huggingface":
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                model_path = resolve_model_path(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)
                Settings.embed_model = HuggingFaceEmbedding(model_name=model_path, embed_batch_size=EMBEDDING_BATCH_SIZE)
            else:
                raise ValueError(f"不支持的嵌入模型服务: {EMBEDDING_SERVICE}")
            logger.info(f"嵌入模型初始化完成: {EMBEDDING_MODEL}")

        # 设置 LLM（使用 SimpleLLM 包装器，兼容 deepseek 等非 OpenAI 模型）
        # 注意：不设置 Settings.llm，因为 SimpleLLM 不是 llama_index 的 LLM 类型，
        # Settings.llm 已在 AgentAdapter._ensure_embed_model() 中被设置为 MockLLM
        self.llm = SimpleLLM(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
        )

        # 设置评估打分专用 LLM（可与生成模型不同，用于 LLM as Judge）
        self.judge_llm = SimpleLLM(
            model=EVAL_LLM_MODEL,
            api_key=EVAL_LLM_API_KEY,
            base_url=EVAL_LLM_BASE_URL,
            temperature=0,
        )

        if LLM_MODEL == EVAL_LLM_MODEL:
            logger.info(f"生成和评估使用同一模型: {LLM_MODEL}")
        else:
            logger.info(f"生成模型: {LLM_MODEL}，评估打分模型: {EVAL_LLM_MODEL}")
        logger.info("LLM 和 Embedding 模型初始化完成")

    def get_enabled_knowledge_bases(self) -> List[str]:
        """获取启用的知识库列表"""
        kb_configs = self.config.get("knowledge_bases", [])
        return [kb["name"] for kb in kb_configs if kb.get("enabled", True)]

    def build_datasets(self, kb_list: List[str] = None) -> Dict[str, Dict]:
        """
        构建评测集

        Args:
            kb_list: 知识库名称列表，为 None 时使用配置中启用的知识库

        Returns:
            {kb_name: dataset_dict} 格式的评测集字典
        """
        if kb_list is None:
            kb_list = self.get_enabled_knowledge_bases()
        dataset_config = self.config.get("dataset", {})

        logger.info(f"\n{'='*60}")
        logger.info("开始构建评测集")
        logger.info(f"{'='*60}")

        datasets_dir = os.path.join(PROJECT_ROOT, "eval", "datasets")

        # project 知识库使用 SQL 模版驱动的 NL2SQL 评测集，不走旧的分块式构建
        sql_kbs = [k for k in kb_list if k == "project"]
        chunk_kbs = [k for k in kb_list if k != "project"]

        # 为 law 等非 project 知识库构建旧式的分块式评测集
        if chunk_kbs:
            datasets = self.dataset_builder.build_all_datasets(
                kb_list=chunk_kbs,
                questions_per_chunk=dataset_config.get("questions_per_chunk", 1),
                test_size=dataset_config.get("test_size", 20),
                output_dir=datasets_dir,
            )
        else:
            datasets = {}

        # 为 project 知识库构建 NL2SQL 评测集
        for kb_name in sql_kbs:
            logger.info(f"\n使用 SQLBasedDatasetBuilder 为知识库 {kb_name} 生成 NL2SQL 评测集...")

            # 尝试加载 ProjectQueryEngine (Vanna) 以生成高质量的 expected_sql
            engine = None
            try:
                from server.rag.project_query_engine import ProjectQueryEngine
                engine = ProjectQueryEngine()
                logger.info("已加载 ProjectQueryEngine，将使用 Vanna 生成 expected_sql")
            except Exception as e:
                logger.warning(f"无法加载 ProjectQueryEngine ({e})，将退回模板模式")

            sql_builder = SQLBasedDatasetBuilder(llm=self.llm, engine=engine)
            nl2sql_dataset = sql_builder.build_dataset(
                test_size=dataset_config.get("test_size", 20),
                output_dir=datasets_dir,
            )
            if nl2sql_dataset:
                datasets[kb_name] = nl2sql_dataset

        logger.info(f"评测集构建完成，共 {len(datasets)} 个知识库")
        return datasets

    def load_datasets(self) -> Dict[str, Dict]:
        """
        从文件加载已存在的评测集

        Returns:
            {kb_name: dataset_dict} 格式的评测集字典
        """
        kb_list = self.get_enabled_knowledge_bases()
        datasets = {}
        datasets_dir = os.path.join(PROJECT_ROOT, "eval", "datasets")

        for kb_name in kb_list:
            # project 知识库优先加载 NL2SQL 评测集
            if kb_name == "project":
                nl2sql_path = os.path.join(datasets_dir, "project_nl2sql_eval_dataset.json")
                if os.path.exists(nl2sql_path):
                    with open(nl2sql_path, "r", encoding="utf-8") as f:
                        datasets[kb_name] = json.load(f)
                    logger.info(f"已加载 NL2SQL 评测集: {nl2sql_path}")
                    continue

            dataset_path = os.path.join(datasets_dir, f"{kb_name}_eval_dataset.json")
            if os.path.exists(dataset_path):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    datasets[kb_name] = json.load(f)
                logger.info(f"已加载评测集: {dataset_path}")
            else:
                logger.warning(f"评测集文件不存在: {dataset_path}")

        return datasets

    def run_retrieval_evaluation(
        self, datasets: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        执行检索质量评估（支持多 top_k 对比）

        Args:
            datasets: 评测集字典

        Returns:
            检索评估结果字典
        """
        retrieval_config = self.config.get("retrieval", {})
        top_k = retrieval_config.get("top_k", 5)

        logger.info(f"\n{'='*60}")
        logger.info("开始检索质量评估")
        logger.info(f"Top-K 配置: {top_k}")
        logger.info(f"{'='*60}")

        # project 是 NL2SQL，跳过检索评估（无节点召回，指标恒为 0）
        retrieval_datasets = {k: v for k, v in datasets.items() if k != "project"}

        if not retrieval_datasets:
            logger.info("没有需要检索评估的知识库，跳过")
            return {}

        results = self.retrieval_evaluator.evaluate_all_knowledge_bases(
            datasets=retrieval_datasets,
            top_k=top_k,
            output_dir=REPORTS_DIR,
        )

        logger.info("检索质量评估完成")
        return results

    def run_generation_evaluation(
        self, datasets: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        执行生成质量评估（双模式对比）

        Args:
            datasets: 评测集字典

        Returns:
            生成评估结果字典
        """
        generation_config = self.config.get("generation", {})
        retrieval_top_k = self.config.get("retrieval", {}).get("top_k", 5)
        # generation_evaluator 需要整数 top_k，取列表中的最大值
        if isinstance(retrieval_top_k, list):
            similarity_top_k = max(retrieval_top_k)
        else:
            similarity_top_k = retrieval_top_k

        logger.info(f"\n{'='*60}")
        logger.info("开始生成质量评估（双模式对比）")
        logger.info(f"{'='*60}")

        results = self.generation_evaluator.evaluate_all_knowledge_bases(
            datasets=datasets,
            similarity_top_k=similarity_top_k,
            output_dir=REPORTS_DIR,
        )

        logger.info("生成质量评估完成")
        return results

    def generate_final_report(
        self,
        retrieval_results: Dict[str, Dict],
        generation_results: Dict[str, Dict],
        datasets: Dict[str, Dict] = None,
    ):
        """
        生成最终的综合评估报告

        Args:
            retrieval_results: 检索评估结果
            generation_results: 生成评估结果
            datasets: 评测集字典（用于获取实际大小）
        """
        summary_dir = os.path.join(REPORTS_DIR, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        # 计算各知识库的实际评测集大小
        actual_sizes = {}
        if datasets:
            for kb_name, dataset in datasets.items():
                queries = dataset.get("queries", [])
                actual_sizes[kb_name] = len(queries)

        report_lines = [
            f"# 综合评估报告\n",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**评估配置**:",
        ]

        # 配置信息
        dataset_config = self.config.get("dataset", {})
        retrieval_config = self.config.get("retrieval", {})
        generation_config = self.config.get("generation", {})

        if actual_sizes:
            size_str = ", ".join(f"{kb}: {n}" for kb, n in actual_sizes.items())
            report_lines.append(f"- 实际评测集: {size_str}")
        else:
            report_lines.append(f"- 评测集大小: {dataset_config.get('test_size', 20)} 个查询/知识库")
        report_lines.append(f"- 嵌入模型: {EMBEDDING_MODEL}")
        report_lines.append(f"- 检索 Top-K: {retrieval_config.get('top_k', [1, 3, 5, 10])}")
        report_lines.append(f"- 生成模型: {LLM_MODEL}")
        report_lines.append(f"- 对比模式: {'启用' if generation_config.get('compare_modes', True) else '禁用'}")
        report_lines.append("")

        # 检索质量汇总
        report_lines.append("---")
        report_lines.append("## 一、检索质量评估\n")

        if retrieval_results:
            report_lines.append("| 知识库 | Top-K | Hit Rate | MRR | Precision | Recall | NDCG |")
            report_lines.append("|--------|-------|----------|-----|-----------|--------|------|")

            for kb_name, result in retrieval_results.items():
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
        else:
            report_lines.append("（未执行检索评估）")

        # 生成质量汇总
        report_lines.append("\n---")
        report_lines.append("## 二、生成质量评估（RAG vs LLM Only）\n")

        if generation_results:
            report_lines.append(
                "| 知识库 | 模式 | Correctness | Relevancy | Faithfulness | Recall | 综合得分 |"
            )
            report_lines.append(
                "|--------|------|-------------|-----------|--------------|--------|----------|"
            )

            for kb_name, result in generation_results.items():
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
                report_lines.append(
                    f"| {kb_name} | 差值 "
                    f"| **{delta_c:+.4f}** "
                    f"| **{delta_r:+.4f}** "
                    f"| **{delta_f:+.4f}** "
                    f"| **{delta_rec:+.4f}** "
                    f"| **{improvement:+.2f}%** |"
                )
        else:
            report_lines.append("（未执行生成评估）")

        # 优化建议
        report_lines.append("\n---")
        report_lines.append("## 三、优化建议\n")

        if retrieval_results:
            for kb_name, result in retrieval_results.items():
                top_k_results = result.get("top_k_results", {})
                # 取最大的 top_k 对应的 hit_rate 做建议
                max_k_key = sorted(top_k_results.keys())[-1] if top_k_results else "top_5"
                max_result = top_k_results.get(max_k_key, {})
                metrics = max_result.get("metrics", {})
                hit_rate = metrics.get("hit_rate", 0)
                top_k = max_result.get("top_k", 5)

                if hit_rate < 0.7:
                    suggestion = "召回率偏低，建议调整分块策略或增加 top_k"
                elif hit_rate < 0.9:
                    suggestion = "召回率中等，可考虑优化检索参数"
                else:
                    suggestion = "召回率良好，当前策略可维持"

                report_lines.append(
                    f"- **{kb_name}知识库**: Hit Rate={hit_rate:.4f} @ top_k={top_k}。{suggestion}"
                )

        if generation_results:
            for kb_name, result in generation_results.items():
                comparison = result.get("comparison", {})
                improvement = comparison.get("improvement_pct", 0)
                rag_overall = comparison.get("rag_overall_score", 0)
                rag = comparison.get("rag", {})
                delta = comparison.get("delta", {})
                rag_recall = rag.get("recall", 0)
                delta_recall = delta.get("recall", 0)
                delta_correctness = delta.get("correctness", 0)

                # 综合提升建议
                if improvement < 5:
                    suggestion = "RAG 提升效果不明显，建议检查检索质量或调整分块策略"
                elif improvement < 15:
                    suggestion = "RAG 有一定提升，可考虑优化检索参数进一步提升"
                else:
                    suggestion = "RAG 提升效果显著，可考虑推广当前检索策略"

                report_lines.append(
                    f"- **{kb_name}知识库**: RAG 综合得分 {rag_overall:.4f}，提升 {improvement:+.2f}%。{suggestion}"
                )

                # Recall 专项诊断：低 recall + 高 faithfulness = 检索召回不够，模型没东西可答
                if rag_recall < 0.6 and rag.get("faithfulness", 0) > 0.8:
                    report_lines.append(
                        f"  -  Recall={rag_recall:.4f} 偏低但 Faithfulness 较高，"
                        f"说明生成忠实但信息覆盖不全，建议提高检索 top_k 或优化分块大小。"
                    )
                # Recall 单独低 = 模型在生成时遗漏了关键信息
                elif rag_recall < 0.6 and delta_recall < 0.05:
                    report_lines.append(
                        f"  -  Recall={rag_recall:.4f} 偏低且 RAG 无明显提升，"
                        f"建议优化生成 prompt 以鼓励模型更全面地覆盖信息点。"
                    )

        report = "\n".join(report_lines)

        # 保存报告
        model_short = get_model_short_name(LLM_MODEL)
        timestamp = get_timestamp()
        report_path = os.path.join(summary_dir, f"final_evaluation_report_{model_short}_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"综合评估报告已保存到 {report_path}")

        # 打印报告摘要
        print("\n" + "=" * 60)
        print("评估完成！报告已生成到以下位置：")
        print(f"  - 综合报告: {report_path}")
        print(f"  - 各知识库报告: ./eval/reports/<kb_name>/")
        print(f"  - 汇总报告: ./eval/reports/summary/")
        print("=" * 60)

    def run_full_evaluation(
        self,
        skip_dataset_build: bool = False,
        skip_retrieval: bool = False,
        skip_generation: bool = False,
    ):
        """
        运行完整的评估流程

        Args:
            skip_dataset_build: 是否跳过数据集构建（使用已有数据集）
            skip_retrieval: 是否跳过检索评估
            skip_generation: 是否跳过生成评估
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("开始完整评估流程")
        logger.info("=" * 60)

        # Step 1: 构建或加载评测集
        if skip_dataset_build:
            datasets = self.load_datasets()
        else:
            datasets = self.build_datasets()

        if not datasets:
            logger.error("没有可用的评测集，评估终止")
            return

        # Step 2: 检索评估
        retrieval_results = {}
        if not skip_retrieval:
            retrieval_results = self.run_retrieval_evaluation(datasets)

        # Step 3: 生成评估
        generation_results = {}
        if not skip_generation:
            generation_results = self.run_generation_evaluation(datasets)

        # Step 4: 生成最终报告
        self.generate_final_report(retrieval_results, generation_results, datasets)

        elapsed = time.time() - start_time
        logger.info(f"\n评估流程完成，总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAG 评估系统")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认: eval/knowledge_base_config.yaml）",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="跳过数据集构建（使用已有数据集）",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="跳过检索评估",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="跳过生成评估",
    )
    parser.add_argument(
        "--only-dataset",
        action="store_true",
        help="仅构建数据集",
    )
    parser.add_argument(
        "--only-retrieval",
        action="store_true",
        help="仅执行检索评估",
    )
    parser.add_argument(
        "--only-generation",
        action="store_true",
        help="仅执行生成评估",
    )
    parser.add_argument(
        "--kb-name",
        type=str,
        default=None,
        help="指定知识库名称（如 law / project），不指定则处理所有启用的知识库",
    )
    return parser.parse_args()


def _filter_datasets_by_kb(datasets: Dict[str, Dict], kb_name: str = None) -> Dict[str, Dict]:
    """按知识库名称过滤数据集"""
    if kb_name is None:
        return datasets
    if kb_name in datasets:
        return {kb_name: datasets[kb_name]}
    logger.warning(f"知识库 {kb_name} 不存在，可选: {list(datasets.keys())}")
    return {}


def main():
    """主函数"""
    args = parse_args()

    pipeline = EvaluationPipeline(config_path=args.config)

    if args.only_dataset:
        kb_list = pipeline.get_enabled_knowledge_bases()
        if args.kb_name:
            kb_list = [args.kb_name] if args.kb_name in kb_list else []
        if kb_list:
            datasets = pipeline.build_datasets(kb_list=kb_list)
            logger.info(f"评测集构建完成: {list(datasets.keys())}")
    elif args.only_retrieval:
        datasets = pipeline.load_datasets()
        datasets = _filter_datasets_by_kb(datasets, args.kb_name)
        if datasets:
            pipeline.run_retrieval_evaluation(datasets)
    elif args.only_generation:
        datasets = pipeline.load_datasets()
        datasets = _filter_datasets_by_kb(datasets, args.kb_name)
        if datasets:
            pipeline.run_generation_evaluation(datasets)
    else:
        pipeline.run_full_evaluation(
            skip_dataset_build=args.skip_dataset,
            skip_retrieval=args.skip_retrieval,
            skip_generation=args.skip_generation,
        )


if __name__ == "__main__":
    main()
