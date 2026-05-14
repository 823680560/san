# -*- coding: utf-8 -*-
"""
评测集构建模块（支持多知识库）
==============================
功能：
1. 从现有知识库索引中提取节点，随机抽取节点组成评测组
2. 每组选 2-5 个节点，合并文本后用 LLM 生成自然问题
3. 选中的节点直接作为 context_chunks（不依赖检索器）
4. 用 LLM 反向过滤：判断每个 context_chunk 是否与问题相关
5. 基于过滤后的 context_chunks 生成参考答案
6. 按知识库分别输出评测集 JSON 文件
"""

import json
import os
import logging
import re
import random
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from configs.config import (
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    KB_LAW,
    EMBEDDING_MODEL, EMBEDDING_SERVICE, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_NORMALIZE, EMBEDDING_BATCH_SIZE, resolve_model_path,
    LAW_CHUNK_SIZE, LAW_CHUNK_OVERLAP, PROJECT_CHUNK_SIZE, PROJECT_CHUNK_OVERLAP,
)
from eval.llm_wrapper import SimpleLLM
from eval.agent_adapter import AgentAdapter
from configs.prompt_config import (
    chunk_relevance_filter_prompt,
    law_question_gen_prompt,
    project_question_gen_prompt,
    ground_truth_answer_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _init_embedding_model():
    """初始化嵌入模型和 LLM（从 config 统一读取参数）"""
    # 初始化嵌入模型
    if Settings._embed_model is None:
        if EMBEDDING_SERVICE == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            Settings.embed_model = OllamaEmbedding(
                model_name=EMBEDDING_MODEL,
                normalize=EMBEDDING_NORMALIZE,
                embed_batch_size=EMBEDDING_BATCH_SIZE
            )
            logger.info(f"已初始化 Ollama 嵌入模型: {EMBEDDING_MODEL}")
        elif EMBEDDING_SERVICE == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            model_path = resolve_model_path(EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME)
            Settings.embed_model = HuggingFaceEmbedding(model_name=model_path, embed_batch_size=EMBEDDING_BATCH_SIZE)
            logger.info(f"已初始化 HuggingFace 嵌入模型: {model_path}")
        else:
            raise ValueError(f"不支持的嵌入模型服务: {EMBEDDING_SERVICE}")

    # 初始化 LLM（避免 QueryFusionRetriever 等组件访问 Settings.llm 时触发默认 OpenAI 加载）
    if Settings._llm is None:
        from llama_index.core.llms.mock import MockLLM
        logger.info("初始化 MockLLM（避免 Settings.llm 触发默认 OpenAI 加载）")
        Settings.llm = MockLLM()


class DatasetBuilder:
    """
    评测集构建器
    负责从知识库索引中提取 chunks，并为每个 chunk 生成高质量的问题-答案对。
    使用检索器找出所有相关的上下文块，并使用 LLM 生成简洁的参考答案。
    """

    def __init__(self, llm=None):
        """
        初始化评测集构建器

        Args:
            llm: LLM 实例，用于生成问题和答案。若为 None 则自动创建
        """
        _init_embedding_model()
        
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._create_llm()
        # 初始化 AgentAdapter 用于检索相关上下文
        self.agent_adapter = AgentAdapter(llm=self.llm)

    def _create_llm(self):
        """创建评估用的 LLM（使用 SimpleLLM 包装器，基于 langchain_openai.ChatOpenAI）"""
        logger.info(f"初始化 DatasetBuilder LLM，模型: {LLM_MODEL}")
        return SimpleLLM(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.3,
        )

    def extract_nodes_from_index(self, index: VectorStoreIndex) -> List:
        """
        从 VectorStoreIndex 中提取所有节点（chunks）

        Args:
            index: LlamaIndex 的 VectorStoreIndex 实例

        Returns:
            nodes 列表
        """
        try:
            docstore = index.docstore
            all_docs = docstore.docs
            nodes = []
            for node_id, node in all_docs.items():
                if hasattr(node, 'text') or hasattr(node, 'get_content'):
                    nodes.append(node)
            logger.info(f"从索引中提取了 {len(nodes)} 个节点")
            return nodes
        except Exception as e:
            logger.error(f"提取节点失败: {e}")
            return []

    def load_index(self, kb_name: str) -> Optional[VectorStoreIndex]:
        """
        加载指定知识库的索引

        Args:
            kb_name: 知识库名称 ("law" 或 "project")

        Returns:
            VectorStoreIndex 实例，失败返回 None
        """
        path_map = {
            "law": KB_LAW,
        }
        save_path = path_map.get(kb_name)
        if not save_path or not os.path.exists(save_path):
            logger.error(f"知识库 {kb_name} 的索引路径不存在: {save_path}")
            return None

        try:
            from llama_index.vector_stores.faiss import FaissVectorStore
            vector_store = FaissVectorStore.from_persist_dir(save_path)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=save_path
            )
            index = load_index_from_storage(storage_context)
            logger.info(f"成功加载知识库 {kb_name} 的索引")
            return index
        except Exception as e:
            logger.error(f"加载知识库 {kb_name} 索引失败: {e}")
            return None

    def build_dataset(
        self,
        kb_name: str,
        index: VectorStoreIndex,
        questions_per_chunk: int = 1,
        test_size: Optional[int] = 20,
        output_dir: str = "./eval/datasets",
        max_nodes_per_group: int = 5,
    ) -> Dict:
        """
        为指定知识库构建评测集

        Args:
            kb_name: 知识库名称
            index: VectorStoreIndex 实例
            questions_per_chunk: 每个评测组生成的问题数
            test_size: 评测组数量（从全部节点中随机抽取）
            output_dir: 输出目录
            max_nodes_per_group: 每个评测组最多选取的 node 数（默认5）

        Returns:
            评测集字典
        """
        # 提取节点
        nodes = self.extract_nodes_from_index(index)
        if not nodes:
            logger.warning(f"知识库 {kb_name} 没有可用的节点，尝试通过重新加载文档构建...")
            nodes = self._fallback_load_nodes(kb_name)

        if not nodes:
            logger.error(f"知识库 {kb_name} 无法获取节点，跳过")
            return {}

        logger.info(f"知识库 {kb_name} 共有 {len(nodes)} 个节点")

        # 直接从全部节点中随机抽取节点组成评测组
        random.seed(42)
        selected_groups = {}
        num_groups = min(test_size or 0, len(nodes))
        for i in range(num_groups):
            n = min(random.randint(2, max_nodes_per_group), len(nodes))
            group_nodes = random.sample(nodes, n)
            selected_groups[f"group_{i+1}"] = group_nodes

        total_selected = sum(len(v) for v in selected_groups.values())
        logger.info(f"从全部节点中随机抽取了 {len(selected_groups)} 个评测组，共 {total_selected} 个节点")

        # 使用 LLM 为每个组生成问题
        logger.info(f"开始为知识库 {kb_name} 生成问题-上下文-答案三元组...")
        dataset = self._generate_qa_triplets(kb_name, selected_groups, questions_per_chunk)

        # 保存到文件
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{kb_name}_eval_dataset.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"评测集已保存到 {output_path}")

        return dataset

    def _generate_qa_triplets(
        self, kb_name: str, selected_groups: Dict[str, List], questions_per_chunk: int = 1
    ) -> Dict:
        """
        为每个评测组生成问题-上下文-答案三元组

        流程：
        1. 将同组的多个 node 文本合并后传给 LLM 生成问题
        2. 选中的 node 直接作为 context_chunks（含完整文本和元信息）
        3. 用 LLM 反向过滤：判断每个 context_chunk 是否与问题相关，不相关的剔除
        4. 同步更新 source_node_ids 与 context_chunks 一一对应
        5. 基于过滤后的 context_chunks 生成答案

        Args:
            kb_name: 知识库名称
            selected_groups: {group_key: [node1, node2, ...]} 格式的分组字典
            questions_per_chunk: 每个评测组生成的问题数

        Returns:
            标准格式的评测集字典
        """
        queries = []
        qid = 1

        MAX_COMBINED_LENGTH = 4000  # 合并文本最大长度，超长截断

        for group_key, group_nodes in tqdm(selected_groups.items(), desc=f"  Generating Q&A for {kb_name}"):
            # 1. 合并文本：用分隔符连接多个 node 的文本
            texts = []
            for node in group_nodes:
                text = node.text if hasattr(node, 'text') else node.get_content()
                texts.append(text)
            combined_text = "\n---\n".join(texts)

            # 超长截断
            if len(combined_text) > MAX_COMBINED_LENGTH:
                combined_text = combined_text[:MAX_COMBINED_LENGTH] + "\n...（以下内容已截断）"

            # 取第一个节点的 metadata 作为代表
            first_node = group_nodes[0]
            metadata = dict(first_node.metadata) if hasattr(first_node, 'metadata') else {}

            # 2. 所有选中节点的 node_id 作为 source_node_ids
            source_node_ids = [node.node_id for node in group_nodes if hasattr(node, 'node_id')]

            # 3. 构建 context_chunks：直接来自选中的 node（含完整文本和元信息）
            context_chunks = []
            for node in group_nodes:
                text = node.text if hasattr(node, 'text') else node.get_content()
                node_metadata = dict(node.metadata) if hasattr(node, 'metadata') else {}
                context_chunks.append({
                    "text": text,
                    "metadata": node_metadata,
                    "node_id": node.node_id if hasattr(node, 'node_id') else "",
                })

            # 4. 生成自然问题（传入 metadata 使问题更有针对性）
            question = self._generate_natural_question(combined_text, kb_name, metadata)

            # 5. LLM 反向过滤：判断每个 context_chunk 是否与问题相关
            filtered_chunks, filtered_ids = self._filter_context_chunks(question, context_chunks, source_node_ids)

            # 6. 基于过滤后的 context_chunks 生成答案
            if filtered_chunks:
                # 合并过滤后的文本用于生成答案
                filtered_texts = [chunk["text"] for chunk in filtered_chunks]
                combined_filtered = "\n---\n".join(filtered_texts)
                if len(combined_filtered) > MAX_COMBINED_LENGTH:
                    combined_filtered = combined_filtered[:MAX_COMBINED_LENGTH] + "\n...（以下内容已截断）"
                answer = self._generate_concise_answer(question, combined_filtered, kb_name)
            else:
                # 如果全部被过滤掉，用原始合并文本生成答案
                answer = self._generate_concise_answer(question, combined_text, kb_name)
                filtered_chunks = context_chunks
                filtered_ids = source_node_ids

            queries.append({
                "qid": qid,
                "query": question,
                "source_node_ids": filtered_ids,
                "context_chunks": filtered_chunks,
                "ground_truth_answer": answer,
            })
            qid += 1

            # 如果需要每个评测组生成多个问题
            for _ in range(questions_per_chunk - 1):
                question = self._generate_natural_question(combined_text, kb_name, metadata)
                filtered_chunks, filtered_ids = self._filter_context_chunks(question, context_chunks, source_node_ids)
                if filtered_chunks:
                    filtered_texts = [chunk["text"] for chunk in filtered_chunks]
                    combined_filtered = "\n---\n".join(filtered_texts)
                    if len(combined_filtered) > MAX_COMBINED_LENGTH:
                        combined_filtered = combined_filtered[:MAX_COMBINED_LENGTH] + "\n...（以下内容已截断）"
                    answer = self._generate_concise_answer(question, combined_filtered, kb_name)
                else:
                    answer = self._generate_concise_answer(question, combined_text, kb_name)
                    filtered_chunks = context_chunks
                    filtered_ids = source_node_ids
                queries.append({
                    "qid": qid,
                    "query": question,
                    "source_node_ids": filtered_ids,
                    "context_chunks": filtered_chunks,
                    "ground_truth_answer": answer,
                })
                qid += 1

        return {
            "knowledge_base": kb_name,
            "queries": queries,
        }

    def _filter_context_chunks(
        self,
        question: str,
        context_chunks: List[Dict],
        source_node_ids: List[str],
    ) -> Tuple[List[Dict], List[str]]:
        """
        使用 LLM 反向过滤：判断每个 context_chunk 是否与问题相关

        Args:
            question: 生成的问题
            context_chunks: 待过滤的上下文块列表
            source_node_ids: 对应的 node_id 列表

        Returns:
            (过滤后的 context_chunks, 过滤后的 source_node_ids) 元组，两者一一对应
        """
        if len(context_chunks) <= 1:
            # 只有一个 chunk，无需过滤
            return context_chunks, source_node_ids

        # 构建每个 chunk 的摘要信息供 LLM 判断
        chunks_info = []
        for i, chunk in enumerate(context_chunks):
            text_preview = chunk["text"][:200]  # 取前200字作为预览
            title = chunk["metadata"].get("title", "")
            chunks_info.append(f"[{i}] 标题: {title}\n    内容预览: {text_preview}...")

        chunks_text = "\n\n".join(chunks_info)

        prompt = chunk_relevance_filter_prompt(question, chunks_text)

        try:
            result = self.llm.complete(prompt)
            response = result.text.strip()

            if response.lower() == "all":
                return context_chunks, source_node_ids
            elif response.lower() == "none":
                # 全部不相关，返回空
                return [], []
            else:
                # 解析编号列表
                indices = []
                for part in response.split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part)
                        if 0 <= idx < len(context_chunks):
                            indices.append(idx)

                if not indices:
                    # 解析失败，全部保留
                    return context_chunks, source_node_ids

                filtered_chunks = [context_chunks[i] for i in indices]
                filtered_ids = [source_node_ids[i] for i in indices]
                return filtered_chunks, filtered_ids

        except Exception as e:
            logger.warning(f"LLM 过滤 context_chunks 失败: {e}")
            return context_chunks, source_node_ids

    # 问题类型列表（用于 project 知识库，代码层面强制随机化）
    PROJECT_QUESTION_TYPES = [
        "投标截止日期、开标时间、项目周期等时间信息",
        "采购人、招标人、代理机构等主体信息",
        "预算金额、中标金额、采购预算等金额信息",
        "实施地点、开标地点、项目地点等地点信息",
        "对投标人的资格要求、资质条件等要求",
        "中标单位、中标候选人等结果信息",
        "采购范围、服务内容、项目规模等内容信息",
        "联系方式、地址、邮箱等联系信息",
    ]

    def _generate_natural_question(self, text: str, kb_name: str, metadata: Dict = None) -> str:
        """
        使用 LLM 生成自然、口语化的问题

        改进：确保生成的是真正的问题，而非提示词残留

        Args:
            text: 文本内容
            kb_name: 知识库名称
            metadata: 节点的元数据（用于生成更有针对性的问题）

        Returns:
            生成的自然问题
        """
        # 截取文本前 1500 字符作为上下文（避免超出 token 限制）
        truncated_text = text[:1500]
        metadata = metadata or {}

        if kb_name == "law":
            prompt = law_question_gen_prompt(truncated_text)
        else:
            # project 类型：利用 metadata 中的项目名称、招标单位等信息生成更有针对性的问题
            project_name = metadata.get('project_name', '')
            title = metadata.get('title', '')
            category = metadata.get('category', '')
            budget = metadata.get('budget', '')
            region = metadata.get('region', '')

            # 代码层面随机选择问题类型，确保多样性
            question_type = random.choice(self.PROJECT_QUESTION_TYPES)

            prompt = project_question_gen_prompt(
                truncated_text=truncated_text,
                project_name=project_name,
                title=title,
                category=category,
                budget=budget,
                region=region,
                question_type=question_type,
            )

        try:
            result = self.llm.complete(prompt)
            question = result.text.strip()
            # 清理：移除可能的引号、编号前缀、"问题："等
            question = re.sub(r'^(问题|问)[：:]\s*', '', question)
            question = question.strip('"').strip("'").strip()
            # 确保以问号结尾
            if not question.endswith('？') and not question.endswith('?'):
                question += '？'
            # 如果问题为空或太短，使用备用方案
            if len(question) < 5:
                return self._fallback_question(text)

            # 后处理：检查是否包含模糊指代，若有则替换为具体项目名称
            if kb_name == "project":
                project_name = metadata.get('project_name', '')
                if project_name:
                    # 替换"这个项目"、"该项目"、"本项目"等模糊指代为具体项目名称
                    question = re.sub(r'这个项目', project_name, question)
                    question = re.sub(r'该项目', project_name, question)
                    question = re.sub(r'本项目', project_name, question)
                    question = re.sub(r'此项目', project_name, question)

            return question
        except Exception as e:
            logger.warning(f"LLM 生成问题失败: {e}")
            return self._fallback_question(text)

    def _fallback_question(self, text: str) -> str:
        """
        备用方案：当 LLM 生成失败时，从文本中提取关键信息构造问题

        改进：生成真正的问题而非提示词残留

        Args:
            text: 文本内容

        Returns:
            构造的问题
        """
        # 尝试从文本中提取关键短语来构造问题
        # 查找文本中的关键信息点
        sentences = re.split(r'[。！？]', text)
        # 过滤出有实质内容的句子
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 10]

        if meaningful:
            # 取第一个有意义的句子作为问题的基础
            key_info = meaningful[0][:60]
            # 构造一个自然的问题
            return f"关于「{key_info}……」的具体规定是什么？"
        else:
            # 极端情况：文本太短
            preview = text[:40].strip()
            return f"「{preview}」的具体内容是什么？"

    def _generate_concise_answer(self, query: str, source_text: str, kb_name: str) -> str:
        """
        使用 LLM 生成参考答案

        改进：不直接复制原文，而是基于原文生成准确、全面的答案

        Args:
            query: 问题
            source_text: 原始文本（当前节点的内容）
            kb_name: 知识库名称

        Returns:
            参考答案
        """
        truncated_text = source_text[:3000]

        prompt = ground_truth_answer_prompt(query, truncated_text)

        try:
            result = self.llm.complete(prompt)
            answer = result.text.strip()
            # 清理
            answer = re.sub(r'^(答案|答)[：:]\s*', '', answer)
            answer = answer.strip('"').strip("'").strip()
            if len(answer) < 5:
                # 如果答案太短，使用原文摘要
                return source_text[:300].strip()
            return answer
        except Exception as e:
            logger.warning(f"LLM 生成答案失败: {e}")
            return source_text[:300].strip()

    def _fallback_load_nodes(self, kb_name: str) -> List:
        """
        备用方案：通过重新加载原始数据并分片来获取节点

        Args:
            kb_name: 知识库名称

        Returns:
            节点列表
        """
        try:
            from server.rag.data_loader import DataLoader
            from configs.config import PROJECT_EXCEL, LAW_JSON

            if kb_name == "project":
                documents = DataLoader.load_excel_data(PROJECT_EXCEL)
            elif kb_name == "law":
                documents = DataLoader.load_json_data(LAW_JSON)
            else:
                return []

            if kb_name == "project":
                chunk_sz, chunk_ol = PROJECT_CHUNK_SIZE, PROJECT_CHUNK_OVERLAP
            else:
                chunk_sz, chunk_ol = LAW_CHUNK_SIZE, LAW_CHUNK_OVERLAP
            splitter = SentenceSplitter(chunk_size=chunk_sz, chunk_overlap=chunk_ol)
            nodes = splitter.get_nodes_from_documents(documents)
            logger.info(f"备用方案：从原始数据加载并分片得到 {len(nodes)} 个节点")
            return nodes
        except Exception as e:
            logger.error(f"备用加载节点失败: {e}")
            return []

    def build_all_datasets(
        self,
        kb_list: List[str] = None,
        questions_per_chunk: int = 1,
        test_size: int = 20,
        output_dir: str = "./eval/datasets",
        max_nodes_per_group: int = 5,
    ) -> Dict[str, Dict]:
        """
        为所有指定的知识库构建评测集

        Args:
            kb_list: 知识库名称列表，默认 ["law", "project"]
            questions_per_chunk: 每个评测组生成的问题数
            test_size: 每个知识库的评测组数量
            output_dir: 输出目录
            max_nodes_per_group: 每个评测组最多选取的 node 数（默认5）

        Returns:
            {kb_name: dataset_dict} 格式的字典
        """
        if kb_list is None:
            kb_list = ["law", "project"]

        all_datasets = {}
        for kb_name in kb_list:
            logger.info(f"\n{'='*60}")
            logger.info(f"开始为知识库 {kb_name} 构建评测集")
            logger.info(f"{'='*60}")

            index = self.load_index(kb_name)
            if index is None:
                logger.warning(f"跳过知识库 {kb_name}：无法加载索引")
                continue

            dataset = self.build_dataset(
                kb_name=kb_name,
                index=index,
                questions_per_chunk=questions_per_chunk,
                test_size=test_size,
                output_dir=output_dir,
                max_nodes_per_group=max_nodes_per_group,
            )
            if dataset:
                all_datasets[kb_name] = dataset

        return all_datasets


class SQLBasedDatasetBuilder:
    """
    NL2SQL 评测数据集构建器（基于 SQL 模版）

    与 DatasetBuilder（基于文本块检索）不同，本类直接从数据库出发：
    1. 分析数据库中各列的数据分布
    2. 定义参数化 SQL 模版（覆盖 exact / fuzzy / aggregation 三大类型）
    3. 用实际数据值填充模版，得到可执行的 SQL
    4. 执行 SQL 获取 expected_results
    5. 用 LLM 将 SQL 转换为自然语言问题
    6. 输出 (question, expected_sql, expected_results, query_type) 评测集

    这样保证了每个问题都有确定的 SQL 对应关系，结果可数据库验证。
    """

    # 可用于筛选条件的分类列
    CATEGORY_COLUMNS = [
        "大区", "省份", "城市", "区县", "城乡分类",
        "项目类型", "项目状态", "是否PPP", "采购主体",
    ]

    # 数值列
    NUMERIC_COLUMNS = [
        "年化金额_元", "年限_年", "合同金额_元",
        "在运营年化金额_元", "剩余合同期限_月",
    ]

    # 日期列
    DATE_COLUMNS = [
        "中标时间", "时间节点", "预计合同到期时间",
    ]

    # 文本搜索列
    TEXT_COLUMNS = [
        "项目名称", "中标公司", "招标单位", "招标代理",
    ]

    def __init__(self, db_path: str = None, llm=None, engine=None):
        """
        Args:
            db_path: SQLite 数据库路径
            llm: LLM 实例，用于将 SQL 转为自然语言问题
            engine: ProjectQueryEngine 实例，用于生成 Vanna SQL（未传入则退回模板模式）
        """
        from configs.config import PROJECT_SQLITE_DB
        self.db_path = db_path or PROJECT_SQLITE_DB
        self.conn = None
        self.llm = llm or self._create_llm()
        self.engine = engine  # Vanna NL2SQL 引擎
        self._data_profile = {}  # 数据分布概要

    def _create_llm(self):
        """创建 LLM（与 DatasetBuilder 一致）"""
        from eval.llm_wrapper import SimpleLLM
        from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
        return SimpleLLM(
            model=LLM_MODEL, api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL, temperature=0.3,
        )

    def _connect(self):
        """建立数据库连接"""
        import sqlite3
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def analyze_data_distribution(self) -> Dict:
        """
        分析数据库中的数据分布

        获取各列的 distinct 值、数值范围、日期范围等，
        用于后续 SQL 模版的参数填充。
        """
        conn = self._connect()
        profile = {}

        # 1. 分类列：获取 distinct 值（最多前 20 个）
        for col in self.CATEGORY_COLUMNS:
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT \"{col}\" FROM sanitation_projects "
                    f"WHERE \"{col}\" IS NOT NULL AND \"{col}\" != '' "
                    f"ORDER BY \"{col}\" LIMIT 20"
                ).fetchall()
                profile[col] = [r[0] for r in rows]
            except Exception as e:
                logger.warning(f"分析列 {col} 失败: {e}")
                profile[col] = []

        # 2. 数值列：获取 min, max, avg
        for col in self.NUMERIC_COLUMNS:
            try:
                row = conn.execute(
                    f"SELECT MIN(\"{col}\") as min_v, MAX(\"{col}\") as max_v, "
                    f"AVG(\"{col}\") as avg_v FROM sanitation_projects "
                    f"WHERE \"{col}\" IS NOT NULL"
                ).fetchone()
                profile[col] = {
                    "min": row[0], "max": row[1], "avg": row[2],
                }
            except Exception as e:
                logger.warning(f"分析列 {col} 失败: {e}")
                profile[col] = {}

        # 3. 日期列：获取 min, max
        for col in self.DATE_COLUMNS:
            try:
                row = conn.execute(
                    f"SELECT MIN(\"{col}\") as min_d, MAX(\"{col}\") as max_d "
                    f"FROM sanitation_projects WHERE \"{col}\" IS NOT NULL"
                ).fetchone()
                profile[col] = {"min": row[0], "max": row[1]}
            except Exception as e:
                logger.warning(f"分析列 {col} 失败: {e}")
                profile[col] = {}

        # 4. 总行数
        profile["_total_rows"] = conn.execute(
            "SELECT COUNT(*) FROM sanitation_projects"
        ).fetchone()[0]

        self._data_profile = profile
        logger.info(f"数据分析完成：{profile['_total_rows']} 行，"
                     f"{sum(1 for k in profile if k != '_total_rows' and profile[k])} 列有数据")
        return profile

    # ========== SQL 模版生成器 ==========

    def _gen_templates(self) -> List[Dict]:
        """
        生成参数化的 SQL 模版列表

        每个模版是一个 dict：
        {
            "type": "exact"|"fuzzy"|"aggregation",
            "template_name": str,
            "build_sql": callable(profile) -> (sql, params_info),
        }
        """
        p = self._data_profile
        templates = []

        # --- 1. 单条件精确查询 ---
        for col in self.CATEGORY_COLUMNS:
            vals = p.get(col, [])
            if not vals:
                continue
            templates.append({
                "type": "exact",
                "name": f"single_condition_{col}",
                "build": lambda _col=col, _vals=vals: (
                    f"SELECT * FROM sanitation_projects WHERE \"{_col}\" = "
                    f"'{_vals[0]}' LIMIT 10",
                    {_col: _vals[0]},
                ),
            })

        # --- 2. 多条件组合查询 ---
        region_cols = ["大区", "省份", "城市"]
        active_cols = [c for c in region_cols if p.get(c)]
        if len(active_cols) >= 2:
            templates.append({
                "type": "exact",
                "name": "multi_condition_region",
                "build": lambda: self._build_multi_condition(active_cols),
            })

        # 区域 + 项目类型
        type_vals = p.get("项目类型", [])
        if active_cols and type_vals:
            templates.append({
                "type": "exact",
                "name": "region_and_type",
                "build": lambda: self._build_multi_condition(
                    [active_cols[0], "项目类型"]
                ),
            })

        # --- 3. 数值范围查询 ---
        for col in self.NUMERIC_COLUMNS[:3]:  # 取前 3 个主要数值列
            stats = p.get(col, {})
            if stats.get("min") is not None and stats.get("max") is not None:
                templates.append({
                    "type": "exact",
                    "name": f"range_{col}",
                    "build": lambda _col=col, _stats=stats: self._build_range_query(
                        _col, _stats
                    ),
                })

        # --- 4. 日期范围查询 ---
        for col in self.DATE_COLUMNS[:1]:  # 取中标时间
            stats = p.get(col, {})
            if stats.get("min") and stats.get("max"):
                templates.append({
                    "type": "exact",
                    "name": f"date_range_{col}",
                    "build": lambda _col=col, _stats=stats: self._build_date_range(
                        _col, _stats
                    ),
                })

        # --- 5. 聚合查询 ---
        templates.append({
            "type": "aggregation",
            "name": "count_by_province",
            "build": lambda: (
                "SELECT 省份, COUNT(*) AS 项目数 FROM sanitation_projects "
                "WHERE 省份 IS NOT NULL AND 省份 != '' "
                "GROUP BY 省份 ORDER BY 项目数 DESC LIMIT 10",
                {"聚合": "按省份统计项目数量"},
            ),
        })

        templates.append({
            "type": "aggregation",
            "name": "avg_amount_by_type",
            "build": lambda: (
                "SELECT 项目类型, AVG(年化金额_元) AS 平均年化 FROM sanitation_projects "
                "WHERE 年化金额_元 IS NOT NULL AND 项目类型 IS NOT NULL "
                "GROUP BY 项目类型 ORDER BY 平均年化 DESC LIMIT 10",
                {"聚合": "各项目类型平均年化金额"},
            ),
        })

        templates.append({
            "type": "aggregation",
            "name": "total_contract_by_region",
            "build": lambda: (
                "SELECT 大区, SUM(合同金额_元) AS 总金额 FROM sanitation_projects "
                "WHERE 合同金额_元 IS NOT NULL AND 大区 IS NOT NULL "
                "GROUP BY 大区 ORDER BY 总金额 DESC LIMIT 10",
                {"聚合": "各大区合同总金额"},
            ),
        })

        # --- 6. 模糊搜索 (FTS5) ---
        fts_keywords_pool = []
        # 从项目名称中提取关键词
        try:
            names = self.conn.execute(
                "SELECT 项目名称 FROM sanitation_projects "
                "WHERE 项目名称 IS NOT NULL LIMIT 50"
            ).fetchall()
            for row in names:
                name = row[0]
                for kw in ["保洁", "垃圾", "清扫", "绿化", "河道", "物业",
                           "分类", "转运", "处理", "道路", "水域", "环卫"]:
                    if kw in name:
                        fts_keywords_pool.append(kw)
        except Exception:
            pass
        fts_keywords_pool = list(set(fts_keywords_pool))

        for kw in fts_keywords_pool[:5]:
            templates.append({
                "type": "fuzzy",
                "name": f"fts_search_{kw}",
                "build": lambda _kw=kw: self._build_fts_query(_kw),
            })

        # --- 7. TOP-N/排序查询 ---
        templates.append({
            "type": "aggregation",
            "name": "top_n_contract_amount",
            "build": lambda: (
                "SELECT 项目名称, 省份, 合同金额_元 FROM sanitation_projects "
                "WHERE 合同金额_元 IS NOT NULL "
                "ORDER BY 合同金额_元 DESC LIMIT 10",
                {"排序": "合同金额最高的 10 个项目"},
            ),
        })

        templates.append({
            "type": "aggregation",
            "name": "top_n_yearly_amount",
            "build": lambda: (
                "SELECT 项目名称, 省份, 年化金额_元 FROM sanitation_projects "
                "WHERE 年化金额_元 IS NOT NULL "
                "ORDER BY 年化金额_元 DESC LIMIT 10",
                {"排序": "年化金额最高的 10 个项目"},
            ),
        })

        return templates

    def _pick_random_value(self, col: str) -> str:
        """从数据分布中随机选取一个列值"""
        import random
        vals = self._data_profile.get(col, [])
        if vals:
            return random.choice(vals)
        return ""

    def _build_multi_condition(self, cols: List[str]) -> Tuple[str, Dict]:
        """构建多条件组合查询"""
        import random
        conditions = []
        params = {}
        for col in cols:
            val = self._pick_random_value(col)
            if val:
                conditions.append(f"\"{col}\" = '{val}'")
                params[col] = val
        if conditions:
            sql = f"SELECT * FROM sanitation_projects WHERE {' AND '.join(conditions)} LIMIT 10"
            return sql, params
        return "SELECT * FROM sanitation_projects LIMIT 10", {}

    def _build_range_query(self, col: str, stats: Dict) -> Tuple[str, Dict]:
        """构建数值范围查询"""
        import random
        min_v, max_v = stats["min"], stats["max"]
        if min_v is None or max_v is None:
            return f"SELECT * FROM sanitation_projects LIMIT 10", {}

        # 取中位值附近作为阈值
        threshold = min_v + (max_v - min_v) * random.uniform(0.3, 0.7)
        return (
            f"SELECT * FROM sanitation_projects "
            f"WHERE \"{col}\" > {threshold:.0f} "
            f"ORDER BY \"{col}\" DESC LIMIT 10",
            {col: f">{threshold:.0f}"},
        )

    def _build_date_range(self, col: str, stats: Dict) -> Tuple[str, Dict]:
        """构建日期范围查询"""
        import random
        from datetime import datetime, timedelta
        try:
            min_d = datetime.strptime(stats["min"][:10], "%Y-%m-%d")
            max_d = datetime.strptime(stats["max"][:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return f"SELECT * FROM sanitation_projects LIMIT 10", {}

        if min_d >= max_d:
            return f"SELECT * FROM sanitation_projects LIMIT 10", {}

        # 在中间偏前的位置选一个日期
        delta = (max_d - min_d) * random.uniform(0.3, 0.5)
        pivot = min_d + delta
        return (
            f"SELECT * FROM sanitation_projects "
            f"WHERE \"{col}\" >= '{pivot.date()}' "
            f"ORDER BY \"{col}\" LIMIT 10",
            {col: f">={pivot.date()}"},
        )

    def _build_fts_query(self, keyword: str) -> Tuple[str, Dict]:
        """构建 FTS5 模糊搜索查询"""
        return (
            f"SELECT sp.* FROM sanitation_projects_fts fts "
            f"JOIN sanitation_projects sp ON fts.rowid = sp.id "
            f"WHERE sanitation_projects_fts MATCH '{keyword}' LIMIT 10",
            {"关键词": keyword},
        )

    def _sql_to_question(self, sql: str, results: List[Dict]) -> str:
        """用 LLM 将 SQL + 执行结果转为自然语言问题"""
        from configs.prompt_config import sql_to_question_prompt

        # 取前 5 行结果作为示例
        sample_lines = []
        for i, row in enumerate(results[:5]):
            sample_lines.append(f"行{i + 1}: " + ", ".join(
                f"{k}={v}" for k, v in row.items()
            ))
        sample_str = "\n".join(sample_lines) if sample_lines else "(空结果)"

        table_ddl = (
            "sanitation_projects (id, 大区, 省份, 城市, 区县, 城乡分类, "
            "项目名称, 招标单位, 招标代理, 采购主体, 是否PPP, "
            "项目类型, 项目状态, 中标时间, 年化金额_元, 年限_年, "
            "合同金额_元, 时间节点, 在运营年化金额_元, 剩余合同期限_月, "
            "预计合同到期时间, 中标公司, 清扫面积_㎡, 单价_元_㎡, "
            "转运或处理补贴_元_吨, 转运或处理量_吨_年, 转运站个数_个, 备注)"
        )

        prompt = sql_to_question_prompt(sql, sample_str, table_ddl)
        try:
            result = self.llm.complete(prompt)
            question = result.text.strip()
            # 清理
            import re
            question = re.sub(r'^(问题|问)[：:]\s*', '', question)
            question = question.strip('"').strip("'").strip()
            if not question.endswith('？') and not question.endswith('?'):
                question += '？'
            if len(question) >= 5:
                return question
        except Exception as e:
            logger.warning(f"SQL 转问题失败: {e}")

        # fallback
        return f"查询：{results[0] if results else '(空)'}"

    def _gen_seed_questions(self) -> List[str]:
        """从数据分布中构造多样化的自然语言种子问题，覆盖 exact / fuzzy / aggregation 三类"""
        import random
        p = self._data_profile
        questions = []

        # --- exact: 按省份/城市筛选 ---
        provinces = p.get("省份", [])
        cities = p.get("城市", [])
        for prov in random.sample(provinces, min(20, len(provinces))):
            questions.append(f"{prov}的环卫项目有哪些")
        for city in random.sample(cities, min(20, len(cities))):
            questions.append(f"{city}的环卫项目")

        # --- exact: 按区县筛选 ---
        districts = p.get("区县", [])
        for d in random.sample(districts, min(10, len(districts))):
            questions.append(f"{d}的环卫项目")

        # --- exact: 按项目类型 ---
        types = p.get("项目类型", [])
        for t in random.sample(types, min(10, len(types))):
            questions.append(f"{t}类项目有哪些")

        # --- exact: 按项目状态 ---
        statuses = p.get("项目状态", [])
        for s in random.sample(statuses, min(8, len(statuses))):
            questions.append(f"状态为{s}的项目")

        # --- exact: 按采购主体 ---
        buyers = p.get("采购主体", [])
        for b in random.sample(buyers, min(8, len(buyers))):
            questions.append(f"{b}采购的项目")

        # --- exact: 按是否PPP ---
        ppp_vals = p.get("是否PPP", [])
        for ppp in ppp_vals:
            questions.append(f"是否PPP为{ppp}的项目")

        # --- exact: 多条件组合 ---
        if provinces and types:
            for _ in range(10):
                pv = random.choice(provinces)
                tp = random.choice(types)
                questions.append(f"{pv}的{tp}项目")
        if cities and types:
            for _ in range(5):
                ct = random.choice(cities)
                tp = random.choice(types)
                questions.append(f"{ct}的{tp}项目")
        if provinces and statuses:
            for _ in range(5):
                pv = random.choice(provinces)
                st = random.choice(statuses)
                questions.append(f"{pv}状态为{st}的项目")

        # --- exact: 金额范围 ---
        for col in self.NUMERIC_COLUMNS[:3]:
            stats = p.get(col, {})
            if stats.get("min") is not None and stats.get("max") is not None:
                mid = (stats["min"] + stats["max"]) / 2
                col_label = "年化金额" if "年化" in col else ("合同金额" if "合同" in col else col)
                questions.append(f"{col_label}超过{mid/10000:.0f}万的项目")
                questions.append(f"{col_label}最高的10个项目")
                # 增加多区间
                lo = stats["min"] + (stats["max"] - stats["min"]) * 0.15
                hi = stats["min"] + (stats["max"] - stats["min"]) * 0.75
                questions.append(f"{col_label}在{lo/10000:.0f}万到{hi/10000:.0f}万之间的项目")

        # --- exact: 日期范围 ---
        for col in self.DATE_COLUMNS[:2]:
            stats = p.get(col, {})
            if stats.get("min") and stats.get("max"):
                try:
                    from datetime import datetime
                    min_d = datetime.strptime(stats["min"][:10], "%Y-%m-%d")
                    max_d = datetime.strptime(stats["max"][:10], "%Y-%m-%d")
                    min_y = min_d.year
                    max_y = max_d.year
                    mid_y = (min_y + max_y) // 2
                    questions.append(f"{mid_y}年中标的项目")
                    questions.append(f"最近中标的10个项目")
                    questions.append(f"{min_y}年到{max_y}年间的项目")
                except (ValueError, TypeError):
                    pass

        # --- fuzzy: 关键词搜索 ---
        keywords_pool = ["保洁", "垃圾", "清扫", "绿化", "河道", "物业",
                         "分类", "转运", "处理", "道路", "水域", "环卫", "PPP"]
        for kw in random.sample(keywords_pool, min(10, len(keywords_pool))):
            questions.append(f"搜索{kw}相关的项目")
            questions.append(f"包含{kw}关键词的项目")
            questions.append(f"有哪些{kw}项目")

        # --- fuzzy: 公司名搜索 ---
        try:
            companies = self.conn.execute(
                "SELECT DISTINCT 中标公司 FROM sanitation_projects "
                "WHERE 中标公司 IS NOT NULL AND 中标公司 != '' LIMIT 30"
            ).fetchall()
            for row in random.sample(companies, min(10, len(companies))):
                name = row[0]
                if len(name) >= 4:
                    questions.append(f"查询{name}的中标项目")
        except Exception:
            pass

        # --- aggregation: 分组统计 ---
        group_cols = ["省份", "大区", "项目类型", "项目状态", "城乡分类"]
        for col in group_cols:
            vals = p.get(col, [])
            if vals:
                questions.append(f"各{col}的项目数量分别是多少")
                questions.append(f"每个{col}有多少个项目")
        questions.append("合同金额最高的10个项目是哪些")
        questions.append("平均年化金额最高的项目类型")
        questions.append("项目数量最多的省份是哪个")

        # --- aggregation: 时间趋势 ---
        if p.get("中标时间", {}).get("min"):
            questions.append("每年中标的项目数量趋势")
            questions.append("每年合同总金额变化趋势")
            questions.append("每年中标金额的变化")

        # --- exact: 中标公司 + 项目类型组合 ---
        try:
            companies = self.conn.execute(
                "SELECT DISTINCT 中标公司 FROM sanitation_projects "
                "WHERE 中标公司 IS NOT NULL AND 中标公司 != '' LIMIT 20"
            ).fetchall()
            comp_list = [r[0] for r in companies if len(r[0]) >= 4]
            for _ in range(5):
                if comp_list and types:
                    comp = random.choice(comp_list)
                    tp = random.choice(types)
                    questions.append(f"{comp}中标的{tp}项目")
        except Exception:
            pass

        # 去重并打乱
        seen = set()
        unique = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        random.shuffle(unique)
        return unique

    @staticmethod
    def _sanitize_json(obj):
        """递归清洗 NaN / NaT / Inf 等非标准 JSON 值，替换为 None"""
        import math
        if isinstance(obj, dict):
            return {k: SQLBasedDatasetBuilder._sanitize_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [SQLBasedDatasetBuilder._sanitize_json(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj

    def _gen_sql_from_vanna(self, seed_questions: List[str], test_size: int) -> List[Dict]:
        """使用 Vanna (ProjectQueryEngine) 为种子问题生成 SQL，组成评测条目

        当初始种子问题池不够时，会动态生成更多随机种子问题，直到达到 test_size。
        """
        items = []
        seen_sql = set()

        # 将 seed_questions 转为队列，同时保留原始列表用于动态生成
        question_queue = list(seed_questions)
        idx = 0
        total_attempts = 0
        max_total_attempts = test_size * 5  # 总尝试上限，避免死循环

        while len(items) < test_size and total_attempts < max_total_attempts:
            total_attempts += 1

            # 如果队列用完，动态生成更多种子问题
            if idx >= len(question_queue):
                new_questions = self._gen_seed_questions_batch(20)
                if not new_questions:
                    logger.warning("无法生成更多种子问题，提前结束")
                    break
                question_queue.extend(new_questions)
                logger.info(f"  种子问题池已耗尽，动态补充 {len(new_questions)} 个新问题")

            seed_q = question_queue[idx]
            idx += 1

            try:
                result = self.engine.query(seed_q)
            except Exception as e:
                logger.warning(f"Vanna 查询失败 '{seed_q[:50]}': {e}")
                continue

            sql = result.get("sql", "")
            error = result.get("error", "")
            results = result.get("results", [])
            query_type = result.get("query_type", "exact")
            row_count = result.get("row_count", 0)

            # 过滤：必须执行成功、有结果、SQL 不重复
            if error or not sql or row_count == 0:
                continue
            if sql in seen_sql:
                continue
            seen_sql.add(sql)

            # 用 LLM 精炼问题：基于实际 SQL 和结果生成更自然的表述
            refined_q = self._sql_to_question(sql, results)
            if len(refined_q) < 6:
                refined_q = seed_q

            # LLM 整理 ground_truth_answer
            ground_truth = self._results_to_answer(refined_q, results)

            items.append({
                "query": refined_q,
                "query_type": query_type,
                "seed_question": seed_q,
                "expected_sql": sql,
                "expected_results": results,
                "ground_truth_answer": ground_truth,
            })

            logger.info(f"  [{len(items)}/{test_size}] '{refined_q[:60]}...' "
                        f"type={query_type}, rows={row_count}")

        return items

    def _gen_seed_questions_batch(self, batch_size: int = 20) -> List[str]:
        """动态生成一批种子问题，用于种子问题池不够时的补充"""
        import random
        p = self._data_profile
        questions = []

        provinces = p.get("省份", [])
        cities = p.get("城市", [])
        types = p.get("项目类型", [])
        statuses = p.get("项目状态", [])
        buyers = p.get("采购主体", [])
        districts = p.get("区县", [])

        templates = []

        # 单条件精确查询模板
        if provinces:
            templates.append(lambda: f"{random.choice(provinces)}的环卫项目有哪些")
        if cities:
            templates.append(lambda: f"{random.choice(cities)}的环卫项目")
        if districts:
            templates.append(lambda: f"{random.choice(districts)}的环卫项目")
        if types:
            templates.append(lambda: f"{random.choice(types)}类项目有哪些")
        if statuses:
            templates.append(lambda: f"状态为{random.choice(statuses)}的项目")
        if buyers:
            templates.append(lambda: f"{random.choice(buyers)}采购的项目")

        # 多条件组合
        if provinces and types:
            templates.append(lambda: f"{random.choice(provinces)}的{random.choice(types)}项目")
        if cities and types:
            templates.append(lambda: f"{random.choice(cities)}的{random.choice(types)}项目")
        if provinces and statuses:
            templates.append(lambda: f"{random.choice(provinces)}状态为{random.choice(statuses)}的项目")

        # 金额范围
        for col in self.NUMERIC_COLUMNS[:2]:
            stats = p.get(col, {})
            if stats.get("min") is not None and stats.get("max") is not None:
                col_label = "年化金额" if "年化" in col else "合同金额"
                mid = (stats["min"] + stats["max"]) / 2
                lo = stats["min"] + (stats["max"] - stats["min"]) * random.uniform(0.1, 0.4)
                hi = stats["min"] + (stats["max"] - stats["min"]) * random.uniform(0.6, 0.9)
                templates.append(lambda cl=col_label, m=mid: f"{cl}超过{m/10000:.0f}万的项目")
                templates.append(lambda cl=col_label, l=lo, h=hi: f"{cl}在{l/10000:.0f}万到{h/10000:.0f}万之间的项目")

        # 关键词搜索
        keywords_pool = ["保洁", "垃圾", "清扫", "绿化", "河道", "物业",
                         "分类", "转运", "处理", "道路", "水域", "环卫", "PPP"]
        for kw in keywords_pool:
            templates.append(lambda k=kw: f"搜索{k}相关的项目")
            templates.append(lambda k=kw: f"有哪些{k}项目")

        # 聚合查询
        group_cols = ["省份", "大区", "项目类型", "项目状态"]
        for col in group_cols:
            if p.get(col):
                templates.append(lambda c=col: f"各{c}的项目数量分别是多少")
        templates.append(lambda: "合同金额最高的10个项目是哪些")
        templates.append(lambda: "项目数量最多的省份是哪个")

        # 公司查询
        try:
            companies = self.conn.execute(
                "SELECT DISTINCT 中标公司 FROM sanitation_projects "
                "WHERE 中标公司 IS NOT NULL AND 中标公司 != '' "
                "ORDER BY RANDOM() LIMIT 15"
            ).fetchall()
            for row in companies:
                name = row[0]
                if len(name) >= 4:
                    templates.append(lambda n=name: f"查询{n}的中标项目")
        except Exception:
            pass

        if not templates:
            return []

        # 从模板中随机生成 batch_size 个问题
        for _ in range(batch_size):
            try:
                q = random.choice(templates)()
                questions.append(q)
            except Exception:
                continue

        # 去重
        seen = set()
        unique = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def build_dataset(
        self,
        test_size: int = 30,
        output_dir: str = "./eval/datasets",
    ) -> Dict:
        """
        构建 NL2SQL 评测数据集

        优先使用 Vanna (ProjectQueryEngine) 生成 expected_sql，由 LLM 整理 ground_truth_answer。
        若未传入 engine 则回退到手写模板模式。

        Args:
            test_size: 生成的 question-SQL 对数
            output_dir: 输出目录

        Returns:
            标准格式的评测集字典
        """
        # 1. 连接数据库并分析数据分布
        self._connect()
        self.analyze_data_distribution()

        # 2. 生成评测条目
        if self.engine is not None:
            logger.info("使用 Vanna (ProjectQueryEngine) 生成 expected_sql ...")
            seed_questions = self._gen_seed_questions()
            logger.info(f"生成了 {len(seed_questions)} 个种子问题，开始调用 Vanna...")
            items = self._gen_sql_from_vanna(seed_questions, test_size)
        else:
            logger.info("未传入 ProjectQueryEngine，退回模板模式 ...")
            items = self._build_from_templates(test_size)

        logger.info(f"成功生成 {len(items)}/{test_size} 条评测数据")

        # 3. 分配 qid
        dataset_items = []
        for i, item in enumerate(items):
            dataset_items.append({
                "qid": i + 1,
                "query": item["query"],
                "query_type": item.get("query_type", "exact"),
                "seed_question": item.get("seed_question", ""),
                "expected_sql": item["expected_sql"],
                "expected_results": item.get("expected_results", []),
                "ground_truth_answer": item["ground_truth_answer"],
            })

        if self.engine is not None:
            generation_method = "vanna_nl2sql"
        else:
            generation_method = "sql_template_based"

        dataset = {
            "knowledge_base": "project",
            "generation_method": generation_method,
            "queries": dataset_items,
        }

        # 写入前统一清洗 NaN，避免 JSON 非法值
        dataset = self._sanitize_json(dataset)

        # 保存
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "project_nl2sql_eval_dataset.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"NL2SQL 评测集已保存到 {output_path}")

        return dataset

    def _build_from_templates(self, test_size: int) -> List[Dict]:
        """回退方案：使用手写 SQL 模板构建评测集"""
        import random

        templates = self._gen_templates()
        logger.info(f"生成了 {len(templates)} 个 SQL 模版")

        dataset_items = []
        attempts = 0
        max_attempts = test_size * 3

        while len(dataset_items) < test_size and attempts < max_attempts:
            attempts += 1
            tmpl = random.choice(templates)

            try:
                sql, params_info = tmpl["build"]()
                if not sql:
                    continue

                rows = self.conn.execute(sql).fetchall()
                results = [dict(r) for r in rows]

                if not results:
                    continue

                question = self._sql_to_question(sql, results)
                if len(question) < 6:
                    continue

                ground_truth = self._results_to_answer(question, results)

                dataset_items.append({
                    "query": question,
                    "query_type": tmpl["type"],
                    "expected_sql": sql,
                    "expected_results": results,
                    "ground_truth_answer": ground_truth,
                })
            except Exception as e:
                logger.warning(f"生成第 {len(dataset_items) + 1} 条数据失败: {e}")
                continue

        return dataset_items

    @staticmethod
    def _results_to_text(results: List[Dict]) -> str:
        """将查询结果转为结构化文本（内部使用，用于 LLM 整理前的输入）"""
        if not results:
            return "(空结果)"
        lines = []
        for i, row in enumerate(results[:20]):
            parts = [f"{k}: {v}" for k, v in row.items() if v is not None and v != ""]
            lines.append(f"第{i + 1}行: " + " | ".join(parts))
        return "\n".join(lines)

    def _results_to_answer(self, question: str, results: List[Dict]) -> str:
        """使用 LLM 将 SQL 查询结果整理为自然语言参考答案（与 RAG 回答格式一致）"""
        from configs.prompt_config import nl2sql_ground_truth_answer_prompt

        if not results:
            return "未找到相关记录。"

        results_text = self._results_to_text(results)
        prompt = nl2sql_ground_truth_answer_prompt(question, results_text)

        try:
            result = self.llm.complete(prompt)
            answer = result.text.strip()
            import re
            answer = re.sub(r'^(答案|答)[：:]\s*', '', answer)
            answer = answer.strip('"').strip("'").strip()
            if len(answer) >= 10:
                return answer
        except Exception as e:
            logger.warning(f"LLM 整理 ground_truth_answer 失败: {e}")

        # fallback：使用旧的格式化文本
        return self._results_to_text(results)


