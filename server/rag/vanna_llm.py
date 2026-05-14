import re
import uuid
import logging
import pandas as pd
import os
from typing import List
import requests
from vanna.base import VannaBase

# 静默第三方库的 verbose 日志
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["TQDM_DISABLE"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

for _lib in ["transformers", "sentence_transformers", "huggingface_hub", "jieba"]:
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class DeepseekVanna(VannaBase):
    """自定义 Vanna LLM 适配器，调用 deepseek (OpenAI 兼容 API)，内存存储训练数据"""

    def __init__(self, config: dict = None):
        config = config or {}
        if "model" not in config:
            config["model"] = "deepseek-chat"
        # 手动 setup 父类需要的属性（不调 super().__init__ 避免 VannaDB 自动建表）
        self.config = config
        self.model = config["model"]
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.deepseek.com/v1")
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)
        self.run_sql_is_set = False
        self.run_sql = None
        self.static_documentation = ""

        # 内存训练数据存储
        self._ddl_store = []
        self._doc_store = []
        self._training_data = {}  # {id: {question, sql, ...}}

        # embedding 相似度检索（可选，用于优化 few-shot 示例检索）
        self._embedder = None
        self._training_embeddings = {}  # {id: np.array}
        self._embedding_model_path = config.get("embedding_model_path", "")
        self._embedding_device = config.get("embedding_device", None)  # None = 自动检测

    # ========== LLM 接口 ==========

    def submit_prompt(self, prompt, **kwargs) -> str:
        """调用 deepseek (OpenAI 兼容 API)，支持消息列表和纯文本"""
        if isinstance(prompt, list):
            # VannaBase.get_sql_prompt 返回 [system_str, user_str, assistant_str, user_str, ...]
            roles = ["system", "user", "assistant"]
            messages = []
            for i, msg in enumerate(prompt):
                role = roles[i % len(roles)]
                if i == len(prompt) - 1:
                    role = "user"  # 最后一条始终是 user
                messages.append({"role": role, "content": msg})
        else:
            messages = [{"role": "user", "content": prompt}]

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ========== Prompt 覆写：禁止 intermediate_sql ==========

    def get_sql_prompt(self, **kwargs):
        """覆写父类：将内置规则 #3 从"解释原因"改为"尽力输出 SQL" """
        import re as _re

        message_log = super().get_sql_prompt(**kwargs)

        if message_log and isinstance(message_log[0], str):
            message_log[0] = _re.sub(
                r"3\. If the provided context is insufficient, please explain why it can't be generated\.\n",
                "3. If the provided context is insufficient, output the best SQL query you can based on the available tables and columns.\n",
                message_log[0],
            )

        return message_log

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """覆写父类：增加中间 SQL 执行失败的回退逻辑"""
        import re as _re

        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)

        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        # 尝试父类的 intermediate_sql 流程（先查再精查），失败则回退到提取最终 SQL
        if 'intermediate_sql' in llm_response:
            if allow_llm_to_see_data:
                try:
                    intermediate_sql = self.extract_sql(llm_response)
                    df = self.run_sql(intermediate_sql)
                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list + [f"The following is a pandas DataFrame: \n{df.to_markdown()}"],
                        **kwargs,
                    )
                    llm_response = self.submit_prompt(prompt, **kwargs)
                except Exception:
                    # 中间 SQL 执行失败，回退：去掉 intermediate_sql 前缀，取最终 SQL
                    llm_response = _re.sub(r'.*?\bintermediate_sql\b\s*', '', llm_response, flags=_re.IGNORECASE)

        return self.extract_sql(llm_response)

    # ========== 消息格式化（VannaBase 要求实现） ==========

    @staticmethod
    def system_message(message: str):
        return message

    @staticmethod
    def user_message(message: str):
        return message

    @staticmethod
    def assistant_message(message: str):
        return message

    # ========== 日志静默 ==========

    def log(self, message: str, title: str = "Info"):
        """覆写父类 log：静默训练噪音，仅 DEBUG 输出 SQL Prompt 等关键信息"""
        if title in ("SQL Prompt", "LLM Response"):
            logger.debug(f"{title}: {message}")

    # ========== 嵌入（无实际向量需求，返回空） ==========

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        return []

    # ========== DDL 存储 ==========

    def add_ddl(self, ddl: str, **kwargs) -> str:
        _id = str(uuid.uuid4())
        self._ddl_store.append({"id": _id, "ddl": ddl})
        return _id

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return self._ddl_store

    # ========== 文档存储 ==========

    def add_documentation(self, documentation: str, **kwargs) -> str:
        _id = str(uuid.uuid4())
        self._doc_store.append({"id": _id, "documentation": documentation})
        return _id

    def get_related_documentation(self, question: str, **kwargs) -> list:
        return self._doc_store

    # ========== 问答对存储 ==========

    def _ensure_embedder(self):
        """延迟初始化 sentence_transformers 嵌入模型"""
        if self._embedder is not None:
            return True
        if not self._embedding_model_path:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            device = self._embedding_device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = SentenceTransformer(
                self._embedding_model_path,
                device=device,
                trust_remote_code=True,
            )
            self._embedder.eval()
            return True
        except Exception as e:
            logger.warning(f"嵌入模型加载失败（跳过 embedding 检索）: {e}")
            self._embedder = False
            return False

    def _compute_embedding(self, text: str):
        """计算单条文本的 embedding"""
        if not self._ensure_embedder():
            return None
        try:
            emb = self._embedder.encode(text, normalize_embeddings=True)
            return emb
        except Exception:
            return None

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        _id = str(uuid.uuid4())
        self._training_data[_id] = {"question": question, "sql": sql}
        # 预计算并缓存 embedding
        emb = self._compute_embedding(question)
        if emb is not None:
            self._training_embeddings[_id] = emb
        return _id

    def _tokenize(self, text: str) -> set:
        """中文分词，返回有意义的 token 集合"""
        try:
            import jieba
            tokens = set(w.strip() for w in jieba.cut(text) if len(w.strip()) > 1)
        except ImportError:
            # fallback：对非中文按空格分词，中文用二元组
            words = text.split()
            if len(words) > 1:
                tokens = set(words)
            else:
                tokens = set(text[i:i+2] for i in range(len(text) - 1))
        # 过滤常见停用词
        stop_words = {
            "的", "了", "是", "在", "有", "和", "与", "或",
            "吗", "吧", "啊", "呢", "哈", "呀", "哦",
            "什么", "怎么", "哪些", "哪个", "请问", "是否",
            "这个", "那个", "可以", "没有", "不是", "就是",
            "一下", "情况", "相关", "关于", "搜索", "查找",
        }
        return tokens - stop_words

    def _jaccard_similarity(self, question: str) -> List[tuple]:
        """jieba 分词 + Jaccard 关键词匹配"""
        results = []
        q_tokens = self._tokenize(question.lower())
        if not q_tokens:
            return results
        for _id, item in self._training_data.items():
            train_tokens = self._tokenize(item["question"].lower())
            if not train_tokens:
                continue
            overlap = q_tokens & train_tokens
            if overlap:
                score = len(overlap) / len(q_tokens | train_tokens)
                results.append((score, _id, item))
        return results

    def _embedding_similarity(self, question: str) -> List[tuple]:
        """语义 embedding 余弦相似度匹配"""
        if not self._training_embeddings:
            return []
        q_emb = self._compute_embedding(question)
        if q_emb is None:
            return []
        import numpy as np
        results = []
        for _id, item in self._training_data.items():
            t_emb = self._training_embeddings.get(_id)
            if t_emb is None:
                continue
            # cosine similarity (已经 normalize_embeddings=True，点积即可)
            score = float(np.dot(q_emb, t_emb))
            results.append((score, _id, item))
        return results

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        混合检索：jieba Jaccard + embedding 语义匹配

        融合策略：
        - 先用 jieba Jaccard 得到关键词相似度（权重 0.3）
        - 再用 embedding 得到语义相似度（权重 0.7）
        - 加权融合后排序，返回 top-5

        这样既保留了关键词精确匹配的能力，又获得了语义泛化的优势。
        """
        # 1. Jaccard 匹配
        jaccard_results = self._jaccard_similarity(question)
        # 2. Embedding 匹配
        embed_results = self._embedding_similarity(question)

        # 3. 融合排序
        fused = {}  # _id -> {item, jaccard_score, embed_score}
        for score, _id, item in jaccard_results:
            if _id not in fused:
                fused[_id] = {"item": item, "jaccard": 0.0, "embed": 0.0}
            fused[_id]["jaccard"] = score

        for score, _id, item in embed_results:
            if _id not in fused:
                fused[_id] = {"item": item, "jaccard": 0.0, "embed": 0.0}
            fused[_id]["embed"] = score

        # 归一化后加权融合
        jaccard_scores = [v["jaccard"] for v in fused.values()]
        embed_scores = [v["embed"] for v in fused.values()]
        j_max = max(jaccard_scores) if jaccard_scores else 1
        e_max = max(embed_scores) if embed_scores else 1

        scored_items = []
        for _id, v in fused.items():
            j_norm = v["jaccard"] / j_max if j_max > 0 else 0
            e_norm = v["embed"] / e_max if e_max > 0 else 0
            # 加权融合：embedding 权重更高，因为语义更准确
            combined = 0.3 * j_norm + 0.7 * e_norm
            scored_items.append((combined, v["item"]))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        # top_k 从 3 扩到 5，给 LLM 更多参考
        return [item for score, item in scored_items[:5]]

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        rows = []
        for _id, item in self._training_data.items():
            rows.append({"id": _id, "question": item["question"], "sql": item["sql"]})
        return pd.DataFrame(rows)

    def remove_training_data(self, id: str, **kwargs) -> bool:
        return self._training_data.pop(id, None) is not None
