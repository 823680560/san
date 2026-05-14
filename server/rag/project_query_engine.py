import os
import re
import pandas as pd
import logging
import requests

import jieba

from server.rag.project_db import init_project_db, get_connection, import_excel_to_sqlite
from server.rag.vanna_setup import init_vanna_instances
from server.rag.sql_validator import SQLValidator
from configs.prompt_config import query_classifier_prompt
from configs.config import (
    PROJECT_EXCEL, PROJECT_SQLITE_DB,
    SQL_ROUTER_MODEL, SQL_ROUTER_API_KEY, SQL_ROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)

_MATCH_RE = re.compile(r"(MATCH\s+)(['\"])(.*?)\2", re.IGNORECASE)


def _segment_sql_match(sql: str) -> str:
    """对 SQL MATCH 子句做 jieba 分词 + 前缀匹配，消除城市名后缀（保定→保定市）等分词差异"""
    _FTS5_OPS = {"AND", "OR", "NOT", "NEAR"}
    def _segment(m: re.Match) -> str:
        prefix = m.group(1)
        quote = m.group(2)
        content = m.group(3)
        raw = content.replace(" ", "")
        tokens = [t for t in jieba.cut(raw) if t.strip()]
        # 给非运算符 token 加 * 前缀匹配，解决"保定" vs "保定市"等分词差异
        tokens = [f"{t}*" if t.upper() not in _FTS5_OPS else t for t in tokens]
        segmented = " ".join(tokens)
        return f"{prefix}{quote}{segmented}{quote}"
    return _MATCH_RE.sub(_segment, sql)


def _segment_query(query: str) -> str:
    """对用户查询做 jieba 分词，使 LLM 生成 SQL 时使用与 FTS5 入库一致的 token"""
    return " ".join(t for t in jieba.cut(query) if t.strip())


try:
    import sqlite3
    USING_SQLITE = True
except ImportError:
    USING_SQLITE = False


class ProjectQueryEngine:
    """查询路由引擎：分类器 + Vanna NL2SQL + SQLGlot校验 + 执行"""

    def __init__(self, db_path: str = None, llm_config: dict = None):
        self.db_path = db_path or PROJECT_SQLITE_DB
        self.llm_config = llm_config or {
            "model": SQL_ROUTER_MODEL,
            "api_key": SQL_ROUTER_API_KEY,
            "base_url": SQL_ROUTER_BASE_URL,
        }
        self.sqlite_conn = init_project_db(self.db_path)

        # 检查数据：Excel 存在时自动导入/更新（内部有 mtime 哈希比较，未变更时快速返回 0）
        if os.path.exists(PROJECT_EXCEL):
            imported = import_excel_to_sqlite(PROJECT_EXCEL, self.db_path)
            if imported > 0:
                logger.info(f"项目数据已更新，共 {imported} 条记录")
        else:
            count = self.sqlite_conn.execute("SELECT COUNT(*) FROM sanitation_projects").fetchone()[0]
            if count == 0:
                logger.warning(
                    f"项目数据库为空且 Excel 源文件不存在: {PROJECT_EXCEL}。"
                    f"请将 '环卫项目数据.xlsx' 放到 data/project/ 目录后重启。"
                )
        self.vanna_sqlite, self.vanna_duckdb, self.duckdb_conn = init_vanna_instances(
            self.llm_config, self.sqlite_conn, self.db_path
        )
        self.validator = SQLValidator()

    def query(self, user_query: str) -> dict:
        """执行 NL2SQL 查询

        Returns:
            {
                "query_type": "exact" | "fuzzy" | "aggregation",
                "sql": str,
                "results": list[dict],
                "row_count": int,
            }
            或 {"error": str, "sql": str} 当出错时
        """
        # Step 0: jieba 分词用户查询，使 LLM 看到与 FTS5 入库一致的 token
        segmented_query = _segment_query(user_query)

        # Step 1: 轻量分类（用原始查询，避免分类关键词被 jieba 拆开）
        query_type = self._classify(user_query)

        # Step 2: Vanna NL2SQL
        try:
            if query_type == "aggregation":
                raw_sql = self.vanna_duckdb.generate_sql(segmented_query, allow_llm_to_see_data=True)
                dialect = "duckdb"
            else:
                raw_sql = self.vanna_sqlite.generate_sql(segmented_query, allow_llm_to_see_data=True)
                dialect = "sqlite"
        except Exception as e:
            return {"error": f"NL2SQL 生成失败: {e}", "sql": ""}

        # Step 2.5: 对 MATCH 子句做 jieba 分词，保证与 FTS5 入库分词一致
        raw_sql = _segment_sql_match(raw_sql)

        # Step 3: SQLGlot 校验
        is_valid, error = self.validator.validate(raw_sql, dialect=dialect)
        if not is_valid:
            return {"error": f"SQL 校验失败: {error}", "sql": raw_sql}

        # Step 4: 执行
        try:
            if query_type == "aggregation":
                df = self.duckdb_conn.execute(raw_sql).fetchdf()
            else:
                df = pd.read_sql(raw_sql, self.sqlite_conn)
        except Exception as e:
            return {"error": f"执行失败: {e}", "sql": raw_sql}

        return {
            "query_type": query_type,
            "sql": raw_sql,
            "results": df.to_dict(orient="records"),
            "row_count": len(df),
        }

    def _classify(self, query: str) -> str:
        """规则 + LLM fallback 判断查询类型"""
        q = query.lower().strip()

        # --- 规则部分 ---
        aggr_keywords = [
            "排名", "平均", "总计", "统计", "趋势",
            "分布", "对比", "比较", "哪个最多", "哪个最少",
            "增长率", "同比", "环比", "占比", "比例",
            "排序", "最多", "最少", "汇总", "总共", "合计",
            "top", "avg", "sum", "count",
        ]
        fuzzy_keywords = [
            "搜索", "查找", "找", "查询",
            "哪些公司", "什么项目", "包含", "关键词",
            "相关", "关于",
        ]

        for kw in aggr_keywords:
            if kw in q:
                return "aggregation"
        for kw in fuzzy_keywords:
            if kw in q:
                return "fuzzy"

        # --- LLM fallback ---
        try:
            prompt = query_classifier_prompt(query)
            result = self._call_llm(prompt).strip().lower()
            if result in ("exact", "fuzzy", "aggregation"):
                return result
        except Exception as e:
            logger.warning(f"LLM fallback 分类失败: {e}")

        return "exact"

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（与 Vanna 使用相同的配置）"""
        response = requests.post(
            f"{self.llm_config['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {self.llm_config['api_key']}"},
            json={
                "model": self.llm_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# 全局单例
_engine_instance = None


def get_query_engine() -> ProjectQueryEngine:
    """获取全局 ProjectQueryEngine 单例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ProjectQueryEngine()
    return _engine_instance
