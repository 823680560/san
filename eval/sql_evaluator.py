# -*- coding: utf-8 -*-
"""
NL2SQL 评估模块（四个核心指标）
================================
  VE  — Valid Execution:     SQL 能否无错解析并执行
  EX  — Execution Accuracy:  执行结果与 ground truth 是否一致（忽略行序）
  CM  — Component Match:     SELECT / WHERE / GROUP / ORDER / KEYWORDS 五组件打分
  SIM — SQL Similarity:      叶子节点（表名/列名/字面值/算子）Jaccard 相似度
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

import sqlglot

logger = logging.getLogger(__name__)

COMPONENT_WEIGHTS = {
    "select":   0.20,
    "where":    0.30,
    "group":    0.25,
    "order":    0.15,
    "keywords": 0.10,
}


@dataclass
class SQLMetrics:
    """NL2SQL 四指标集合"""
    ve: float = 0.0          # Valid Execution (0 or 1)
    ex: float = 0.0          # Execution Accuracy: results match (0 or 1)
    cm: float = 0.0          # Component Match (0 ~ 1, weighted)
    cm_select: float = 0.0
    cm_where: float = 0.0
    cm_group: float = 0.0
    cm_order: float = 0.0
    cm_keywords: float = 0.0
    sim: float = 0.0         # SQL Similarity: leaf-node Jaccard (0 ~ 1)

    def overall(self) -> float:
        """综合得分：VE + EX + CM + SIM 等权"""
        return round((self.ve + self.ex + self.cm + self.sim) / 4, 4)

    def to_dict(self) -> Dict:
        return {
            "ve": round(self.ve, 4),
            "ex": round(self.ex, 4),
            "cm": round(self.cm, 4),
            "cm_select": round(self.cm_select, 4),
            "cm_where": round(self.cm_where, 4),
            "cm_group": round(self.cm_group, 4),
            "cm_order": round(self.cm_order, 4),
            "cm_keywords": round(self.cm_keywords, 4),
            "sim": round(self.sim, 4),
            "overall": self.overall(),
        }


class SqlEvaluator:
    """NL2SQL 质量评估器 — 四指标（VE / EX / CM / SIM）"""

    def evaluate_single(
        self,
        query: str = "",
        sql: str = "",
        sql_error: str = "",
        row_count: int = 0,
        expected_sql: str = "",
        expected_results: Optional[List[Dict]] = None,
        generated_results: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        评估单条 SQL 的四项指标。

        Args:
            query: 用户问题（保留参数兼容，CM/SIM 不需要）
            sql: 生成的 SQL
            sql_error: 执行错误信息（空串 = 执行成功）
            row_count: 返回行数（保留兼容）
            expected_sql: 期望 SQL（CM / SIM 需要）
            expected_results: 期望结果（EX 需要）
            generated_results: 实际结果（EX 需要）

        Returns:
            SQLMetrics 的 dict 形式
        """
        metric = SQLMetrics()

        # ---- VE: 能跑吗 ----
        metric.ve = 1.0 if not sql_error else 0.0

        # ---- EX: 结果对吗 ----
        metric.ex = self._evaluate_ex(expected_results, generated_results)

        # ---- CM: 哪写错了 ----
        if expected_sql:
            cm_scores = self._component_match(sql, expected_sql)
            metric.cm_select = cm_scores["select"]
            metric.cm_where = cm_scores["where"]
            metric.cm_group = cm_scores["group"]
            metric.cm_order = cm_scores["order"]
            metric.cm_keywords = cm_scores["keywords"]
            metric.cm = sum(
                cm_scores[k] * COMPONENT_WEIGHTS[k] for k in COMPONENT_WEIGHTS
            )

        # ---- SIM: 差多远 ----
        if expected_sql:
            metric.sim = self._leaf_similarity(sql, expected_sql)

        return metric.to_dict()

    # ================================================================
    #  EX
    # ================================================================

    def _evaluate_ex(
        self, expected: Optional[List[Dict]], generated: Optional[List[Dict]]
    ) -> float:
        """两个结果集作为无序集合是否一致（含列名关联）。

        Returns 1.0 一致，0.0 不一致。
        """
        if not expected and not generated:
            return 1.0
        if not expected or not generated:
            return 0.0

        exp_set = {self._normalize_row(r) for r in expected}
        gen_set = {self._normalize_row(r) for r in generated}
        return 1.0 if exp_set == gen_set else 0.0

    @staticmethod
    def _normalize_row(row: Dict) -> frozenset:
        """保留列名+值的对应关系，避免列值碰撞"""
        return frozenset(
            (k, str(v) if v is not None else "\0SQLNULL\0")
            for k, v in sorted(row.items())
        )

    # ================================================================
    #  CM — Component Match（Spider 标准，5 组件）
    # ================================================================

    def _component_match(self, generated_sql: str, expected_sql: str) -> Dict[str, float]:
        """基于 sqlglot AST 做组件级匹配，每组件返回 0.0 或 1.0"""
        try:
            gen = sqlglot.parse_one(generated_sql, dialect='sqlite')
            exp = sqlglot.parse_one(expected_sql, dialect='sqlite')
            if gen is None or exp is None:
                return {"select": 0, "where": 0, "group": 0, "order": 0, "keywords": 0}
        except Exception:
            # 解析失败：fallback 到 SIM 值（全部 0）
            return {"select": 0, "where": 0, "group": 0, "order": 0, "keywords": 0}

        return {
            "select":   1.0 if self._select_cols(gen) == self._select_cols(exp) else 0.0,
            "where":    1.0 if self._where_conds(gen) == self._where_conds(exp) else 0.0,
            "group":    1.0 if self._group_cols(gen) == self._group_cols(exp) else 0.0,
            "order":    1.0 if self._order_specs(gen) == self._order_specs(exp) else 0.0,
            "keywords": 1.0 if self._keywords(gen) == self._keywords(exp) else 0.0,
        }

    # --- SELECT ---

    @staticmethod
    def _select_cols(ast: sqlglot.exp.Expression) -> Set[str]:
        """提取 SELECT 列名集合（忽略别名、忽略顺序）"""
        cols: Set[str] = set()
        for expr in ast.expressions if hasattr(ast, 'expressions') else []:
            if isinstance(expr, sqlglot.exp.Star):
                cols.add("*")
            elif isinstance(expr, sqlglot.exp.Column):
                cols.add(expr.name)
            elif isinstance(expr, sqlglot.exp.Alias):
                if isinstance(expr.this, sqlglot.exp.Column):
                    cols.add(expr.this.name)
                else:
                    cols.add(str(expr.this))
            else:
                # 聚合/函数等：保留文本表示
                cols.add(str(expr).strip())
        return cols

    # --- WHERE ---

    @staticmethod
    def _where_conds(ast: sqlglot.exp.Expression) -> Set[tuple]:
        """提取 WHERE 条件为 (算子, 左操作数, 右操作数) 三元组集合"""
        conds: Set[tuple] = set()
        where_expr = ast.args.get("where")
        if where_expr is None:
            return conds

        all_nodes = list(where_expr.walk())

        # 找出被 Not 包裹的节点（穿透 Paren），统一加 NOT_ 前缀
        negated_ids = set()
        for n in all_nodes:
            if isinstance(n, sqlglot.exp.Not):
                inner = n.this
                while isinstance(inner, sqlglot.exp.Paren):
                    inner = inner.this
                negated_ids.add(id(inner))

        for node in all_nodes:
            kind = type(node)
            negated = id(node) in negated_ids

            try:
                left = str(node.this).strip() if hasattr(node, 'this') else ""
            except Exception:
                left = ""
            try:
                right = str(node.expression).strip() if hasattr(node, 'expression') else ""
            except Exception:
                right = ""

            if kind in {
                sqlglot.exp.EQ, sqlglot.exp.NEQ,
                sqlglot.exp.GT, sqlglot.exp.GTE,
                sqlglot.exp.LT, sqlglot.exp.LTE,
            }:
                op = f"NOT_{kind.__name__}" if negated else kind.__name__
                conds.add((op, left, right))
            elif kind is sqlglot.exp.Is:
                # IS NULL / IS NOT NULL
                op = "NOT_Is" if negated else "Is"
                conds.add((op, left, right))
            elif kind is sqlglot.exp.Match:
                op = "NOT_MATCH" if negated else "MATCH"
                conds.add((op, left, right))
            elif kind is sqlglot.exp.Like:
                op = "NOT_LIKE" if negated else "LIKE"
                conds.add((op, left, right))
            elif kind is sqlglot.exp.Between:
                low = SqlEvaluator._safe_str(node.args.get("low"))
                high = SqlEvaluator._safe_str(node.args.get("high"))
                op = "NOT_BETWEEN" if negated else "BETWEEN"
                conds.add((op, left, f"{low},{high}"))
            elif kind is sqlglot.exp.In:
                in_values = node.args.get("expressions")
                if in_values is not None:
                    right = f"({', '.join(SqlEvaluator._safe_str(v) for v in in_values)})"
                else:
                    right = SqlEvaluator._safe_str(node.args.get("query"))
                op = "NOT_IN" if negated else "IN"
                conds.add((op, left, right))
            elif kind in {sqlglot.exp.And, sqlglot.exp.Or}:
                pass  # 递归 walk 已经覆盖子节点
            elif kind is sqlglot.exp.Not:
                pass  # 由包裹的内层节点以 negated=True 处理

        return conds

    @staticmethod
    def _safe_str(node) -> str:
        """安全地将 AST 节点转字符串"""
        if node is None:
            return ""
        return str(node).strip()

    # --- GROUP BY ---

    @staticmethod
    def _group_cols(ast: sqlglot.exp.Expression) -> Set[str]:
        group = ast.args.get("group")
        if group is None:
            return set()
        return {
            col.name if isinstance(col, sqlglot.exp.Column) else str(col).strip()
            for col in group.expressions if hasattr(group, 'expressions')
        }

    # --- ORDER BY ---

    @staticmethod
    def _order_specs(ast: sqlglot.exp.Expression) -> Set[tuple]:
        order = ast.args.get("order")
        if order is None:
            return set()
        specs: Set[tuple] = set()
        for node in order.expressions if hasattr(order, 'expressions') else []:
            col = node.this.name if isinstance(node.this, sqlglot.exp.Column) else str(node.this).strip()
            desc = "DESC" if node.args.get("desc") else "ASC"
            specs.add((col, desc))
        return specs

    # --- KEYWORDS ---

    @staticmethod
    def _keywords(ast: sqlglot.exp.Expression) -> Set[str]:
        kw: Set[str] = set()
        if ast.args.get("distinct"):
            kw.add("DISTINCT")
        if ast.args.get("limit"):
            kw.add("LIMIT")
        if ast.args.get("having"):
            kw.add("HAVING")
        if ast.args.get("offset"):
            kw.add("OFFSET")
        return kw

    # ================================================================
    #  SIM — 叶子节点 Jaccard（替代 AST-key Jaccard）
    # ================================================================

    def _leaf_similarity(self, generated_sql: str, expected_sql: str) -> float:
        """收集两种 SQL 的有语义意义的叶子 token 集合，计算 Jaccard 相似度"""
        try:
            gen_tokens = self._leaf_tokens(sqlglot.parse_one(generated_sql, dialect='sqlite'))
            exp_tokens = self._leaf_tokens(sqlglot.parse_one(expected_sql, dialect='sqlite'))
            if not gen_tokens or not exp_tokens:
                return 0.0
            return len(gen_tokens & exp_tokens) / len(gen_tokens | exp_tokens)
        except Exception as e:
            logger.warning(f"SIM _leaf_tokens 解析失败，回退到 keyword_jaccard: {e}")
            return self._keyword_jaccard(generated_sql, expected_sql)

    @staticmethod
    def _leaf_tokens(ast) -> Set[str]:
        """收集 AST 中有语义意义的叶子节点 token

        每类 token 加前缀以区分不同语义层：
          TABLE:<表名>    表引用
          COL:<列名>      列引用
          VAL:<值>        字面常量
          OP:<算子>       比较/逻辑算子
          AGG:<聚合函数>  聚合算子
        """
        tokens: Set[str] = set()
        if ast is None:
            return tokens

        all_nodes = list(ast.walk())

        negated_ids = set()
        for n in all_nodes:
            if isinstance(n, sqlglot.exp.Not):
                inner = n.this
                while isinstance(inner, sqlglot.exp.Paren):
                    inner = inner.this
                negated_ids.add(id(inner))

        for node in all_nodes:
            negated = id(node) in negated_ids

            if isinstance(node, sqlglot.exp.Table):
                tokens.add(f"TABLE:{node.name}")
            elif isinstance(node, sqlglot.exp.Column):
                tokens.add(f"COL:{node.name}")
            elif isinstance(node, (sqlglot.exp.Literal, sqlglot.exp.Boolean)):
                val = str(node.this).strip()
                if val:
                    tokens.add(f"VAL:{val}")
            elif isinstance(node, (
                sqlglot.exp.EQ, sqlglot.exp.NEQ,
                sqlglot.exp.GT, sqlglot.exp.GTE,
                sqlglot.exp.LT, sqlglot.exp.LTE,
                sqlglot.exp.Like, sqlglot.exp.Between, sqlglot.exp.In,
                sqlglot.exp.Match, sqlglot.exp.Is,
            )):
                key = f"not_{node.key}" if negated else node.key
                tokens.add(f"OP:{key}")
            elif isinstance(node, (sqlglot.exp.And, sqlglot.exp.Or)):
                tokens.add(f"OP:{node.key}")
            elif isinstance(node, (
                sqlglot.exp.Sum, sqlglot.exp.Avg,
                sqlglot.exp.Count, sqlglot.exp.Max, sqlglot.exp.Min,
            )):
                tokens.add(f"AGG:{node.key}")
            elif isinstance(node, sqlglot.exp.Ordered):
                direction = "DESC" if node.args.get("desc") else "ASC"
                tokens.add(f"OP:ORDER_{direction}")
        return tokens

    @staticmethod
    def _keyword_jaccard(sql_a: str, sql_b: str) -> float:
        """fallback：SQL 关键词集合 Jaccard"""
        import re
        kw_a = set(re.findall(r'\b[A-Za-z_]\w*\b', sql_a))
        kw_b = set(re.findall(r'\b[A-Za-z_]\w*\b', sql_b))
        if not kw_a or not kw_b:
            return 0.0
        return len(kw_a & kw_b) / len(kw_a | kw_b)

    # ================================================================
    #  汇总
    # ================================================================

    @staticmethod
    def aggregate_metrics(sql_metrics_list: List[Dict]) -> Dict:
        """汇总多条查询的 SQL 评估指标"""
        n = len(sql_metrics_list)
        if n == 0:
            return {
                "ve": 0.0, "ex": 0.0, "cm": 0.0, "sim": 0.0, "overall": 0.0,
            }
        keys = ["ve", "ex", "cm",
                "cm_select", "cm_where", "cm_group", "cm_order", "cm_keywords",
                "sim", "overall"]
        result = {}
        for k in keys:
            vals = [m.get(k, 0) for m in sql_metrics_list]
            result[k] = round(sum(vals) / n, 4)
        return result
