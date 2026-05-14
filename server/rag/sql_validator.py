import re
import sqlglot


class SQLValidator:
    """SQLGlot 安全校验：解析 + 操作白名单"""

    ALLOWED_STMT_TYPES = {"select", "with"}

    BLOCKED_PATTERNS = [
        r'\binsert\b', r'\bupdate\b', r'\bdelete\b',
        r'\bdrop\b', r'\balter\b', r'\bcreate\b',
        r'\battach\b', r'\bdetach\b', r'\bpragm?a\b',
        r'\bexecute\b', r'\bexec\b', r'\bimport\b',
        r'\bcopy\b', r'\bvacuum\b', r'\breindex\b',
    ]

    def validate(self, sql: str, dialect: str = "sqlite") -> tuple:
        """校验 SQL 是否安全。

        Returns:
            (is_valid: bool, error_message: str)
        """
        # 空检查
        if not sql or not sql.strip():
            return False, "SQL 为空"

        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
        except Exception as e:
            return False, f"语法错误: {e}"

        if parsed.key.lower() not in self.ALLOWED_STMT_TYPES:
            return False, f"不允许的语句类型: {parsed.key}"

        sql_lower = sql.lower()
        for pat in self.BLOCKED_PATTERNS:
            if re.search(pat, sql_lower):
                return False, f"包含禁止操作: {pat}"

        return True, ""
