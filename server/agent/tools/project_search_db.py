import re
from langchain.tools import tool
from typing import Optional

from server.rag.project_query_engine import get_query_engine


def _sanitize(text: str) -> str:
    """去除非法 Unicode 代理字符"""
    return re.sub(r'[\ud800-\udfff]', '', str(text))


def _format_result(result: dict) -> str:
    """格式化查询结果为可读文本"""
    if "error" in result:
        return f"查询出错: {_sanitize(result['error'])}"

    lines = [f"查询类型: {result['query_type']}  |  SQL: {_sanitize(result['sql'])}"]
    lines.append(f"共 {result['row_count']} 条结果:\n")

    # 动态格式化：根据实际字段展示
    for row in result["results"][:10]:
        if "项目名称" in row:
            # 精确/模糊查询 → 详细格式
            items = [
                f"项目: {_sanitize(str(row.get('项目名称', '-')))}",
                f"省份: {_sanitize(str(row.get('省份', '-')))}",
            ]
            if row.get("中标公司"):
                items.append(f"中标: {_sanitize(str(row['中标公司']))}")
            if row.get("年化金额_元") and not _is_nan(row["年化金额_元"]):
                items.append(f"年化: {row['年化金额_元']:.0f}元")
            if row.get("项目类型"):
                items.append(f"类型: {_sanitize(str(row['项目类型']))}")
            if row.get("中标时间"):
                items.append(f"时间: {row['中标时间']}")
        else:
            # 聚合查询 → 动态展示所有字段
            items = [f"{k}: {_sanitize(str(v))}" for k, v in row.items() if v is not None]
        lines.append(" | ".join(items))
    return "\n".join(lines)


def _is_nan(val) -> bool:
    """判断是否为 NaN"""
    try:
        return val != val
    except Exception:
        return False


# 全局缓存
_project_search_tool_func: Optional[callable] = None


def get_project_search_db_tool():
    """工厂函数：返回 LangChain 兼容的 project_search_db 工具"""
    global _project_search_tool_func

    if _project_search_tool_func is None:
        engine = get_query_engine()

        @tool
        def project_search_db(query: str) -> str:
            """
            搜索环卫项目数据（招标公告、中标信息、项目预算、公司等）。支持按地区、金额、日期精确查询，关键词模糊搜索，以及统计分析。当用户询问类似项目案例、预算参考或中标情况时调用此工具。

            Args:
                query: 检索关键字或自然语言问题
            """
            result = engine.query(query)
            return _format_result(result)

        _project_search_tool_func = project_search_db

    return _project_search_tool_func
