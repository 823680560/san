# -*- coding: utf-8 -*-
"""通用工具函数。"""

import re
from datetime import datetime


def get_model_short_name(model: str) -> str:
    """获取模型简写名称，用于报告文件命名。

    示例:
        deepseek-chat -> deepseek
        deepseek-v3 -> deepseek_v3
        gpt-4o -> gpt_4o
        bge-m3 -> bge_m3
    """
    short = model.replace("-chat", "").replace("-instruct", "")
    short = re.sub(r"[^a-zA-Z0-9]", "_", short)
    short = re.sub(r"_+", "_", short).strip("_")
    return short


def get_timestamp() -> str:
    """获取当前时间戳字符串，格式 YYYYMMDD_HHMM。"""
    return datetime.now().strftime("%Y%m%d_%H%M")
