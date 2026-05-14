# -*- coding: utf-8 -*-
"""
LLM 包装器模块
==============
使用 langchain_openai 的 ChatOpenAI 完成大模型调用。
兼容 deepseek 等非 OpenAI 模型，提供统一的 complete 接口。
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleLLM:
    """
    简单的 LLM 包装器
    使用 langchain_openai 的 ChatOpenAI 完成大模型调用。
    提供与 llama_index LLM 兼容的 complete 接口。
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, temperature: float = 0):
        """
        初始化 LLM 包装器

        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
            temperature: 温度参数
        """
        self.model = model or LLM_MODEL
        self.api_key = api_key or LLM_API_KEY
        self.base_url = base_url or LLM_BASE_URL
        self.temperature = temperature

        self.client = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=4096,
            request_timeout=120.0,
        )

        logger.info(f"SimpleLLM 初始化完成，模型: {self.model}")

    def complete(self, prompt: str) -> "CompletionResponse":
        """
        调用 LLM 生成文本

        Args:
            prompt: 输入提示

        Returns:
            CompletionResponse 对象（包含 text 属性）
        """
        try:
            response = self.client.invoke(prompt)
            text = response.content
            return CompletionResponse(text=text)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise


class CompletionResponse:
    """
    模拟 llama_index 的 CompletionResponse
    """

    def __init__(self, text: str):
        self.text = text
