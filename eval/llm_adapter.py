# -*- coding: utf-8 -*-
"""
纯模型适配器模块（无检索）
==========================
功能：
1. 直接调用大语言模型回答问题，不提供任何检索上下文
2. 用于与 Agent RAG 模式进行对比评估
3. 使用与 Agent 相同的 LLM 模型，确保对比公平性
"""

import logging
from typing import Optional

from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from configs.prompt_config import llm_only_answer_prompt
from eval.llm_wrapper import SimpleLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LLMOnlyAdapter:
    """
    纯模型适配器
    直接调用大语言模型回答问题，不提供任何检索上下文。
    用于与 Agent RAG 模式进行对比评估。
    """

    def __init__(self, llm=None, model_name: str = None):
        """
        初始化纯模型适配器

        Args:
            llm: LlamaIndex LLM 实例。若为 None，则自动创建
            model_name: 模型名称，默认使用 config 中的配置
        """
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._create_llm(model_name)

    def _create_llm(self, model_name: str = None) -> SimpleLLM:
        """
        创建 LLM 实例（使用 SimpleLLM 包装器，兼容 deepseek 等非 OpenAI 模型）

        Args:
            model_name: 模型名称

        Returns:
            SimpleLLM 实例
        """
        model = model_name or LLM_MODEL
        logger.info(f"初始化纯模型适配器，模型: {model}")

        return SimpleLLM(
            model=model,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,  # 评估时使用 temperature=0 保证确定性
        )

    def generate_answer(self, query: str) -> str:
        """
        直接使用 LLM 生成回答（无检索上下文）

        Args:
            query: 用户问题

        Returns:
            生成的回答文本
        """
        prompt = llm_only_answer_prompt(query)

        try:
            response = self.llm.complete(prompt)
            answer = response.text
            logger.info(f"LLM Only 回答生成成功 (长度: {len(answer)} 字符)")
            return answer
        except Exception as e:
            logger.error(f"LLM Only 回答生成失败: {e}")
            return f"生成回答时出错: {e}"


