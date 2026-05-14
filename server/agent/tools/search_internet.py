import logging
import requests
from langchain.tools import tool
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from configs.prompt_config import WEB_SEARCH_SYSTEM_PROMPT, web_search_user_prompt


def _get_llm():
    """延迟获取 LLM 实例（使用函数内导入避免循环依赖）"""
    from configs.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        request_timeout=60.0,
    )


def compress_content(content: str, query: str):
    try:
        llm = _get_llm()
        if llm is None:
            return content

        messages = [
            {'role': 'system', 'content': WEB_SEARCH_SYSTEM_PROMPT},
            {'role': 'user', 'content': web_search_user_prompt(content, query)}
        ]
        logging.info("开始进行互联网信息压缩....")
        output = llm.invoke(messages)
        if isinstance(output, AIMessage):
            content = output.content
        else:
            content = output
    except Exception as e:
        logging.error("进行互联网的信息压缩操作异常", exc_info=e)
    return content


def search_engine_iter(query: str):
    header = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer c427032a692f465ea7a3e4c09e1517bd.5ZWEpwdGSYK8qBGJ'
    }
    
    data = {
        "search_query": query,
        "search_engine": "search_std",
        "search_intent": False,
        "count": 5,
        "search_recency_filter": "noLimit",
        "content_size": "medium",
    }
    
    try:
        response = requests.post(
            url='https://open.bigmodel.cn/api/paas/v4/web_search',
            json=data,
            headers=header,
            timeout=10
        )
        
        if response.status_code != 200:
            return f"智谱搜索请求失败: {response.status_code}"
            
        res_json = response.json()
        datas = res_json.get('search_result', [])
        
        if not datas:
            return "未在互联网上找到相关结果。"
            
        raw_content = []
        for i, item in enumerate(datas):
            # ✅ 提取所有关键字段
            title = item.get('title', '无标题')
            content = item.get('content', '')
            link = item.get('link', '')
            publish_date = item.get('publish_date', '未知日期')
            refer = item.get('refer', '') # 来源网站
            
            # 格式化成一个清晰的块，方便大模型阅读
            info_block = (
                f"[资料 {i+1}]\n"
                f"标题: {title}\n"
                f"发布时间: {publish_date}\n"
                f"来源: {refer}\n"
                f"摘要: {content}\n"
                f"链接: {link}"
            )
            raw_content.append(info_block)

            # 注释掉大模型对搜索的内容进行总结
            # raw_content = [
            #     f"资料{i}:{compress_content(data['raw_content'], query)}" for i, data in enumerate(datas) if
            #     data['raw_content'] is not None
            # ]
            # raw_content = [f"资料{i}:{data.get('content')}" for i, data in enumerate(datas) if
            #     data.get('content') is not None]
            
        return "\n\n---\n\n".join(raw_content)

    except Exception as e:
        logging.error(f"联网搜索出错: {str(e)}")
        return f"互联网搜索工具执行出错: {str(e)}"

_SEARCH_CACHE = {}

def clear_search_cache():
    """清除搜索缓存（每次新查询时调用）"""
    _SEARCH_CACHE.clear()

@tool
def search_internet(query: str) -> str:
    """
    互联网实时信息搜索工具。
    当用户询问最新的新闻、实时天气、股票行情、或者本地知识库（法律/项目）中不存在的信息时，调用此工具。
    注意：同一关键词只需调用一次，重复调用不会获得新信息。

    Args:
        query: 需要联网检索的关键字或问题。
    """
    try:
        if not query or len(query.strip()) == 0:
            return "搜索关键字不能为空。"
    except Exception as e:
        logging.error(f"参数校验出错: {str(e)}")
        return f"参数校验出错: {str(e)}"

    cache_key = query.strip()
    if cache_key in _SEARCH_CACHE:
        logging.info(f"联网搜索缓存命中: {cache_key[:50]}...")
        return _SEARCH_CACHE[cache_key]

    result = search_engine_iter(query)
    _SEARCH_CACHE[cache_key] = result
    return result

class SearchInternetInput(BaseModel):
    query: str = Field(description="在互联网上检索的query关键字")
