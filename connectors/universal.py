# 通用RAG连接器 - 基于策略模式和工厂模式

import logging
import re
import asyncio
from typing import Dict, Any
from .factory import RAGConnectorFactory

logger = logging.getLogger(__name__)

class UniversalRAGConnector:
    """通用RAG连接器 - 基于策略模式支持多种RAG系统"""
    
    def __init__(self, system_name: str, config: Dict[str, Any]):
        """
        初始化通用连接器
        
        Args:
            system_name: RAG系统名称 (dify, ragflow等)
            config: 系统配置
        """
        self.system_name = system_name
        self.config = config
        
        # 使用工厂模式创建具体的连接器
        self.connector = RAGConnectorFactory.create_connector(system_name, config)
        
        logger.info(f"Universal connector initialized for {system_name}")
    
    def _extract_answer_from_think_tags(self, raw_answer: str) -> str:
        """
        从包含think标签的回答中提取实际回答内容
        
        Args:
            raw_answer: 包含think标签的原始回答
            
        Returns:
            think标签以外的实际回答内容
        """
        if not raw_answer:
            return raw_answer
            
        # 使用正则表达式提取think标签以外的内容
        # 模式：匹配<think>...</think>标签之间的内容，然后将其移除
        think_pattern = r'<think>.*?</think>'
        
        # 移除think标签及其内容
        clean_answer = re.sub(think_pattern, '', raw_answer, flags=re.DOTALL)
        
        # 清理多余的空白字符
        clean_answer = clean_answer.strip()
        
        # 如果移除think标签后内容为空，则返回原始回答
        if not clean_answer:
            return raw_answer
            
        return clean_answer

    async def query_async(self, question: str, max_retries: int = 2, **kwargs) -> Dict[str, Any]:
        """
        异步查询RAG系统

        Args:
            question: 要查询的问题
            max_retries: 最大重试次数
            **kwargs: 额外参数

        Returns:
            {"answer": str, "contexts": list, "error": str}
        """
        # 检查连接器是否支持异步查询
        if hasattr(self.connector, 'query_async'):
            result = await self.connector.query_async(question, max_retries, **kwargs)
        else:
            # 如果不支持异步，使用线程池运行同步查询
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.connector.query, question, max_retries, **kwargs)
        
        # 提取实际回答内容（移除think标签）
        if "answer" in result and result["answer"]:
            result["answer"] = self._extract_answer_from_think_tags(result["answer"])
        
        return result

    async def query_with_timeout(self, question: str, timeout: int = 30, **kwargs) -> Dict[str, Any]:
        """
        带超时的异步查询RAG系统

        Args:
            question: 要查询的问题
            timeout: 超时时间（秒）
            **kwargs: 额外参数

        Returns:
            {"answer": str, "contexts": list, "error": str}
        """
        try:
            result = await asyncio.wait_for(
                self.query_async(question, **kwargs),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"RAG查询超时（{timeout}秒）: {question[:100]}...")
            return {"answer": "", "contexts": [], "error": f"查询超时（{timeout}秒）"}

    def query(self, question: str, max_retries: int = 2, **kwargs) -> Dict[str, Any]:
        """
        同步查询RAG系统（向后兼容）

        Args:
            question: 要查询的问题
            max_retries: 最大重试次数
            **kwargs: 额外参数

        Returns:
            {"answer": str, "contexts": list, "error": str}
        """
        try:
            # 检查是否已经在事件循环中
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果在运行的事件循环中，使用线程池
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.query_async(question, max_retries, **kwargs))
                    return future.result()
            else:
                # 如果没有运行的事件循环，直接运行
                return asyncio.run(self.query_async(question, max_retries, **kwargs))
        except Exception as e:
            return {"answer": "", "contexts": [], "error": f"同步查询失败: {str(e)}"}
    
    async def test_connection_async(self) -> bool:
        """异步测试连接是否正常"""
        if hasattr(self.connector, 'test_connection_async'):
            return await self.connector.test_connection_async()
        else:
            # 如果不支持异步，使用线程池运行同步测试
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.connector.test_connection)
    
    def test_connection(self) -> bool:
        """同步测试连接是否正常（向后兼容）"""
        try:
            # 检查是否已经在事件循环中
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果在运行的事件循环中，使用线程池
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.test_connection_async())
                    return future.result()
            else:
                # 如果没有运行的事件循环，直接运行
                return asyncio.run(self.test_connection_async())
        except Exception as e:
            logger.error(f"同步连接测试失败: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return self.connector.get_system_info()

    async def batch_query_async(self, questions: list, timeout: int = 30, concurrency: int = 3) -> list:
        """
        批量异步查询RAG系统

        Args:
            questions: 问题列表
            timeout: 每个查询的超时时间（秒）
            concurrency: 并发数

        Returns:
            查询结果列表
        """
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.async_utils import AsyncUtils
        
        # 创建查询任务
        tasks = []
        for question in questions:
            task = asyncio.create_task(self.query_with_timeout(question, timeout))
            tasks.append(task)
        
        # 并发执行查询
        results = await AsyncUtils.gather_with_concurrency(tasks, concurrency)
        
        return results
