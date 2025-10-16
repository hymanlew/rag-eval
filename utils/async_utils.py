# 异步工具类 - 提供通用的异步操作工具

import asyncio
import time
import sys
import os
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import logging

# 添加路径以便导入配置管理器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class AsyncUtils:
    """异步工具类"""
    
    def __init__(self, config_manager=None):
        """
        初始化异步工具类
        
        Args:
            config_manager: 配置管理器实例
        """
        # 延迟导入以避免循环依赖
        if config_manager:
            self.config = config_manager
        else:
            try:
                from async_config import get_async_config
                self.config = get_async_config()
            except ImportError:
                self.config = None
    
    @staticmethod
    async def run_in_threadpool(func: Callable, *args, **kwargs) -> Any:
        """
        在线程池中运行同步函数
        
        Args:
            func: 要运行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数结果
        """
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, func, *args, **kwargs)
            return result
    
    async def gather_with_concurrency(self, tasks: List[asyncio.Task], concurrency: int = None) -> List[Any]:
        """
        带并发控制的gather操作
        
        Args:
            tasks: 任务列表
            concurrency: 并发数
            
        Returns:
            结果列表
        """
        # 如果没有指定并发数，使用配置中的默认值
        if concurrency is None:
            if self.config:
                concurrency = self.config.get_concurrency_config()["max_concurrency"]
            else:
                concurrency = 3  # 默认值
        
        results = []
        
        # 分批处理任务
        for i in range(0, len(tasks), concurrency):
            batch = tasks[i:i + concurrency]
            try:
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"批量任务执行失败: {e}")
                # 对于失败的任务，添加None作为结果
                results.extend([None] * len(batch))
        
        return results
    
    async def retry_async(
        self,
        func: Callable,
        max_retries: int = None,
        delay: float = None,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,)
    ) -> Any:
        """
        异步重试机制
        
        Args:
            func: 要重试的异步函数
            max_retries: 最大重试次数
            delay: 初始延迟（秒）
            backoff: 延迟倍数
            exceptions: 需要重试的异常类型
            
        Returns:
            函数结果
        """
        # 使用配置中的默认值
        if max_retries is None:
            if self.config:
                max_retries = self.config.get_retry_config()["retry_attempts"]
            else:
                max_retries = 3
        
        if delay is None:
            if self.config:
                delay = self.config.get_retry_config()["retry_delay"]
            else:
                delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except exceptions as e:
                if attempt == max_retries:
                    logger.error(f"异步重试失败（{max_retries}次）: {e}")
                    raise
                
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"第{attempt + 1}次重试失败，{wait_time}秒后重试: {e}")
                await asyncio.sleep(wait_time)
    
    @staticmethod
    def create_timeout_handler(timeout: int, operation_name: str = "操作"):
        """
        创建超时处理器
        
        Args:
            timeout: 超时时间（秒）
            operation_name: 操作名称
            
        Returns:
            超时处理器函数
        """
        async def timeout_handler():
            await asyncio.sleep(timeout)
            raise asyncio.TimeoutError(f"{operation_name}超时（{timeout}秒）")
        
        return timeout_handler
    
    @staticmethod
    async def execute_with_timeout(
        coro,
        timeout: int,
        operation_name: str = "操作",
        on_timeout: Optional[Callable] = None
    ) -> Any:
        """
        执行带超时的协程
        
        Args:
            coro: 要执行的协程
            timeout: 超时时间（秒）
            operation_name: 操作名称
            on_timeout: 超时回调函数
            
        Returns:
            协程结果
        """
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError as e:
            logger.warning(f"{operation_name}超时: {e}")
            if on_timeout:
                return await on_timeout()
            raise
    
    @staticmethod
    async def batch_process(
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10,
        concurrency: int = 3,
        **kwargs
    ) -> List[Any]:
        """
        批量异步处理
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数
            batch_size: 批次大小
            concurrency: 并发数
            **kwargs: 额外参数
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 分批次处理
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # 为每个批次创建任务
            tasks = []
            for item in batch:
                task = asyncio.create_task(process_func(item, **kwargs))
                tasks.append(task)
            
            # 并发执行批次任务
            batch_results = await AsyncUtils.gather_with_concurrency(tasks, concurrency)
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    class ProgressTracker:
        """进度跟踪器"""
        
        def __init__(self, total: int, name: str = "任务"):
            self.total = total
            self.completed = 0
            self.name = name
            self.start_time = time.time()
            self.last_update = time.time()
        
        def update(self, increment: int = 1):
            """更新进度"""
            self.completed += increment
            current_time = time.time()
            
            # 限制更新频率，避免过多日志
            if current_time - self.last_update >= 1.0:  # 每秒最多更新一次
                progress = (self.completed / self.total) * 100
                elapsed = current_time - self.start_time
                logger.info(f"{self.name}进度: {self.completed}/{self.total} ({progress:.1f}%) - 耗时: {elapsed:.1f}秒")
                self.last_update = current_time
        
        def finish(self):
            """完成进度跟踪"""
            elapsed = time.time() - self.start_time
            logger.info(f"{self.name}完成: {self.completed}/{self.total} - 总耗时: {elapsed:.1f}秒")
    
    @staticmethod
    async def measure_execution_time(coro, operation_name: str = "操作"):
        """
        测量异步操作执行时间
        
        Args:
            coro: 要测量的协程
            operation_name: 操作名称
            
        Returns:
            (结果, 执行时间)
        """
        start_time = time.time()
        result = await coro
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"{operation_name}执行时间: {execution_time:.2f}秒")
        return result, execution_time