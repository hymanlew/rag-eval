# 异步连接器基类 - 为RAG系统提供异步接口

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import logging

logger = logging.getLogger(__name__)

class AsyncBaseRAGConnector(ABC):
    """异步连接器基类"""
    
    def __init__(self, system_name: str, config: Dict[str, Any]):
        """
        初始化异步连接器
        
        Args:
            system_name: 系统名称
            config: 配置信息
        """
        self.system_name = system_name
        self.config = config
        self.timeout = config.get('timeout', 30)
        
        logger.info(f"Async connector initialized for {system_name}")
    
    @abstractmethod
    async def query_async(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        异步查询RAG系统
        
        Args:
            question: 查询问题
            **kwargs: 额外参数
            
        Returns:
            {"answer": str, "contexts": list, "error": str}
        """
        pass
    
    @abstractmethod
    async def test_connection_async(self) -> bool:
        """
        异步测试连接
        
        Returns:
            连接是否成功
        """
        pass
    
    async def query_with_timeout(self, question: str, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        带超时的异步查询
        
        Args:
            question: 查询问题
            timeout: 超时时间（秒）
            **kwargs: 额外参数
            
        Returns:
            查询结果或错误信息
        """
        timeout = timeout or self.timeout
        
        try:
            result = await asyncio.wait_for(
                self.query_async(question, **kwargs),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            error_msg = f"查询超时（{timeout}秒）"
            logger.warning(f"{self.system_name} {error_msg}")
            return {"answer": "", "contexts": [], "error": error_msg}
        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            logger.error(f"{self.system_name} {error_msg}")
            return {"answer": "", "contexts": [], "error": error_msg}
    
    async def test_connection_with_timeout(self, timeout: Optional[int] = None) -> bool:
        """
        带超时的异步连接测试
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            连接是否成功
        """
        timeout = timeout or self.timeout
        
        try:
            return await asyncio.wait_for(
                self.test_connection_async(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"{self.system_name} 连接测试超时（{timeout}秒）")
            return False
        except Exception as e:
            logger.error(f"{self.system_name} 连接测试失败: {str(e)}")
            return False
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误信息列表，空列表表示验证通过
        """
        pass
    
    @abstractmethod
    def build_request(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        构建请求
        
        Args:
            question: 查询问题
            **kwargs: 额外参数
            
        Returns:
            请求字典
        """
        pass
    
    @abstractmethod
    def parse_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析响应
        
        Args:
            response_data: 响应数据
            
        Returns:
            解析后的结果
        """
        pass
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        return {
            "system_name": self.system_name,
            "timeout": self.timeout,
            "config": {k: v for k, v in self.config.items() if k != 'api_key'}
        }

# 向后兼容的别名
BaseRAGConnector = AsyncBaseRAGConnector