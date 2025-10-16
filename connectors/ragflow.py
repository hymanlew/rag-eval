# RagFlow RAG连接器实现

import aiohttp
import asyncio
import logging
from typing import Dict, Any, List
from .base import AsyncBaseRAGConnector

logger = logging.getLogger(__name__)

class RagFlowConnector(AsyncBaseRAGConnector):
    """RagFlow RAG系统连接器"""
    
    def validate_config(self) -> List[str]:
        """验证RagFlow配置"""
        errors = []
        if not self.config.get("api_key"):
            errors.append("RAGFLOW_API_KEY is required")
        if not self.config.get("base_url"):
            errors.append("RAGFLOW_BASE_URL is required")
        return errors
    
    def build_request(self, question: str, **kwargs) -> Dict[str, Any]:
        """构建RagFlow API请求"""
        return {
            "method": "POST",
            "url": f"{self.config['base_url']}/api/v1/completion",
            "headers": {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            },
            "body": {
                "question": question,
                "streaming": False
            }
        }
    
    async def send_request_async(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步发送HTTP请求到RagFlow API"""
        headers = request_data["headers"]
        url = request_data["url"]
        body = request_data["body"]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"RagFlow API error: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise Exception("RagFlow API请求超时")
        except Exception as e:
            raise Exception(f"RagFlow API请求失败: {str(e)}")

    async def query_async(self, question: str, max_retries: int = 2, **kwargs) -> Dict[str, Any]:
        """异步查询RagFlow系统"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.async_utils import AsyncUtils
        
        # 创建AsyncUtils实例
        async_utils = AsyncUtils()
        
        request_data = self.build_request(question, **kwargs)
        
        async def make_request():
            response_data = await self.send_request_async(request_data)
            return self.parse_response(response_data)
        
        # 使用异步重试机制
        try:
            result = await async_utils.retry_async(
                make_request,
                max_retries=max_retries,
                delay=1.0,
                exceptions=(Exception,)
            )
            return result
        except Exception as e:
            return {"answer": "", "contexts": [], "error": str(e)}

    async def test_connection_async(self) -> bool:
        """异步测试RagFlow连接"""
        try:
            result = await self.query_async("test connection", max_retries=1)
            return result.get("error") is None
        except Exception as e:
            logger.error(f"RagFlow连接测试失败: {e}")
            return False

    def parse_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析RagFlow响应"""
        answer = ""
        contexts = []
        
        # 提取答案和上下文
        if "data" in response_data:
            data = response_data["data"]
            answer = data.get("answer", "")
            contexts = data.get("chunks", [])
        
        return {
            "answer": answer,
            "contexts": contexts,
            "error": None
        }