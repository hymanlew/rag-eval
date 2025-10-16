# Dify RAG连接器实现

import aiohttp
import asyncio
import logging
from typing import Dict, Any, List
from .base import AsyncBaseRAGConnector

logger = logging.getLogger(__name__)

class DifyConnector(AsyncBaseRAGConnector):
    """Dify RAG系统连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Dify连接器"""
        super().__init__("Dify", config)
    
    def validate_config(self) -> List[str]:
        """验证Dify配置"""
        errors = []
        if not self.config.get("api_key"):
            errors.append("DIFY_API_KEY is required")
        if not self.config.get("base_url"):
            errors.append("DIFY_BASE_URL is required")
        return errors
    
    def build_request(self, question: str, **kwargs) -> Dict[str, Any]:
        """构建Dify API请求"""
        user_id = kwargs.get("user_id", self.config.get("user_id", "rag-evaluator"))
        
        return {
            "method": "POST",
            "url": f"{self.config['base_url']}/chat-messages",
            "headers": {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            },
            "body": {
                "inputs": {
                    "background": "I am a software developer working on Japanese apps",
                    "instruction": "Please help with software development questions"
                },
                "query": question,
                "response_mode": "streaming",
                "auto_generate_name": True,
                "user": user_id
            }
        }
    
    async def send_request_async(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步发送HTTP请求到Dify API"""
        headers = request_data["headers"]
        url = request_data["url"]
        body = request_data["body"]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body) as response:
                    if response.status == 200:
                        # 检查是否是streaming模式
                        if body.get("response_mode") == "streaming":
                            return await self._parse_streaming_response(response)
                        else:
                            return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Dify API error: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise Exception("Dify API请求超时")
        except Exception as e:
            raise Exception(f"Dify API请求失败: {str(e)}")
    
    async def _parse_streaming_response(self, response) -> Dict[str, Any]:
        """解析streaming响应"""
        import json
        
        message_events = []
        message_end_result = None
        
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # 移除 'data: ' 前缀
                    event_type = data.get("event")
                    
                    if event_type == "message":
                        message_events.append(data)
                    elif event_type == "message_end":
                        message_end_result = data
                        break  # 找到message_end后停止，这是最完整的响应
                        
                except json.JSONDecodeError:
                    continue
        
        # 合并所有message事件的答案
        full_answer = ""
        for msg in message_events:
            if "answer" in msg:
                full_answer += msg["answer"]
        
        # 使用message_end作为基础，它包含完整的metadata
        if message_end_result:
            final_result = message_end_result
            final_result["answer"] = full_answer
            return final_result
        elif message_events:
            # 如果没有message_end，使用最后一个message事件
            last_message = message_events[-1]
            last_message["answer"] = full_answer
            return last_message
        else:
            raise Exception("无法从streaming响应中解析出有效消息")

    async def query_async(self, question: str, max_retries: int = 2, **kwargs) -> Dict[str, Any]:
        """异步查询Dify系统"""
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
        """异步测试Dify连接"""
        try:
            result = await self.query_async("test connection", max_retries=1)
            return result.get("error") is None
        except Exception as e:
            logger.error(f"Dify连接测试失败: {e}")
            return False

    def parse_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析Dify响应"""
        answer = response_data.get("answer", "")
        contexts = []
        
        # 提取上下文信息
        if "metadata" in response_data and "retriever_resources" in response_data["metadata"]:
            contexts = [
                resource.get("content", "") 
                for resource in response_data["metadata"]["retriever_resources"]
                if resource.get("content")
            ]
        
        return {
            "answer": answer,
            "contexts": contexts,
            "error": None
        }