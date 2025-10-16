# 通用嵌入适配器 - 支持多种嵌入模型提供商的无缝适配

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """嵌入模型提供商枚举"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class UniversalEmbeddingAdapter(ABC):
    """通用嵌入适配器抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = config.get("model", "unknown")
        self.timeout = config.get("timeout", 30)
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个文本"""
        pass
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文本"""
        pass
    
    async def embed_with_timeout(self, text: str, timeout: Optional[int] = None) -> List[float]:
        """带超时的嵌入"""
        timeout = timeout or self.timeout
        try:
            return await asyncio.wait_for(
                self.embed_query(text), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"嵌入超时: {timeout}秒")
            raise Exception(f"嵌入超时: {timeout}秒")

class LangChainEmbeddingAdapter(UniversalEmbeddingAdapter):
    """基于LangChain的嵌入适配器 - 推荐方案"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedder = self._create_langchain_embedder()
        self.provider = self._detect_provider()
    
    def _detect_provider(self) -> EmbeddingProvider:
        """检测嵌入模型提供商"""
        model = self.config.get("model", "").lower()
        base_url = self.config.get("base_url", "").lower()
        
        # Ollama检测策略
        # 1. 检查端口（Ollama默认使用11434）
        if ":11434" in base_url or base_url.endswith("11434"):
            return EmbeddingProvider.OLLAMA
        
        # 2. 检查URL路径中的ollama标识
        if "ollama" in base_url:
            return EmbeddingProvider.OLLAMA
        
        # 3. 检查已知Ollama模型名称
        ollama_models = ["nomic", "llama", "mistral", "mxbai", "all-minilm"]
        if any(name in model for name in ollama_models):
            return EmbeddingProvider.OLLAMA
        
        # OpenAI检测策略
        if "openai" in base_url or "text-embedding" in model:
            return EmbeddingProvider.OPENAI
        
        # 其他情况归类为自定义
        return EmbeddingProvider.CUSTOM
    
    def _create_langchain_embedder(self):
        """创建LangChain嵌入模型"""
        provider = self._detect_provider()
        
        try:
            if provider == EmbeddingProvider.OLLAMA:
                from langchain_community.embeddings import OllamaEmbeddings
                return OllamaEmbeddings(
                    base_url=self.config["base_url"],
                    model=self.config["model"]
                )
            elif provider == EmbeddingProvider.OPENAI:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    api_key=self.config.get("api_key"),
                    model=self.config["model"],
                    base_url=self.config.get("base_url")
                )
            else:
                # 尝试使用OpenAI格式作为默认
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    api_key=self.config.get("api_key"),
                    model=self.config["model"],
                    base_url=self.config.get("base_url")
                )
        except ImportError as e:
            logger.error(f"LangChain导入失败: {e}")
            raise Exception(f"不支持的嵌入模型配置: {self.config}")
    
    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个文本"""
        try:
            return await self.embedder.aembed_query(text)
        except Exception as e:
            logger.error(f"嵌入失败: {e}")
            raise Exception(f"嵌入失败: {str(e)}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文本"""
        try:
            return await self.embedder.aembed_documents(texts)
        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise Exception(f"批量嵌入失败: {str(e)}")

class DirectEmbeddingAdapter(UniversalEmbeddingAdapter):
    """直接HTTP嵌入适配器 - 轻量级方案"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = self._detect_provider()
        self.api_format = self._get_api_format()
    
    def _detect_provider(self) -> EmbeddingProvider:
        """检测提供商"""
        base_url = self.config.get("base_url", "").lower()
        if "ollama" in base_url:
            return EmbeddingProvider.OLLAMA
        else:
            return EmbeddingProvider.OPENAI  # 默认
    
    def _get_api_format(self) -> Dict[str, Any]:
        """获取API格式配置"""
        if self.provider == EmbeddingProvider.OLLAMA:
            return {
                "input_field": "prompt",
                "response_path": ["embedding"],
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.get('api_key', '')}"
                }
            }
        else:
            return {
                "input_field": "input",
                "response_path": ["data", 0, "embedding"],
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.get('api_key', '')}"
                }
            }
    
    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个文本"""
        url = f"{self.config['base_url'].rstrip('/')}/embeddings"
        
        payload = {
            "model": self.config["model"],
            self.api_format["input_field"]: text
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, headers=self.api_format["headers"], json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 根据响应路径提取嵌入向量
                        embedding = result
                        for key in self.api_format["response_path"]:
                            embedding = embedding[key]
                        
                        return embedding
                    else:
                        error_text = await response.text()
                        raise Exception(f"API错误: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"直接嵌入失败: {e}")
            raise Exception(f"嵌入失败: {str(e)}")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文本"""
        # 并发处理多个文本
        tasks = [self.embed_query(text) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)

class EmbeddingAdapterFactory:
    """嵌入适配器工厂"""
    
    @staticmethod
    def create_adapter(config: Dict[str, Any], use_langchain: bool = True) -> UniversalEmbeddingAdapter:
        """创建嵌入适配器"""
        if use_langchain:
            try:
                return LangChainEmbeddingAdapter(config)
            except Exception as e:
                logger.warning(f"LangChain适配器创建失败，回退到直接适配器: {e}")
                return DirectEmbeddingAdapter(config)
        else:
            return DirectEmbeddingAdapter(config)
    
    @staticmethod
    async def test_adapter(adapter: UniversalEmbeddingAdapter) -> bool:
        """测试适配器"""
        try:
            test_embedding = await adapter.embed_query("Hello world")
            if len(test_embedding) > 0:
                logger.info(f"✅ 嵌入适配器测试成功: {adapter.embedding_model} (维度: {len(test_embedding)})")
                return True
            else:
                logger.error(f"❌ 嵌入适配器测试失败: 返回空向量")
                return False
        except Exception as e:
            logger.error(f"❌ 嵌入适配器测试失败: {e}")
            return False

# 使用示例和便利函数
async def create_and_test_embedding(config: Dict[str, Any]) -> Optional[UniversalEmbeddingAdapter]:
    """创建并测试嵌入适配器的便利函数"""
    adapter = EmbeddingAdapterFactory.create_adapter(config)
    if await EmbeddingAdapterFactory.test_adapter(adapter):
        return adapter
    return None

# 智能配置检测
def detect_embedding_config(base_url: str, model: str, api_key: str = "") -> Dict[str, Any]:
    """智能检测嵌入配置"""
    config = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout": 30
    }
    
    # 使用相同的检测逻辑
    base_url_lower = base_url.lower()
    model_lower = model.lower()
    
    # Ollama检测策略
    if ":11434" in base_url_lower or base_url_lower.endswith("11434"):
        config["provider"] = "ollama"
    elif "ollama" in base_url_lower:
        config["provider"] = "ollama"
    elif any(name in model_lower for name in ["nomic", "llama", "mistral", "mxbai", "all-minilm"]):
        config["provider"] = "ollama"
    elif "openai" in base_url_lower or "text-embedding" in model_lower:
        config["provider"] = "openai"
    else:
        config["provider"] = "openai"  # 默认使用OpenAI格式
    
    return config