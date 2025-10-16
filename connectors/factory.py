# RAG连接器工厂 - 工厂模式实现

from typing import Dict, Any, Type
from .base import BaseRAGConnector
from .dify import DifyConnector
from .ragflow import RagFlowConnector

class RAGConnectorFactory:
    """RAG连接器工厂 - 工厂模式"""
    
    # 注册可用的连接器类型
    _connectors: Dict[str, Type[BaseRAGConnector]] = {
        "dify": DifyConnector,
        "ragflow": RagFlowConnector,
    }
    
    @classmethod
    def register_connector(cls, name: str, connector_class: Type[BaseRAGConnector]):
        """注册新的连接器类型"""
        cls._connectors[name] = connector_class
    
    @classmethod
    def create_connector(cls, system_name: str, config: Dict[str, Any]) -> BaseRAGConnector:
        """
        创建RAG连接器实例
        
        Args:
            system_name: RAG系统名称
            config: 系统配置
            
        Returns:
            RAG连接器实例
            
        Raises:
            ValueError: 不支持的RAG系统
        """
        if system_name not in cls._connectors:
            available = list(cls._connectors.keys())
            raise ValueError(f"Unsupported RAG system: {system_name}. Available: {available}")
        
        connector_class = cls._connectors[system_name]
        
        # 尝试新的异步构造函数，如果失败则使用旧的构造函数
        try:
            return connector_class(system_name, config)
        except TypeError:
            # 如果构造函数不接受system_name参数，使用旧的构造函数
            return connector_class(config)
    
    @classmethod
    def get_available_systems(cls) -> list:
        """获取所有可用的RAG系统"""
        return list(cls._connectors.keys())
    
    @classmethod
    def get_system_info(cls, system_name: str) -> Dict[str, Any]:
        """获取系统信息"""
        if system_name not in cls._connectors:
            return {"name": system_name, "available": False}
        
        connector_class = cls._connectors[system_name]
        
        # 尝试新的异步构造函数，如果失败则使用旧的构造函数
        try:
            temp_instance = connector_class(system_name, {})
        except TypeError:
            temp_instance = connector_class({})
        
        return temp_instance.get_system_info()