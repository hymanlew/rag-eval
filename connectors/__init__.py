# RAG连接器模块 - 策略模式 + 工厂模式

from .universal import UniversalRAGConnector
from .factory import RAGConnectorFactory
from .base import BaseRAGConnector
from .dify import DifyConnector
from .ragflow import RagFlowConnector

__all__ = [
    'UniversalRAGConnector',
    'RAGConnectorFactory', 
    'BaseRAGConnector',
    'DifyConnector',
    'RagFlowConnector'
]
