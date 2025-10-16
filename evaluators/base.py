# 评价器基类 - 为评价系统提供异步接口

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class BaseEvaluator(ABC):
    """评价器基类 - 支持异步API"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化评价器
        
        Args:
            name: 评价器名称
            config: 配置信息
        """
        self.name = name
        self.config = config
        self.timeout = config.get('timeout', 45)
        self._available = False
        
        logger.info(f"Evaluator initialized: {name}")
    
    @abstractmethod
    async def evaluate_answers_async(self, questions: List[str], answers: List[str], 
                                  ground_truths: List[str], contexts: List[List[str]] = None) -> Dict[str, List[float]]:
        """
        异步评价回答
        
        Args:
            questions: 问题列表
            answers: 回答列表
            ground_truths: 标准答案列表
            contexts: 上下文列表（可选）
            
        Returns:
            评价指标字典
        """
        pass
    
    @abstractmethod
    async def evaluate_single_answer_async(self, question: str, answer: str, 
                                        ground_truth: str, context: List[str] = None) -> Dict[str, float]:
        """
        异步评价单个回答
        
        Args:
            question: 问题
            answer: 回答
            ground_truth: 标准答案
            context: 上下文（可选）
            
        Returns:
            评价指标字典
        """
        pass
    
    async def evaluate_with_timeout(self, questions: List[str], answers: List[str], 
                                  ground_truths: List[str], contexts: List[List[str]] = None,
                                  timeout: Optional[int] = None) -> Dict[str, List[float]]:
        """
        带超时的异步评价
        
        Args:
            questions: 问题列表
            answers: 回答列表
            ground_truths: 标准答案列表
            contexts: 上下文列表（可选）
            timeout: 超时时间（秒）
            
        Returns:
            评价指标字典
        """
        timeout = timeout or self.timeout
        
        try:
            result = await asyncio.wait_for(
                self.evaluate_answers_async(questions, answers, ground_truths, contexts),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            error_msg = f"评价超时（{timeout}秒）"
            logger.warning(f"{self.name} {error_msg}")
            # 返回默认值
            return self._get_default_scores(len(answers))
        except Exception as e:
            error_msg = f"评价失败: {str(e)}"
            logger.error(f"{self.name} {error_msg}")
            return self._get_default_scores(len(answers))
    
    async def evaluate_single_with_timeout(self, question: str, answer: str, 
                                         ground_truth: str, context: List[str] = None,
                                         timeout: Optional[int] = None) -> Dict[str, float]:
        """
        带超时的单个回答异步评价
        
        Args:
            question: 问题
            answer: 回答
            ground_truth: 标准答案
            context: 上下文（可选）
            timeout: 超时时间（秒）
            
        Returns:
            评价指标字典
        """
        timeout = timeout or self.timeout
        
        try:
            result = await asyncio.wait_for(
                self.evaluate_single_answer_async(question, answer, ground_truth, context),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            error_msg = f"单个评价超时（{timeout}秒）"
            logger.warning(f"{self.name} {error_msg}")
            return self._get_default_single_score()
        except Exception as e:
            error_msg = f"单个评价失败: {str(e)}"
            logger.error(f"{self.name} {error_msg}")
            return self._get_default_single_score()
    
    def _get_default_scores(self, count: int) -> Dict[str, List[float]]:
        """
        获取默认评分
        
        Args:
            count: 评分数量
            
        Returns:
            默认评分字典
        """
        metrics = self.get_supported_metrics()
        return {metric: [0.0] * count for metric in metrics}
    
    def _get_default_single_score(self) -> Dict[str, float]:
        """
        获取默认单个评分
        
        Returns:
            默认单个评分字典
        """
        metrics = self.get_supported_metrics()
        return {metric: 0.0 for metric in metrics}
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        获取支持的评价指标
        
        Returns:
            评价指标列表
        """
        pass
    
    def is_available(self) -> bool:
        """
        检查评价器是否可用
        
        Returns:
            是否可用
        """
        return self._available
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        """
        获取评价器信息
        
        Returns:
            评价器信息字典
        """
        return {
            "name": self.name,
            "timeout": self.timeout,
            "available": self._available,
            "supported_metrics": self.get_supported_metrics(),
            "config": {k: v for k, v in self.config.items() if k != 'api_key'}
        }