# Ragas评估器 - 使用Ragas原生异步API的评估器

from typing import Dict, List, Any, Optional
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, 
    answer_correctness, 
    faithfulness, 
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from .base import BaseEvaluator
import asyncio
import aiohttp
import math
import logging

logger = logging.getLogger(__name__)


class RagasEvaluator(BaseEvaluator):
    """Ragas评估器 - 使用Ragas原生异步API支持完整的RAG评估指标"""
    
    def __init__(self, config: Dict[str, Any]):
        """Ragas评估器初始化"""
        super().__init__("Ragas", config)
        
        try:
            # Chat LLM初始化 (支持任何OpenAI兼容的API)
            self.eval_llm = LangchainLLMWrapper(ChatOpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config.get("model", "gpt-3.5-turbo"),
                temperature=0,
                max_tokens=1000,
                timeout=60  # 增加超时时间
            ))
            
            # Embeddings初始化 (支持Ollama或其他嵌入模型)
            embedding_config = config.get("embedding", {})
            embedding_type = embedding_config.get("type", "ollama")
            
            if embedding_type == "ollama":
                ollama_base_url = embedding_config.get("base_url", "http://localhost:11434")
                ollama_model = embedding_config.get("model", "nomic-embed-text:latest")
                
                self.embeddings = OllamaEmbeddings(
                    base_url=ollama_base_url,
                    model=ollama_model
                )
                embedding_name = f"{ollama_model} (Ollama)"
            else:
                # 支持其他嵌入模型
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(
                    api_key=config.get("embedding_api_key", config["api_key"]),
                    model=embedding_config.get("model", "text-embedding-ada-002")
                )
                embedding_name = f"{embedding_config.get('model', 'text-embedding-ada-002')} (OpenAI)"
            
            # 初始化Ragas指标
            self.metrics = [
                answer_relevancy, 
                answer_correctness, 
                faithfulness, 
                context_precision,
                context_recall
            ]
            
            # Embeddings测试
            test_result = self.embeddings.embed_query("test")
            if len(test_result) > 0:
                logger.info(f"✅ Embeddings初始化成功: {embedding_name}")
            else:
                raise ValueError("Embeddings test failed")
            
            self._available = True
            logger.info(f"✅ {self.name}评估器初始化成功")
            logger.info(f"   Chat: {config.get('model', 'gpt-3.5-turbo')}")
            logger.info(f"   Embeddings: {embedding_name}")
            
        except Exception as e:
            logger.error(f"❌ {self.name}评估器初始化失败: {e}")
            self._available = False
    
    async def evaluate_single_answer_async(self, question: str, answer: str, ground_truth: str, context: List[str] = None) -> Dict[str, float]:
        """使用Ragas原生异步API评估单个答案"""
        if not self._available:
            return {"relevancy": None, "correctness": None, "faithfulness": None, "context_precision": None, "context_recall": None}
        
        if not answer or not answer.strip():
            return {"relevancy": None, "correctness": None, "faithfulness": None, "context_precision": None, "context_recall": None}
        
        try:
            # 使用Ragas原生异步API
            return await self._evaluate_ragas_native_async(question, answer, ground_truth, context)
            
        except Exception as e:
            logger.error(f"❌ {self.name}异步评估失败: {e}")
            return {"relevancy": None, "correctness": None, "faithfulness": None, "context_precision": None, "context_recall": None}
    
    async def _evaluate_ragas_native_async(self, question: str, answer: str, ground_truth: str, context: List[str] = None) -> Dict[str, float]:
        """使用Ragas原生异步API进行评估"""
        try:
            # 准备评估数据 - Ragas需要retrieved_contexts字段
            retrieved_contexts = context if context else ['No relevant context provided']
            
            # 创建数据集
            from datasets import Dataset
            dataset = Dataset.from_dict({
                'question': [question],
                'answer': [answer],
                'ground_truth': [ground_truth],
                'retrieved_contexts': [retrieved_contexts]
            })
            
            # 使用Ragas评估（同步函数）- 在单独的线程中运行以避免阻塞事件循环
            result = await asyncio.to_thread(
                evaluate,
                dataset=dataset,
                metrics=self.metrics,
                llm=self.eval_llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )
            
            # 处理结果 - Ragas 0.3.2+ 返回EvaluationResult对象
            scores = {}
            try:
                scores_dict = result.scores
                
                # scores_dict 是列表格式，每个元素是一个字典
                if scores_dict and len(scores_dict) > 0:
                    item_scores = scores_dict[0]  # 第一个评估结果
                    
                    if 'answer_relevancy' in item_scores:
                        rel_score = item_scores['answer_relevancy']
                        scores['relevancy'] = float(rel_score) if rel_score is not None and not math.isnan(rel_score) else None
                    
                    if 'answer_correctness' in item_scores:
                        cor_score = item_scores['answer_correctness']
                        scores['correctness'] = float(cor_score) if cor_score is not None and not math.isnan(cor_score) else None
                    
                    if 'faithfulness' in item_scores:
                        faith_score = item_scores['faithfulness']
                        scores['faithfulness'] = float(faith_score) if faith_score is not None and not math.isnan(faith_score) else None
                    
                    if 'context_precision' in item_scores:
                        ctx_prec_score = item_scores['context_precision']
                        scores['context_precision'] = float(ctx_prec_score) if ctx_prec_score is not None and not math.isnan(ctx_prec_score) else None
                    
                    if 'context_recall' in item_scores:
                        ctx_rec_score = item_scores['context_recall']
                        scores['context_recall'] = float(ctx_rec_score) if ctx_rec_score is not None and not math.isnan(ctx_rec_score) else None
                
                logger.debug(f"    Ragas原生异步评估完成: {scores}")
                    
            except Exception as e:
                logger.error(f"    Ragas分数处理错误: {e}")
                scores = {"relevancy": None, "correctness": None, "faithfulness": None, "context_precision": None, "context_recall": None}
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ {self.name}原生异步评估失败: {e}")
            return {"relevancy": None, "correctness": None, "faithfulness": None, "context_precision": None, "context_recall": None}
    
    async def evaluate_answers_async(self, questions: List[str], answers: List[str], 
                                  ground_truths: List[str], contexts: List[List[str]] = None) -> Dict[str, List[float]]:
        """使用Ragas原生异步API批量评估多个答案"""
        if not self._available:
            return {"relevancy": [None] * len(answers), "correctness": [None] * len(answers), "faithfulness": [None] * len(answers), "context_precision": [None] * len(answers), "context_recall": [None] * len(answers)}
        
        try:
            # 准备评估数据 - Ragas需要retrieved_contexts字段
            retrieved_contexts = contexts if contexts else [['No relevant context provided'] for _ in range(len(questions))]
            
            # 创建数据集
            from datasets import Dataset
            dataset = Dataset.from_dict({
                'question': questions,
                'answer': answers,
                'ground_truth': ground_truths,
                'retrieved_contexts': retrieved_contexts
            })
            
            # 使用Ragas批量评估（同步函数）
            result = evaluate(
                dataset, 
                metrics=self.metrics,
                llm=self.eval_llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )
            
            # 处理结果
            relevancy_scores = []
            correctness_scores = []
            faithfulness_scores = []
            context_precision_scores = []
            context_recall_scores = []
            
            try:
                scores_dict = result.scores
                
                # scores_dict 是列表格式，每个元素是一个字典
                if scores_dict and len(scores_dict) > 0:
                    for i, item_scores in enumerate(scores_dict):
                        if i < len(answers):
                            # Answer Relevancy
                            if 'answer_relevancy' in item_scores:
                                rel_score = item_scores['answer_relevancy']
                                relevancy_scores.append(float(rel_score) if rel_score is not None and not math.isnan(rel_score) else None)
                            else:
                                relevancy_scores.append(None)
                            
                            # Answer Correctness
                            if 'answer_correctness' in item_scores:
                                cor_score = item_scores['answer_correctness']
                                correctness_scores.append(float(cor_score) if cor_score is not None and not math.isnan(cor_score) else None)
                            else:
                                correctness_scores.append(None)
                            
                            # Faithfulness
                            if 'faithfulness' in item_scores:
                                faith_score = item_scores['faithfulness']
                                faithfulness_scores.append(float(faith_score) if faith_score is not None and not math.isnan(faith_score) else None)
                            else:
                                faithfulness_scores.append(None)
                            
                            # Context Precision
                            if 'context_precision' in item_scores:
                                ctx_prec_score = item_scores['context_precision']
                                context_precision_scores.append(float(ctx_prec_score) if ctx_prec_score is not None and not math.isnan(ctx_prec_score) else None)
                            else:
                                context_precision_scores.append(None)
                            
                            # Context Recall
                            if 'context_recall' in item_scores:
                                ctx_rec_score = item_scores['context_recall']
                                context_recall_scores.append(float(ctx_rec_score) if ctx_rec_score is not None and not math.isnan(ctx_rec_score) else None)
                            else:
                                context_recall_scores.append(None)
                        else:
                            # 如果没有足够的评估结果，填充None
                            relevancy_scores.append(None)
                            correctness_scores.append(None)
                            faithfulness_scores.append(None)
                            context_precision_scores.append(None)
                            context_recall_scores.append(None)
                
                logger.debug(f"    Ragas原生异步批量评估完成，处理了 {len(relevancy_scores)} 个样本")
                    
            except Exception as e:
                logger.error(f"    Ragas批量分数处理错误: {e}")
                # 返回默认值
                relevancy_scores = [None] * len(answers)
                correctness_scores = [None] * len(answers)
                faithfulness_scores = [None] * len(answers)
                context_precision_scores = [None] * len(answers)
                context_recall_scores = [None] * len(answers)
            
            return {
                "relevancy": relevancy_scores,
                "correctness": correctness_scores,
                "faithfulness": faithfulness_scores,
                "context_precision": context_precision_scores,
                "context_recall": context_recall_scores
            }
            
        except Exception as e:
            logger.error(f"❌ {self.name}异步批量评估失败: {e}")
            return {"relevancy": [None] * len(answers), "correctness": [None] * len(answers), "faithfulness": [None] * len(answers), "context_precision": [None] * len(answers), "context_recall": [None] * len(answers)}
    
    def get_supported_metrics(self) -> List[str]:
        """获取支持的评估指标"""
        return ["relevancy", "correctness", "faithfulness", "context_precision", "context_recall"]
    
    def is_available(self) -> bool:
        """检查评估器是否可用"""
        return self._available
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        """获取评估器信息"""
        return {
            "name": self.name,
            "supported_metrics": self.get_supported_metrics(),
            "description": "异步Ragas框架 - 完整的RAG评估指标集",
            "async": True,
            "available": self.is_available()
        }