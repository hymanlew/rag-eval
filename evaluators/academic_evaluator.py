# 增强学术评估器 - 合并学术和混合模型优势

from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from .base import BaseEvaluator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.embedding_adapter import EmbeddingAdapterFactory, detect_embedding_config
import json
import re
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)

class AcademicEvaluator(BaseEvaluator):
    """增强学术评估器 - 支持可选的嵌入模型辅助评估"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化增强学术评估器"""
        super().__init__("Academic", config)
        
        try:
            # 初始化聊天模型（主要评估模型）
            self.chat_llm = ChatOpenAI(
                api_key=config.get("chat_api_key", config.get("api_key")),
                base_url=config.get("chat_base_url", config.get("base_url")),
                model=config.get("chat_model", config.get("model", "gpt-3.5-turbo")),
                temperature=0
            )
            
            # 初始化通用嵌入适配器
            embedding_config = {
                "api_key": config.get("embedding_api_key", ""),
                "base_url": config.get("embedding_base_url"),
                "model": config.get("embedding_model", "nomic-embed-text:latest"),
                "timeout": config.get("embedding_timeout", 30)
            }
            
            # 创建通用嵌入适配器
            try:
                self.embedding_adapter = EmbeddingAdapterFactory.create_adapter(embedding_config)
                logger.info(f"✅ 通用嵌入适配器初始化成功: {embedding_config['model']}")
            except Exception as e:
                logger.warning(f"⚠️  嵌入适配器初始化失败，将使用文本相似度: {e}")
                self.embedding_adapter = None
            
            # 评估模式：pure_chat（纯聊天模型）或 hybrid（混合模式）
            self.evaluation_mode = config.get("evaluation_mode", "pure_chat")
            
            self._available = True
            logger.info(f"✅ {self.name}增强评估器初始化成功 (模式: {self.evaluation_mode})")
        except Exception as e:
            logger.error(f"❌ {self.name}增强评估器初始化失败: {e}")
            self._available = False
    
    async def evaluate_answers_async(self, questions: List[str], answers: List[str], 
                                   ground_truths: List[str], contexts: List[List[str]] = None) -> Dict[str, List[float]]:
        """异步评估多个回答"""
        if not self._available:
            return {"relevancy": [None] * len(answers), 
                   "correctness": [None] * len(answers),
                   "completeness": [None] * len(answers),
                   "clarity": [None] * len(answers)}
        
        # 并发评估所有回答
        tasks = []
        for question, answer, ground_truth in zip(questions, answers, ground_truths):
            if answer and answer.strip():
                task = self.evaluate_single_answer_async(question, answer, ground_truth)
                tasks.append(task)
            else:
                # 为空回答创建默认结果
                tasks.append(asyncio.create_task(self._get_default_result()))
        
        # 等待所有评估完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        relevancy_scores = []
        correctness_scores = []
        completeness_scores = []
        clarity_scores = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"评估异常: {result}")
                relevancy_scores.append(0.0)
                correctness_scores.append(0.0)
                completeness_scores.append(0.0)
                clarity_scores.append(0.0)
            else:
                relevancy_scores.append(result["relevancy"])
                correctness_scores.append(result["correctness"])
                completeness_scores.append(result["completeness"])
                clarity_scores.append(result["clarity"])
        
        return {
            "relevancy": relevancy_scores,
            "correctness": correctness_scores,
            "completeness": completeness_scores,
            "clarity": clarity_scores
        }
    
    async def evaluate_single_answer_async(self, question: str, answer: str, 
                                         ground_truth: str, context: List[str] = None) -> Dict[str, float]:
        """异步评估单个回答 - 支持多种评估模式和质量指标"""
        
        try:
            if self.evaluation_mode == "hybrid" and self.embedding_adapter:
                # 混合模式：使用嵌入模型计算相关性，聊天模型计算质量指标
                return await self._evaluate_hybrid_mode(question, answer, ground_truth, context)
            else:
                # 纯聊天模式：使用聊天模型评估所有指标
                return await self._evaluate_pure_chat_mode(question, answer, ground_truth, context)
                        
        except Exception as e:
            logger.error(f"异步评估错误: {e}")
            return self._get_enhanced_default_scores()
    
    async def _evaluate_hybrid_mode(self, question: str, answer: str, ground_truth: str, context: List[str] = None) -> Dict[str, float]:
        """混合模式评估：嵌入模型 + 聊天模型"""
        
        try:
            # 并发执行两种评估
            relevancy_task = self._calculate_semantic_similarity(answer, ground_truth)
            quality_task = self._assess_enhanced_quality_with_chat_model(question, answer, ground_truth, context)
            
            # 等待两种评估完成
            relevancy_score, quality_scores = await asyncio.gather(relevancy_task, quality_task, return_exceptions=True)
            
            # 处理异常情况
            if isinstance(relevancy_score, Exception):
                logger.error(f"语义相似度计算失败: {relevancy_score}")
                relevancy_score = 0.0
            
            if isinstance(quality_scores, Exception):
                logger.error(f"质量评估失败: {quality_scores}")
                quality_scores = self._get_default_quality_scores()
            
            # 合并结果
            return {
                "relevancy": relevancy_score,
                **quality_scores
            }
            
        except Exception as e:
            logger.error(f"混合评估错误: {e}")
            return self._get_enhanced_default_scores()
    
    async def _evaluate_pure_chat_mode(self, question: str, answer: str, ground_truth: str, context: List[str] = None) -> Dict[str, float]:
        """纯聊天模式评估：使用聊天模型评估所有指标"""
        
        enhanced_prompt = f"""
请对以下回答进行全面的质量评估，重点关注多个维度：

**问题**: {question}
**回答**: {answer}
**标准答案**: {ground_truth}
**上下文**: {context if context else "无特定上下文"}

请从以下6个维度评估，每个维度给出0.0到1.0的分数：

1. **relevancy** (相关性): 回答与问题的相关程度，是否直接回答了问题
2. **correctness** (正确性): 回答的事实准确性，信息是否正确无误
3. **completeness** (完整性): 回答是否全面，是否涵盖了重要的方面
4. **clarity** (清晰度): 回答的表达是否清晰易懂，逻辑是否连贯
5. **coherence** (连贯性): 回答的结构是否合理，思路是否流畅
6. **helpfulness** (有用性): 回答对用户的实际帮助程度

请以JSON格式返回评估结果：
{{
  "relevancy": 分数,
  "correctness": 分数, 
  "completeness": 分数,
  "clarity": 分数,
  "coherence": 分数,
  "helpfulness": 分数
}}

评估标准：
- 0.9-1.0: 优秀 (Excellent)
- 0.8-0.9: 良好 (Good) 
- 0.6-0.8: 一般 (Fair)
- 0.4-0.6: 较差 (Poor)
- 0.0-0.4: 很差 (Very Poor)
"""
        
        try:
            # 使用异步HTTP请求
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = {
                    "Authorization": f"Bearer {self.config.get('chat_api_key', self.config.get('api_key'))}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.config.get("chat_model", self.config.get("model", "gpt-3.5-turbo")),
                    "messages": [
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    "temperature": 0
                }
                
                logger.debug(f"🔍 增强异步评估请求发送中...")
                async with session.post(
                    f"{self.config.get('chat_base_url', self.config.get('base_url')).rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        result_text = result["choices"][0]["message"]["content"].strip()
                        logger.debug(f"🔍 增强异步评估响应接收: {result_text[:100]}...")
                        
                        # 解析增强评分
                        return self._parse_enhanced_scores(result_text)
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ API请求失败: {response.status} - {error_text}")
                        return self._get_enhanced_default_scores()
                        
        except Exception as e:
            logger.error(f"纯聊天模式评估错误: {e}")
            return self._get_enhanced_default_scores()
    
    async def _calculate_semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """使用嵌入模型计算语义相似度（混合模式用）- 使用通用适配器"""
        
        try:
            # 如果没有嵌入适配器，直接使用文本相似度
            if not self.embedding_adapter:
                logger.debug("🔍 嵌入适配器不可用，使用文本相似度")
                return self._calculate_text_similarity(answer, ground_truth)
            
            # 并发获取两个文本的嵌入向量
            answer_task = self.embedding_adapter.embed_query(answer)
            ground_truth_task = self.embedding_adapter.embed_query(ground_truth)
            
            answer_embedding, ground_truth_embedding = await asyncio.gather(
                answer_task, ground_truth_task, return_exceptions=True
            )
            
            # 处理异常情况
            if isinstance(answer_embedding, Exception):
                logger.error(f"❌ 回答嵌入向量获取失败: {answer_embedding}")
                return self._calculate_text_similarity(answer, ground_truth)
            
            if isinstance(ground_truth_embedding, Exception):
                logger.error(f"❌ 标准答案嵌入向量获取失败: {ground_truth_embedding}")
                return self._calculate_text_similarity(answer, ground_truth)
            
            # 计算余弦相似度
            if len(answer_embedding) > 0 and len(ground_truth_embedding) > 0:
                similarity = self._calculate_cosine_similarity(answer_embedding, ground_truth_embedding)
                logger.debug(f"🔍 嵌入向量语义相似度: {similarity:.4f}")
                return similarity
            else:
                logger.error(f"❌ 嵌入向量为空 - answer: {len(answer_embedding)}, ground_truth: {len(ground_truth_embedding)}")
                return self._calculate_text_similarity(answer, ground_truth)
                        
        except Exception as e:
            logger.error(f"嵌入模型调用失败: {e}")
            return self._calculate_text_similarity(answer, ground_truth)
    
    async def _assess_enhanced_quality_with_chat_model(self, question: str, answer: str, ground_truth: str, context: List[str] = None) -> Dict[str, float]:
        """使用聊天模型进行增强质量评估（混合模式用）"""
        
        quality_prompt = f"""
请评估以下回答的质量，重点关注正确性、完整性、清晰度、连贯性和有用性：

问题: {question}
回答: {answer}
标准答案: {ground_truth}
上下文: {context if context else "无特定上下文"}

请从以下5个维度评估，每个维度给出0.0到1.0的分数：

1. **correctness**: 事实准确性，与标准答案的一致程度
2. **completeness**: 是否涵盖了标准答案中的关键信息
3. **clarity**: 表达是否清晰、逻辑是否连贯
4. **coherence**: 结构是否合理，思路是否流畅
5. **helpfulness**: 对用户的实际帮助程度

请以JSON格式返回评估结果：
{{
  "correctness": 分数,
  "completeness": 分数,
  "clarity": 分数,
  "coherence": 分数,
  "helpfulness": 分数
}}
"""
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
                headers = {
                    "Authorization": f"Bearer {self.config.get('chat_api_key', self.config.get('api_key'))}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.config.get("chat_model", self.config.get("model", "gpt-3.5-turbo")),
                    "messages": [
                        {"role": "user", "content": quality_prompt}
                    ],
                    "temperature": 0
                }
                
                async with session.post(
                    f"{self.config.get('chat_base_url', self.config.get('base_url')).rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        result_text = result["choices"][0]["message"]["content"].strip()
                        logger.debug(f"🔍 质量评估响应: {result_text[:100]}...")
                        
                        return self._parse_quality_scores(result_text)
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 聊天模型请求失败: {response.status} - {error_text}")
                        return self._get_default_quality_scores()
                        
        except Exception as e:
            logger.error(f"聊天模型质量评估失败: {e}")
            return self._get_default_quality_scores()
    
    def _parse_scores(self, result_text: str) -> Dict[str, float]:
        """解析基本评分结果（向后兼容）"""
        return self._parse_enhanced_scores(result_text)
    
    def _parse_enhanced_scores(self, result_text: str) -> Dict[str, float]:
        """解析增强评分结果"""
        scores = {
            "relevancy": 0.0, "correctness": 0.0, "completeness": 0.0, 
            "clarity": 0.0, "coherence": 0.0, "helpfulness": 0.0
        }
        
        # 方法1: JSON形式を探す
        json_match = re.search(r'\{[^}]*"relevancy"[^}]*\}', result_text, re.DOTALL)
        if json_match:
            try:
                scores_dict = json.loads(json_match.group())
                for metric in ["relevancy", "correctness", "completeness", "clarity", "coherence", "helpfulness"]:
                    score = scores_dict.get(metric, 0.0)
                    scores[metric] = max(0.0, min(1.0, float(score)))
                return scores
            except:
                pass
        
        # 方法2: テキストからスコアを抽出
        patterns = {
            "relevancy": r"relevancy[^0-9]*([0-9.]+)",
            "correctness": r"correctness[^0-9]*([0-9.]+)", 
            "completeness": r"completeness[^0-9]*([0-9.]+)",
            "clarity": r"clarity[^0-9]*([0-9.]+)",
            "coherence": r"coherence[^0-9]*([0-9.]+)",
            "helpfulness": r"helpfulness[^0-9]*([0-9.]+)"
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, result_text, re.IGNORECASE)
            if match:
                try:
                    scores[metric] = max(0.0, min(1.0, float(match.group(1))))
                except:
                    scores[metric] = 0.5
        
        logger.debug(f"解析的增强评分: {scores}")
        return scores
    
    def _parse_quality_scores(self, result_text: str) -> Dict[str, float]:
        """解析质量评分结果（混合模式用）"""
        scores = {"correctness": 0.0, "completeness": 0.0, "clarity": 0.0, "coherence": 0.0, "helpfulness": 0.0}
        
        # 方法1: JSON形式を探す
        json_match = re.search(r'\{[^}]*"correctness"[^}]*\}', result_text, re.DOTALL)
        if json_match:
            try:
                scores_dict = json.loads(json_match.group())
                for metric in ["correctness", "completeness", "clarity", "coherence", "helpfulness"]:
                    score = scores_dict.get(metric, 0.0)
                    scores[metric] = max(0.0, min(1.0, float(score)))
                return scores
            except:
                pass
        
        # 方法2: テキストからスコアを抽出
        patterns = {
            "correctness": r"correctness[^0-9]*([0-9.]+)",
            "completeness": r"completeness[^0-9]*([0-9.]+)",
            "clarity": r"clarity[^0-9]*([0-9.]+)",
            "coherence": r"coherence[^0-9]*([0-9.]+)",
            "helpfulness": r"helpfulness[^0-9]*([0-9.]+)"
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, result_text, re.IGNORECASE)
            if match:
                try:
                    scores[metric] = max(0.0, min(1.0, float(match.group(1))))
                except:
                    scores[metric] = 0.5
        
        logger.debug(f"解析的质量评分: {scores}")
        return scores
    
    async def _get_default_result(self) -> Dict[str, float]:
        """获取默认结果（向后兼容）"""
        return self._get_enhanced_default_scores()
    
    def _get_enhanced_default_scores(self) -> Dict[str, float]:
        """获取增强默认评分"""
        return {
            "relevancy": 0.0, "correctness": 0.0, "completeness": 0.0, 
            "clarity": 0.0, "coherence": 0.0, "helpfulness": 0.0
        }
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            import math
            
            if len(vec1) != len(vec2):
                logger.error(f"❌ 向量维度不匹配: {len(vec1)} vs {len(vec2)}")
                return 0.0
            
            # 计算点积
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # 计算向量长度
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            # 计算余弦相似度
            cosine_similarity = dot_product / (magnitude1 * magnitude2)
            
            # 确保结果在[0, 1]范围内
            return max(0.0, min(1.0, cosine_similarity))
            
        except Exception as e:
            logger.error(f"余弦相似度计算失败: {e}")
            return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（备选方法，当embedding不可用时使用）"""
        try:
            # 简单的词汇重叠相似度计算
            import re
            from collections import Counter
            
            # 分词
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            similarity = len(intersection) / len(union) if union else 0.0
            return similarity
            
        except Exception as e:
            logger.error(f"文本相似度计算失败: {e}")
            return 0.0
    
    def _get_default_quality_scores(self) -> Dict[str, float]:
        """获取默认质量评分"""
        return {"correctness": 0.0, "completeness": 0.0, "clarity": 0.0, "coherence": 0.0, "helpfulness": 0.0}
    
    def get_supported_metrics(self) -> List[str]:
        """获取支持的评价指标"""
        # 根据评估模式返回不同的指标
        if self.evaluation_mode == "hybrid":
            return ["relevancy", "correctness", "completeness", "clarity", "coherence", "helpfulness"]
        else:
            return ["relevancy", "correctness", "completeness", "clarity", "coherence", "helpfulness"]
    
    # 为了兼容性，保留同步方法
    def evaluate_answers(self, questions: List[str], answers: List[str], 
                        ground_truths: List[str], contexts: List[List[str]] = None) -> Dict[str, List[float]]:
        """同步评估方法（向后兼容）"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.evaluate_answers_async(questions, answers, ground_truths, contexts)
        )
    
    def evaluate_single_answer(self, question: str, answer: str, ground_truth: str) -> Dict[str, float]:
        """同步单个评估方法（向后兼容）"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.evaluate_single_answer_async(question, answer, ground_truth)
        )