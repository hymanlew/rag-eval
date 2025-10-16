#!/usr/bin/env python3
# 异步RAG评估系统

import argparse
import asyncio
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from config import CHAT_CONFIG, EMBEDDING_CONFIG, ASYNC_CONFIG, get_enabled_rag_systems, validate_config
from connectors.universal import UniversalRAGConnector
from evaluators.factory import EvaluatorManager

class AsyncMultiEvaluatorRAGSystem:
    """异步多评估器RAG评估系统"""
    
    def __init__(self):
        """系统初始化"""
        # 验证配置
        config_errors = validate_config()
        if config_errors:
            raise ValueError(f"配置错误: {config_errors}")
        
        # 初始化RAG连接器
        self.connectors = {}
        enabled_systems = get_enabled_rag_systems()
        
        for system_name, config in enabled_systems.items():
            try:
                connector = UniversalRAGConnector(system_name, config)
                self.connectors[system_name] = connector
                print(f"✅ {system_name} RAG系统连接器创建成功")
            except Exception as e:
                print(f"❌ {system_name} RAG系统初始化错误: {e}")
        
        if not self.connectors:
            raise ValueError("没有可用的RAG系统")
        
        # 初始化异步评估器管理器
        self.async_evaluator_manager = EvaluatorManager(CHAT_CONFIG, EMBEDDING_CONFIG)
    
    async def load_test_cases(self, file_path: str) -> list:
        """加载测试用例"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"测试用例加载失败 {file_path}: {e}")
    
    async def test_connections(self) -> Dict[str, bool]:
        """测试所有连接"""
        results = {}
        
        for system_name, connector in self.connectors.items():
            try:
                is_connected = await connector.test_connection_async()
                results[system_name] = is_connected
                print(f"{'✅' if is_connected else '❌'} {system_name} RAG系统连接{'成功' if is_connected else '失败'}")
            except Exception as e:
                results[system_name] = False
                print(f"❌ {system_name} RAG系统连接测试失败: {e}")
        
        return results
    
    async def query_rag_systems(self, question: str) -> Dict[str, Dict[str, Any]]:
        """查询所有RAG系统"""
        results = {}
        
        for system_name, connector in self.connectors.items():
            try:
                result = await connector.query_with_timeout(
                    question, 
                    timeout=ASYNC_CONFIG["rag_query_timeout"]
                )
                results[system_name] = result
                
                if result.get("error"):
                    print(f"  {system_name} 错误: {result['error']}")
                else:
                    print(f"  {system_name} 成功获取回答")
                    
            except Exception as e:
                results[system_name] = {"answer": "", "contexts": [], "error": str(e)}
                print(f"  {system_name} 查询失败: {e}")
        
        return results
    
    async def run_evaluation(self, test_cases: list, connection_results: Dict[str, bool]) -> Dict[str, Any]:
        """运行评估"""
        evaluation_results = {}
        
        # 只评估连接成功的系统
        successful_systems = [name for name, success in connection_results.items() if success]
        
        if not successful_systems:
            print("❌ 没有可用的RAG系统进行评估")
            return evaluation_results
        
        print(f"\n🔍 开始评估，测试用例数量: {len(test_cases)}")
        
        # 准备评估数据
        all_questions = []
        all_answers = {}
        all_ground_truths = []
        all_contexts = {}
        
        # 初始化数据结构
        for system_name in successful_systems:
            all_answers[system_name] = []
            all_contexts[system_name] = []
        
        # 收集所有问题和标准答案
        for test_case in test_cases:
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            
            all_questions.append(question)
            all_ground_truths.append(ground_truth)
        
        # 查询所有RAG系统
        print("\n📡 查询RAG系统...")
        for i, question in enumerate(all_questions):
            print(f"\n问题 {i+1}/{len(all_questions)}: {question[:50]}...")
            
            rag_results = await self.query_rag_systems(question)
            
            for system_name in successful_systems:
                result = rag_results.get(system_name, {})
                answer = result.get("answer", "")
                contexts = result.get("contexts", [])
                
                all_answers[system_name].append(answer)
                all_contexts[system_name].append(contexts)
                
                # 在测试用例中添加RAG回答
                if i < len(test_cases):
                    test_cases[i][f"{system_name}_answer"] = answer
                
                print(f"  {system_name} 回答长度: {len(answer)} 字符")
        
        # 对每个系统进行评估
        for system_name in successful_systems:
            print(f"\n📊 评估 {system_name} 系统...")
            
            try:
                metrics = await self.async_evaluator_manager.evaluate_all_async(
                    all_questions,
                    all_answers[system_name],
                    all_ground_truths,
                    all_contexts[system_name]
                )
                evaluation_results[system_name] = metrics
                print(f"  ✅ {system_name} 异步评估完成")
            except Exception as e:
                print(f"  ❌ {system_name} 异步评估失败: {e}")
                evaluation_results[system_name] = {}
        
        return evaluation_results
    
    async def save_results(self, evaluation_results: Dict[str, Any], test_cases: list, output_dir: str):
        """保存评估结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存详细的JSON结果
        detailed_results = {
            "test_cases": test_cases,
            "evaluation_results": evaluation_results,
            "summary": {}
        }
        
        # 计算摘要统计
        for system_name, results in evaluation_results.items():
            detailed_results["summary"][system_name] = {}
            for evaluator_name, metrics in results.items():
                detailed_results["summary"][system_name][evaluator_name] = {}
                for metric_name, values in metrics.items():
                    if values and any(v is not None for v in values):
                        valid_values = [v for v in values if v is not None]
                        detailed_results["summary"][system_name][evaluator_name][metric_name] = {
                            "mean": sum(valid_values) / len(valid_values),
                            "min": min(valid_values),
                            "max": max(valid_values),
                            "count": len(valid_values)
                        }
        
        # 保存JSON文件
        json_file = output_path / "detailed_evaluation_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 生成CSV格式结果
        csv_data = []
        
        for i, test_case in enumerate(test_cases):
            row = {
                "question": test_case["question"],
                "ground_truth": test_case["ground_truth"]
            }
            
            for system_name, system_results in evaluation_results.items():
                # 添加RAG回答
                if i < len(test_cases):  # 安全检查
                    row[f"{system_name}_answer"] = test_case.get(f"{system_name}_answer", "")
                
                # 添加评估指标
                for evaluator_name, metrics in system_results.items():
                    for metric_name, values in metrics.items():
                        if i < len(values) and values[i] is not None:
                            row[f"{system_name}_{evaluator_name}_{metric_name}"] = values[i]
                        else:
                            row[f"{system_name}_{evaluator_name}_{metric_name}"] = None
            
            csv_data.append(row)
        
        # 保存CSV文件
        csv_file = output_path / "multi_evaluation_results.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"\n✅ 结果已保存:")
        print(f"  详细结果: {json_file}")
        print(f"  CSV结果: {csv_file}")
    
    async def run(self, test_cases_file: str, output_dir: str):
        """运行完整的评估流程"""
        print("🚀 启动异步多评估器RAG评估系统...")
        
        # 加载测试用例
        test_cases = await self.load_test_cases(test_cases_file)
        print(f"📋 加载了 {len(test_cases)} 个测试用例")
        
        # 测试连接
        connection_results = await self.test_connections()
        
        # 初始化异步评估器
        await self.async_evaluator_manager.initialize_async()
        
        # 运行评估
        evaluation_results = await self.run_evaluation(test_cases, connection_results)
        
        # 保存结果
        await self.save_results(evaluation_results, test_cases, output_dir)
        
        print(f"\n🎉 异步多评估器RAG评估完成！")
        print(f"📊 结果目录: {output_dir}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="异步多评估器RAG评估系统")
    parser.add_argument("--test-cases", default="data/test_cases_jp.json", 
                       help="测试用例文件路径 (默认: data/test_cases_jp.json)")
    parser.add_argument("--output", default="results", 
                       help="结果输出目录 (默认: results)")
    
    args = parser.parse_args()
    
    try:
        evaluator = AsyncMultiEvaluatorRAGSystem()
        await evaluator.run(args.test_cases, args.output)
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())