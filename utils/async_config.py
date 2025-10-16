# 异步配置管理工具

import logging
from typing import Dict, Any
from config import ASYNC_CONFIG

class AsyncConfigManager:
    """异步配置管理器"""
    
    def __init__(self):
        self.config = ASYNC_CONFIG
        self._setup_logging()
    
    def _setup_logging(self):
        """设置异步相关日志配置"""
        logging.basicConfig(
            level=self.config.get("log_level", "INFO"),
            format=self.config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    
    def get_timeout_config(self) -> Dict[str, int]:
        """获取超时配置"""
        return {
            "rag_query_timeout": self.config["rag_query_timeout"],
            "evaluator_timeout": self.config["evaluator_timeout"],
            "http_timeout": self.config["http_timeout"]
        }
    
    def get_concurrency_config(self) -> Dict[str, int]:
        """获取并发配置"""
        return {
            "max_concurrency": self.config["max_concurrency"],
            "batch_size": self.config["batch_size"]
        }
    
    def get_retry_config(self) -> Dict[str, Any]:
        """获取重试配置"""
        return {
            "retry_attempts": self.config["retry_attempts"],
            "retry_delay": self.config["retry_delay"]
        }
    
    def get_http_config(self) -> Dict[str, int]:
        """获取HTTP客户端配置"""
        return {
            "timeout": self.config["http_timeout"],
            "pool_size": self.config["http_pool_size"],
            "max_connections": self.config["http_max_connections"]
        }
    
    def get_progress_config(self) -> Dict[str, Any]:
        """获取进度跟踪配置"""
        return {
            "update_interval": self.config["progress_update_interval"],
            "enabled": self.config["progress_enabled"]
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config.copy()
    
    def update_config(self, key: str, value: Any):
        """更新配置"""
        if key in self.config:
            self.config[key] = value
        else:
            raise KeyError(f"Unknown async config key: {key}")
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 检查超时配置
            if self.config["rag_query_timeout"] <= 0:
                raise ValueError("RAG query timeout must be positive")
            
            if self.config["evaluator_timeout"] <= 0:
                raise ValueError("Evaluator timeout must be positive")
            
            # 检查并发配置
            if self.config["max_concurrency"] <= 0:
                raise ValueError("Max concurrency must be positive")
            
            if self.config["batch_size"] <= 0:
                raise ValueError("Batch size must be positive")
            
            # 检查重试配置
            if self.config["retry_attempts"] < 0:
                raise ValueError("Retry attempts must be non-negative")
            
            if self.config["retry_delay"] < 0:
                raise ValueError("Retry delay must be non-negative")
            
            # 检查HTTP配置
            if self.config["http_timeout"] <= 0:
                raise ValueError("HTTP timeout must be positive")
            
            if self.config["http_pool_size"] <= 0:
                raise ValueError("HTTP pool size must be positive")
            
            if self.config["http_max_connections"] <= 0:
                raise ValueError("HTTP max connections must be positive")
            
            # 检查进度配置
            if self.config["progress_update_interval"] <= 0:
                raise ValueError("Progress update interval must be positive")
            
            return True
            
        except (KeyError, ValueError) as e:
            logging.error(f"Async config validation failed: {e}")
            return False
    
    def get_config_summary(self) -> str:
        """获取配置摘要"""
        summary = f"""
=== 异步配置摘要 ===
超时配置:
  RAG查询超时: {self.config['rag_query_timeout']}秒
  评价器超时: {self.config['evaluator_timeout']}秒
  HTTP超时: {self.config['http_timeout']}秒

并发配置:
  最大并发数: {self.config['max_concurrency']}
  批处理大小: {self.config['batch_size']}

重试配置:
  重试次数: {self.config['retry_attempts']}
  重试延迟: {self.config['retry_delay']}秒

HTTP配置:
  连接池大小: {self.config['http_pool_size']}
  最大连接数: {self.config['http_max_connections']}

进度跟踪:
  启用状态: {'启用' if self.config['progress_enabled'] else '禁用'}
  更新间隔: {self.config['progress_update_interval']}秒

日志级别: {self.config['log_level']}
"""
        return summary

# 全局异步配置管理器实例
async_config = AsyncConfigManager()

def get_async_config() -> AsyncConfigManager:
    """获取异步配置管理器实例"""
    return async_config