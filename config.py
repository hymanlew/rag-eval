# RAG Evaluation System - 多配置文件支持

import os
import glob
import warnings
from dotenv import load_dotenv
from pathlib import Path

# 过滤 Pydantic 命名空间冲突警告
warnings.filterwarnings("ignore", category=UserWarning, message="Field \"model_name\" has conflict with protected namespace")

def load_all_env_files():
    """加载所有配置文件"""
    # 加载默认.env文件
    if Path(".env").exists():
        load_dotenv(".env")

    # 加载所有.env.local.*文件
    config_files = glob.glob(".env.local.*")
    for config_file in sorted(config_files):
        print(f"📁 Loading config: {config_file}")
        load_dotenv(config_file, override=True)  # override=True 允许覆盖已有变量

    if config_files:
        print(f"✅ Loaded {len(config_files)} configuration files")
    else:
        print("ℹ️  No .env.local.* files found, using default .env")

# 加载所有配置文件
load_all_env_files()

# 聊天和嵌入模型配置
CHAT_CONFIG = {
    "api_key": os.getenv("CHAT_API_KEY"),
    "base_url": os.getenv("CHAT_BASE_URL", "https://openrouter.ai/api/v1"),
    "model": os.getenv("CHAT_MODEL", "gpt-3.5-turbo"),
    "timeout": int(os.getenv("CHAT_TIMEOUT", "45"))  # 评价器超时时间
}

EMBEDDING_CONFIG = {
    "base_url": os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434"),
    "model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
    "api_key": os.getenv("EMBEDDING_API_KEY", ""),  # 嵌入模型通常不需要API key，但保留选项
    "timeout": int(os.getenv("EMBEDDING_TIMEOUT", "30"))  # 嵌入模型超时时间
}

# 异步配置
ASYNC_CONFIG = {
    # 基础超时配置
    "rag_query_timeout": int(os.getenv("RAG_QUERY_TIMEOUT", "30")),  # RAG查询超时时间
    "evaluator_timeout": int(os.getenv("EVALUATOR_TIMEOUT", "45")),  # 评价器超时时间
    
    # 并发控制配置
    "max_concurrency": int(os.getenv("MAX_CONCURRENCY", "3")),  # 最大并发数
    "batch_size": int(os.getenv("BATCH_SIZE", "10")),  # 批处理大小
    
    # 重试机制配置
    "retry_attempts": int(os.getenv("RETRY_ATTEMPTS", "2")),  # 重试次数
    "retry_delay": float(os.getenv("RETRY_DELAY", "1.0")),  # 重试延迟
    
    # HTTP客户端配置
    "http_timeout": int(os.getenv("ASYNC_HTTP_TIMEOUT", "30")),  # HTTP超时时间
    "http_pool_size": int(os.getenv("ASYNC_HTTP_POOL_SIZE", "10")),  # 连接池大小
    "http_max_connections": int(os.getenv("ASYNC_HTTP_MAX_CONNECTIONS", "20")),  # 最大连接数
    
    # 进度跟踪配置
    "progress_update_interval": float(os.getenv("PROGRESS_UPDATE_INTERVAL", "1.0")),  # 进度更新间隔
    "progress_enabled": os.getenv("PROGRESS_ENABLED", "true").lower() == "true",  # 是否启用进度跟踪
    
    # 日志配置
    "log_level": os.getenv("ASYNC_LOG_LEVEL", "INFO"),  # 日志级别
    "log_format": os.getenv("ASYNC_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # 日志格式
}

# RAG系统配置 - 支持的RAG系统
RAG_SYSTEMS = {
    "ragflow": {
        "enabled": os.getenv("RAGFLOW_ENABLED", "false").lower() == "true",
        "api_key": os.getenv("RAGFLOW_API_KEY"),
        "base_url": os.getenv("RAGFLOW_BASE_URL"),
        "chat_id": os.getenv("RAGFLOW_CHAT_ID"),  # 可选，自动检测
    },
    "dify": {
        "enabled": os.getenv("DIFY_ENABLED", "true").lower() == "true",
        "api_key": os.getenv("DIFY_API_KEY"),
        "base_url": os.getenv("DIFY_BASE_URL"),
        "app_id": os.getenv("DIFY_APP_ID"),
        "user_id": os.getenv("DIFY_USER_ID", "rag-evaluator"),
    }
}

# 获取启用的RAG系统
def get_enabled_rag_systems():
    """返回所有启用的RAG系统配置"""
    return {name: config for name, config in RAG_SYSTEMS.items() if config.get("enabled", False)}

# 验证配置完整性
def validate_config():
    """验证必要的配置是否完整"""
    errors = []
    
    # 检查聊天模型配置
    if not CHAT_CONFIG.get("api_key"):
        errors.append("CHAT_API_KEY is required")
    
    # 检查嵌入模型配置
    if not EMBEDDING_CONFIG.get("base_url"):
        errors.append("EMBEDDING_BASE_URL is required")
    
    # 检查启用的RAG系统
    enabled_systems = get_enabled_rag_systems()
    if not enabled_systems:
        errors.append("At least one RAG system must be enabled")
    
    # 检查每个启用的RAG系统配置
    for name, config in enabled_systems.items():
        if not config.get("api_key"):
            errors.append(f"{name.upper()}_API_KEY is required")
        if not config.get("base_url"):
            errors.append(f"{name.upper()}_BASE_URL is required")
    
    return errors
