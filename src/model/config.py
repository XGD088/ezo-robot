"""
模型配置文件
"""

# 默认模型配置
DEFAULT_MODEL_CONFIG = {
    "memory": {
        "memory_key": "chat_history",
        "return_messages": True,
        "input_key": "question",
        "output_key": "answer",
        "human_prefix": "Human",
        "ai_prefix": "Assistant"
    },
    "prompt": {
        "system_message": "你是一个有帮助的AI助手，请用中文回答用户的问题。请基于以下上下文信息来回答问题：\n{context}"
    },
    "retrieval": {
        "docs_directory": "statics",
        "collection_name": "my_documents",
        "location": ":memory:"
    },
    "llm": {
        "temperature": 0.7,
        "request_timeout": 30,
        "streaming": True
    }
}

# DeepSeek模型特定配置
DEEPSEEK_CONFIG = {
    "model": "deepseek-chat",
    "api_base": "https://tbnx.plus7.plus/v1"
}

# GPT2模型特定配置
GPT2_CONFIG = {
    "default_service_url": "http://localhost:8018",
    "timeout": 30
} 