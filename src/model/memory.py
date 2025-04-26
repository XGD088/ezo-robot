"""
内存和对话历史管理相关代码
"""
from typing import Dict, Any

from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("memory")

def setup_memory(config: Dict[str, Any]) -> ConversationBufferMemory:
    """
    设置对话内存
    
    Args:
        config: 内存配置
        
    Returns:
        配置好的对话内存
    """
    memory_config = config.get("memory", {})
    
    memory = ConversationBufferMemory(
        memory_key=memory_config.get("memory_key", "chat_history"),
        return_messages=memory_config.get("return_messages", True),
        input_key=memory_config.get("input_key", "question"),
        output_key=memory_config.get("output_key", "answer"),
        chat_memory=ChatMessageHistory(),
        human_prefix=memory_config.get("human_prefix", "Human"),
        ai_prefix=memory_config.get("ai_prefix", "Assistant"),
        output_messages_key=memory_config.get("memory_key", "chat_history"),
        message_class=HumanMessage,
        ai_message_class=AIMessage
    )
    
    logger.info(f"内存已配置: {memory_config.get('memory_key', 'chat_history')}")
    return memory 