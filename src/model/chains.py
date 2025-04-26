"""
链相关代码
"""
from typing import Dict, Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from langchain_core.language_models import BaseLLM

from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("chains")

def setup_prompt(config: Dict[str, Any]) -> ChatPromptTemplate:
    """
    设置提示模板
    
    Args:
        config: 提示模板配置
        
    Returns:
        配置好的提示模板
    """
    prompt_config = config.get("prompt", {})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_config.get("system_message", "你是一个有帮助的AI助手，请用中文回答用户的问题。")),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    logger.info("提示模板已配置")
    return prompt

def create_retrieval_chain(llm: BaseLLM, prompt: ChatPromptTemplate, retriever: Any, memory: Any) -> RunnableSequence:
    """
    创建检索链
    
    Args:
        llm: 语言模型
        prompt: 提示模板
        retriever: 检索器
        memory: 内存
        
    Returns:
        配置好的检索链
    """
    # 创建文档链
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 创建自定义链
    chain = RunnableSequence(
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
        }
        | document_chain
    )
    
    logger.info("检索链已创建")
    return chain 