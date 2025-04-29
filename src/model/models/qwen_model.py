"""
通义千问模型实现
"""
from typing import Dict, Any, Optional

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

from src.model.base_model import RetrievalModel
from src.model.config import DEFAULT_MODEL_CONFIG, QWEN_CONFIG
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("qwen_model")

class QwenModel(RetrievalModel):
    """通义千问模型实现"""
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """
        初始化通义千问模型
        
        Args:
            api_key: 通义千问 API密钥
            config: 模型配置
        """
        # 合并配置
        merged_config = DEFAULT_MODEL_CONFIG.copy()
        merged_config.update(config or {})
        
        super().__init__(merged_config)
        
        if not api_key:
            raise ValueError("通义千问 API密钥不能为空")
        
        self.api_key = api_key
        self.llm = self._setup_llm(api_key)
        self._setup_chain()
        logger.info("通义千问模型已初始化")
    
    def _setup_llm(self, api_key: str) -> BaseLLM:
        """
        设置通义千问语言模型
        
        Args:
            api_key: API密钥
            
        Returns:
            配置好的语言模型
        """
        llm_config = self.config.get("llm", {})
        
        llm = ChatOpenAI(
            model=QWEN_CONFIG.get("model", "qwen-turbo"),
            openai_api_key=api_key,
            openai_api_base=QWEN_CONFIG.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=llm_config.get("temperature", 0.7),
            request_timeout=llm_config.get("request_timeout", 30),
            streaming=llm_config.get("streaming", True),
            model_kwargs={
                "stream": True  # 确保 API 调用也启用流式
            }
        )
        
        logger.info(f"通义千问 LLM已配置: {QWEN_CONFIG.get('model', 'qwen-turbo')}")
        return llm
    
    def _setup_chain(self):
        """设置检索链"""
        if not self.llm:
            raise ValueError("语言模型未初始化")
        
        # 创建检索器
        retriever = self.vectorstore.as_retriever()
        
        # 创建文档链
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # 创建自定义链
        self.chain = RunnableSequence(
            {
                "context": retriever, 
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"]
            }
            | document_chain
        )
        
        logger.info("通义千问检索链已配置") 