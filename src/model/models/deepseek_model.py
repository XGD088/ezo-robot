"""
DeepSeek模型实现
"""
from typing import Dict, Any, Optional

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

from src.model.base_model import RetrievalModel
from src.model.config import DEFAULT_MODEL_CONFIG, DEEPSEEK_CONFIG
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("deepseek_model")

class DeepSeekModel(RetrievalModel):
    """DeepSeek模型实现"""
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """
        初始化DeepSeek模型
        
        Args:
            api_key: DeepSeek API密钥
            config: 模型配置
        """
        # 合并配置
        merged_config = DEFAULT_MODEL_CONFIG.copy()
        merged_config.update(config or {})
        
        super().__init__(merged_config)
        
        if not api_key:
            raise ValueError("DeepSeek API密钥不能为空")
        
        self.api_key = api_key
        self.llm = self._setup_llm(api_key)
        self._setup_chain()
        logger.info("DeepSeek模型已初始化")
    
    def _setup_llm(self, api_key: str) -> BaseLLM:
        """
        设置DeepSeek语言模型
        
        Args:
            api_key: API密钥
            
        Returns:
            配置好的语言模型
        """
        llm_config = self.config.get("llm", {})
        
        llm = ChatOpenAI(
            model=DEEPSEEK_CONFIG.get("model", "deepseek-chat"),
            openai_api_key=api_key,
            openai_api_base=DEEPSEEK_CONFIG.get("api_base", "https://tbnx.plus7.plus/v1"),
            temperature=llm_config.get("temperature", 0.7),
            request_timeout=llm_config.get("request_timeout", 30),
            streaming=llm_config.get("streaming", True),
            model_kwargs={
                "stream": True  # 确保 API 调用也启用流式
            }
        )
        
        logger.info(f"DeepSeek LLM已配置: {DEEPSEEK_CONFIG.get('model', 'deepseek-chat')}")
        return llm 