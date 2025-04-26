"""
基础模型类
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from langchain.callbacks import get_openai_callback
from langchain_core.language_models import BaseLLM

from src.model.callbacks import get_callbacks, safe_api_call
from src.model.chains import setup_prompt, create_retrieval_chain
from src.model.config import DEFAULT_MODEL_CONFIG
from src.model.memory import setup_memory
from src.model.retrieval import setup_vectorstore
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("base_model")

class BaseModel(ABC):
    """模型基类，定义通用接口"""
    @abstractmethod
    async def generate_response(self, message: str) -> str:
        """
        生成响应
        
        Args:
            message: 用户输入消息
            
        Returns:
            模型生成的响应文本
        """
        pass
    
    @abstractmethod
    async def generate_response_stream(self, message: str):
        """
        流式生成响应
        
        Args:
            message: 用户输入消息
            
        Yields:
            模型生成的响应文本片段
        """
        pass

class RetrievalModel(BaseModel):
    """包含检索功能的模型"""
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化检索模型
        
        Args:
            config: 模型配置
        """
        self.config = config or DEFAULT_MODEL_CONFIG
        self.memory = setup_memory(self.config)
        self.prompt = setup_prompt(self.config)
        self.vectorstore = setup_vectorstore(self.config)
        self.llm = None
        self.chain = None
        logger.info("检索模型基础组件已初始化")
    
    def _setup_llm(self, api_key: str) -> BaseLLM:
        """
        设置语言模型
        
        Args:
            api_key: API密钥
            
        Returns:
            配置好的语言模型
        """
        pass
    
    def _setup_chain(self):
        """设置检索链"""
        if not self.llm:
            raise ValueError("语言模型未初始化")
        
        retriever = self.vectorstore.as_retriever()
        self.chain = create_retrieval_chain(self.llm, self.prompt, retriever, self.memory)
    
    async def generate_response(self, message: str) -> str:
        """
        生成响应
        
        Args:
            message: 用户输入消息
            
        Returns:
            模型生成的响应文本
        """
        start_time = time.time()
        
        try:
            with get_openai_callback() as cb:
                callbacks = get_callbacks(verbose=True)
                result = await self.chain.ainvoke(
                    {"question": message},
                    config={"callbacks": callbacks, "verbose": True}
                )
                logger.info(f"Token使用情况: {cb}")
                
        except Exception as e:
            result = {"answer": f"API调用失败: {str(e)}"}
            logger.error(f"API调用错误: {str(e)}")
            
        finally:
            runtime = time.time() - start_time
            logger.info(f"请求完成，耗时: {runtime:.2f}秒")
            
        return result.get("answer", "无法获取响应")
    
    async def generate_response_stream(self, message: str):
        """
        流式生成响应
        
        Args:
            message: 用户输入消息
            
        Yields:
            模型生成的响应文本片段
        """
        try:
            with get_openai_callback() as cb:
                callbacks = get_callbacks(verbose=True)
                full_response = ""
                
                async for chunk in self.chain.astream(
                    message,
                    config={"callbacks": callbacks, "verbose": True}
                ):
                    if isinstance(chunk, str):
                        full_response += chunk
                        yield chunk
                    elif isinstance(chunk, dict):
                        content = chunk.get("answer", "")
                        if content:
                            full_response += content
                            yield content
                
                # 更新内存
                self.memory.save_context(
                    {"question": message},
                    {"answer": full_response}
                )
                
                logger.info(f"Token使用情况: {cb}")
                logger.info(f"生成的响应: {full_response[:100]}...")
                
        except Exception as e:
            logger.error(f"流式生成响应错误: {str(e)}")
            yield f"发生错误: {str(e)}"

class APIModel(BaseModel):
    """外部API调用模型"""
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化API模型
        
        Args:
            config: 模型配置
        """
        self.config = config or DEFAULT_MODEL_CONFIG
        self.memory = setup_memory(self.config)
        logger.info("API模型基础组件已初始化")
    
    @abstractmethod
    async def _call_api(self, message: str) -> str:
        """
        调用外部API
        
        Args:
            message: 用户输入消息
            
        Returns:
            API返回的响应文本
        """
        pass
    
    async def generate_response(self, message: str) -> str:
        """
        生成响应
        
        Args:
            message: 用户输入消息
            
        Returns:
            模型生成的响应文本
        """
        start_time = time.time()
        
        try:
            result = await safe_api_call(self._call_api, message)
            if "error" in result:
                return f"API调用失败: {result['error']}"
            
            # 更新内存
            self.memory.save_context(
                {"question": message},
                {"answer": result}
            )
            
        except Exception as e:
            result = f"API调用失败: {str(e)}"
            logger.error(f"API调用错误: {str(e)}")
            
        finally:
            runtime = time.time() - start_time
            logger.info(f"请求完成，耗时: {runtime:.2f}秒")
            
        return result
    
    async def generate_response_stream(self, message: str):
        """
        流式生成响应 (对于不支持流式输出的API，模拟流式输出)
        
        Args:
            message: 用户输入消息
            
        Yields:
            模型生成的响应文本片段
        """
        try:
            result = await self.generate_response(message)
            
            # 模拟流式输出
            for i in range(0, len(result), 5):
                chunk = result[i:i+5]
                yield chunk
                await asyncio.sleep(0.01)  # 小暂停模拟流式效果
                
        except Exception as e:
            logger.error(f"流式生成响应错误: {str(e)}")
            yield f"发生错误: {str(e)}" 