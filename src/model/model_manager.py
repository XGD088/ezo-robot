import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any

import aiohttp
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import Qdrant
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.files.file_manager import load_documents, split_documents
from src.utils.logger import setup_logger

# 加载环境变量
load_dotenv()

# 配置日志
logger = setup_logger("model_manager")

# 创建回调处理程序
console_handler = ConsoleCallbackHandler()
streaming_handler = StreamingStdOutCallbackHandler()

class TongyiEmbeddings(Embeddings):
    """通义千问的 embeddings 实现"""
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.batch_size = 10  # 通义千问 API 的批量限制

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文本转换为向量"""
        try:
            all_embeddings = []
            # 分批处理文本
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                response = self.client.embeddings.create(
                    model="text-embedding-v3",
                    input=batch_texts,
                    encoding_format="float"
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        except Exception as e:
            logger.error(f"通义千问 embeddings 调用失败: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """将单个文本转换为向量"""
        return self.embed_documents([text])[0]

class BaseModel(ABC):
    def __init__(self):
        self.chain = None
        self.memory = None
        self.prompt = None
        self.vectorstore = None
        self._setup_common_components()

    def _setup_common_components(self):
        """设置所有模型共用的组件"""
        # 配置 memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
            chat_memory=ChatMessageHistory(),
            human_prefix="Human",
            ai_prefix="Assistant",
            output_messages_key="chat_history",  # 明确指定输出消息键
            message_class=HumanMessage,  # 指定消息类型
            ai_message_class=AIMessage,  # 指定 AI 消息类型
            memory_key_prefix="chat_history"  # 明确指定内存键前缀
        )
        
        # 配置基础 prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的AI助手，请用中文回答用户的问题。请基于以下上下文信息来回答问题：\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # 加载文档并创建向量存储
        documents = load_documents("statics")
        texts = split_documents(documents)
        embeddings = TongyiEmbeddings()  # 使用通义千问的 embeddings
        self.vectorstore = Qdrant.from_documents(
            documents=texts,
            embedding=embeddings,
            location=":memory:",
            collection_name="my_documents"
        )

    def _create_chain(self, llm):
        """创建对话链"""
        from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
        from langchain.chains.combine_documents import create_stuff_documents_chain
        
        # 创建检索器
        retriever = self.vectorstore.as_retriever()
        
        # 创建文档链
        document_chain = create_stuff_documents_chain(llm, self.prompt)
        
        # 创建自定义链
        chain = RunnableSequence(
            {
                "context": retriever, 
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"]
            }
            | document_chain
        )
        
        return chain

    async def generate_response(self, message: str) -> str:
        """通用的响应生成方法"""
        start_time = time.time()
        
        try:
            with get_openai_callback() as cb:
                # 添加控制台回调处理程序
                result = await self.chain.ainvoke(
                    {"question": message},
                    config={"callbacks": [console_handler], "verbose": True}
                )
                logger.info(f"Token使用情况: {cb}")
                
        except Exception as e:
            result = f"API调用失败: {str(e)}"
            logger.error(f"API调用错误: {str(e)}")
            
        finally:
            runtime = time.time() - start_time
            logger.info(f"请求完成，耗时: {runtime:.2f}秒")
            
        return result["answer"]

    async def generate_response_stream(self, message: str):
        """流式生成响应"""
        try:
            with get_openai_callback() as cb:  # 添加监控
                # 使用 chain 进行流式输出，而不是直接使用 LLM
                full_response = ""
                async for chunk in self.chain.astream(
                    message,
                    config={"callbacks": [console_handler], "verbose": True}
                ):
                    if isinstance(chunk, str):
                        full_response += chunk
                        yield chunk
                    elif isinstance(chunk, dict):
                        # 处理字典类型的输出
                        content = chunk.get("answer", "")
                        if content:
                            full_response += content
                            yield content
                
                # 更新内存
                self.memory.save_context(
                    {"question": message},
                    {"answer": full_response}
                )
                
                logger.info(f"Token使用情况: {cb}")  # 记录 token 使用情况
                logger.info(f"生成的响应: {full_response[:100]}...")  # 记录响应的前100个字符
        except Exception as e:
            logger.error(f"流式生成响应错误: {str(e)}")
            yield f"发生错误: {str(e)}"

    @abstractmethod
    def _setup_model_specific_components(self):
        """设置模型特定的组件"""
        pass

class DeepSeekModel(BaseModel):
    def __init__(self, api_key: str):
        super().__init__()
        if not api_key:
            raise ValueError("DeepSeek API密钥不能为空")
        self._setup_model_specific_components(api_key)

    def _setup_model_specific_components(self, api_key: str):
        """设置DeepSeek特定的组件"""
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://tbnx.plus7.plus/v1",
            temperature=0.7,
            request_timeout=30,
            streaming=True,  # 启用流式输出
            model_kwargs={
                "stream": True  # 确保 API 调用也启用流式
            }
        )
        self.chain = self._create_chain(self.llm)

class ModelManager:
    """模型管理器，负责注册和获取模型"""
    _instance = None
    _models: Dict[str, BaseModel] = {}

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_model(cls, name: str, model: BaseModel) -> None:
        """
        注册模型
        
        Args:
            name: 模型名称
            model: 模型实例
        """
        cls._models[name] = model
        logger.info(f"模型已注册: {name}")
        logger.info(f"当前已注册的模型列表: {list(cls._models.keys())}")

    @classmethod
    def get_model(cls, name: str) -> Optional[BaseModel]:
        """
        获取模型
        
        Args:
            name: 模型名称
            
        Returns:
            模型实例，如果不存在则返回None
        """
        model = cls._models.get(name)
        if not model:
            logger.warning(f"模型未找到: {name}")
            logger.warning(f"当前可用的模型: {list(cls._models.keys())}")
        return model

    @classmethod
    def list_models(cls) -> List[str]:
        """
        获取所有已注册的模型名称
        
        Returns:
            模型名称列表
        """
        models = list(cls._models.keys())
        logger.info(f"ModelManager.list_models() 返回的模型列表: {models}")
        return models

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        创建模型实例（工厂方法）
        
        Args:
            model_type: 模型类型
            **kwargs: 模型参数
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 如果模型类型不支持
        """
        from src.model.models import DeepSeekModel, QwenModel
        
        if model_type.lower() == "deepseek":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("DeepSeek模型需要提供API密钥")
            return DeepSeekModel(api_key, kwargs.get("config"))
        elif model_type.lower() == "qwen":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("通义千问模型需要提供API密钥")
            return QwenModel(api_key, kwargs.get("config"))
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    @classmethod
    def register_models_from_config(cls, config: Dict[str, Any]) -> None:
        """
        从配置中注册多个模型
        
        Args:
            config: 模型配置字典
        """
        for name, model_config in config.items():
            model_type = model_config.get("type")
            try:
                model = cls.create_model(model_type, **model_config)
                cls.register_model(name, model)
            except Exception as e:
                logger.error(f"注册模型 {name} 失败: {str(e)}")