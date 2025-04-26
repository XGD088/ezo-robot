from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Qdrant
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from openai.types import Embedding
from pydantic import Field
import os
from openai import OpenAI
import aiohttp
import json
import asyncio
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.files.file_manager import load_documents, split_documents
from src.utils.logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage

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

class GPT2Model(BaseModel):
    def __init__(self, service_url: str = "http://localhost:8018"):
        super().__init__()
        if not service_url:
            raise ValueError("GPT2服务URL不能为空")
        self.service_url = service_url
        self._setup_model_specific_components()

    def _setup_model_specific_components(self):
        """设置GPT2特定的组件"""
        self.llm = None  # GPT2 不需要 LLM
        self.chain = None  # GPT2 不需要 chain

    async def generate_response(self, message: str) -> str:
        """生成响应"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 设置30秒超时
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"向GPT2服务发送请求: {self.service_url}/predict")
                logger.info(f"请求内容: {message}")
                
                async with session.post(
                    f"{self.service_url}/predict",
                    json={"input_data": message}
                ) as response:
                    logger.info(f"GPT2服务响应状态码: {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"GPT2服务响应内容: {result}")
                        
                        # 尝试不同的响应字段
                        if "generated_text" in result:
                            return result["generated_text"]
                        elif "prediction" in result:
                            return result["prediction"]
                        elif "text" in result:
                            return result["text"]
                        else:
                            logger.error(f"GPT2服务响应格式不正确: {result}")
                            return "无法解析GPT2服务的响应"
                    else:
                        error_text = await response.text()
                        logger.error(f"GPT2服务错误响应: {error_text}")
                        return f"GPT2服务错误: {response.status} - {error_text}"
                        
        except asyncio.TimeoutError:
            logger.error("GPT2服务请求超时")
            return "GPT2服务请求超时，请稍后重试"
        except Exception as e:
            logger.error(f"GPT2服务调用错误: {str(e)}")
            return f"GPT2服务调用失败: {str(e)}"

class ModelManager:
    _instance = None
    _models: Dict[str, BaseModel] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_model(cls, name: str, model: BaseModel):
        cls._models[name] = model
        logger.info(f"模型已注册: {name}")
        logger.info(f"当前已注册的模型列表: {list(cls._models.keys())}")

    @classmethod
    def get_model(cls, name: str) -> Optional[BaseModel]:
        model = cls._models.get(name)
        if not model:
            logger.warning(f"模型未找到: {name}")
            logger.warning(f"当前可用的模型: {list(cls._models.keys())}")
        return model

    @classmethod
    def list_models(cls) -> list:
        models = list(cls._models.keys())
        logger.info(f"ModelManager.list_models() 返回的模型列表: {models}")
        return models