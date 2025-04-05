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

from src.files.file_manager import load_documents, split_documents
from src.utils.logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

# 加载环境变量
load_dotenv()

# 配置日志
logger = setup_logger("model_manager")

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
            return_messages=True
        )
        
        # 配置基础 prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的AI助手，请用中文回答用户的问题。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
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
        retriever = self.vectorstore.as_retriever()
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            verbose=True
        )

    async def generate_response(self, message: str) -> str:
        """通用的响应生成方法"""
        start_time = time.time()
        
        try:
            with get_openai_callback() as cb:
                result = await self.chain.ainvoke({"question": message})
                logger.info(f"Token使用情况: {cb}")
                
        except Exception as e:
            result = f"API调用失败: {str(e)}"
            logger.error(f"API调用错误: {str(e)}")
            
        finally:
            runtime = time.time() - start_time
            logger.info(f"请求完成，耗时: {runtime:.2f}秒")
            
        return result["answer"]

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
            request_timeout=30
        )
        self.chain = self._create_chain(self.llm)

class GPT2Model(BaseModel):
    def __init__(self, service_url: str = "http://localhost:8018"):
        super().__init__()
        if not service_url:
            raise ValueError("GPT2服务URL不能为空")
        self._setup_model_specific_components(service_url)

    def _setup_model_specific_components(self, service_url: str):
        """设置GPT2特定的组件"""
        class GPT2LLM(BaseLLM):
            service_url: str = Field()

            def __init__(self, service_url: str):
                super().__init__(service_url=service_url)

            def _generate(self, prompts: list[str], **kwargs) -> str:
                import requests
                response = requests.post(
                    f"{self.service_url}/predict",
                    json={"input_data": prompts[0]},  # 只处理第一个提示
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()["prediction"]
                return f"GPT2服务错误: {response.text}"

            @property
            def _llm_type(self) -> str:
                return "gpt2"

        self.llm = GPT2LLM(service_url)
        self.chain = self._create_chain(self.llm)

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

    @classmethod
    def get_model(cls, name: str) -> Optional[BaseModel]:
        model = cls._models.get(name)
        if not model:
            logger.warning(f"模型未找到: {name}")
        return model

    @classmethod
    def list_models(cls) -> list:
        return list(cls._models.keys())