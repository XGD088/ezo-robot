from abc import ABC, abstractmethod
from openai import AsyncOpenAI, AuthenticationError
import httpx
from typing import Dict, Any, Optional
import requests
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain.schema import LLMResult
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self):
        # 初始化 LangSmith 客户端和追踪器
        try:
            self.langsmith_client = Client()
            self.tracer = LangChainTracer(project_name="ezo-robot-chat")
            self.callback_manager = CallbackManager([self.tracer])
            logger.info(f"LangSmith追踪器初始化成功: {self.__class__.__name__}")
        except Exception as e:
            logger.warning(f"LangSmith追踪器初始化失败: {str(e)}")
            self.callback_manager = None

    @abstractmethod
    async def generate_response(self, message: str) -> str:
        pass

    async def _track_run(self, message: str, response: str, metadata: Dict[str, Any]):
        """记录模型运行数据到LangSmith"""
        if not self.callback_manager:
            return

        try:
            # 创建运行记录
            run = self.langsmith_client.create_run(
                name=f"{self.__class__.__name__}_chat",
                run_type="llm",  # 指定运行类型为 LLM
                inputs={"message": message},
                start_time=metadata.get("start_time"),
                extra={
                    "model_name": self.__class__.__name__,
                    "status": metadata.get("status", "success"),
                    **metadata
                }
            )
            
            # 更新运行结果
            self.langsmith_client.update_run(
                run.id,
                outputs={"response": response},
                error=metadata.get("error"),
                end_time=metadata.get("end_time")
            )
            
            logger.info(f"运行记录已保存到LangSmith: {run.id}")
        except Exception as e:
            logger.error(f"保存运行记录失败: {str(e)}")

class DeepSeekModel(BaseModel):
    def __init__(self, api_key: str):
        super().__init__()
        if not api_key:
            raise ValueError("DeepSeek API密钥不能为空")
        
        self.client = AsyncOpenAI(
            base_url='https://tbnx.plus7.plus/v1',
            api_key=api_key,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        )

    async def generate_response(self, message: str) -> str:
        start_time = time.time()
        metadata = {
            "start_time": start_time,
            "model": "deepseek-chat",
            "status": "started"
        }
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答用户的问题。"},
                    {"role": "user", "content": message}
                ],
                stream=False,
                timeout=30
            )
            result = response.choices[0].message.content
            metadata.update({
                "status": "success",
                "token_count": len(message.split()),
                "response_length": len(result)
            })
            
        except AuthenticationError as e:
            result = f"DeepSeek API认证错误: 请检查您的API密钥是否正确。错误详情: {str(e)}"
            metadata.update({"status": "auth_error", "error": str(e)})
            
        except Exception as e:
            result = f"DeepSeek API调用失败: {str(e)}"
            metadata.update({"status": "error", "error": str(e)})
            
        finally:
            end_time = time.time()
            metadata.update({
                "end_time": end_time,
                "runtime": end_time - start_time
            })
            await self._track_run(message, result, metadata)
            
        return result

class GPT2Model(BaseModel):
    def __init__(self, service_url: str = "http://localhost:8018"):
        super().__init__()
        if not service_url:
            raise ValueError("GPT2服务URL不能为空")
        self.service_url = service_url

    async def generate_response(self, message: str) -> str:
        start_time = time.time()
        metadata = {
            "start_time": start_time,
            "model": "gpt2-local",
            "service_url": self.service_url,
            "status": "started"
        }
        
        try:
            response = requests.post(
                f"{self.service_url}/predict",
                json={"input_data": message},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["prediction"]
                metadata.update({
                    "status": "success",
                    "response_code": 200,
                    "token_count": len(message.split()),
                    "response_length": len(result)
                })
            else:
                result = f"GPT2服务错误: {response.text}"
                metadata.update({
                    "status": "api_error",
                    "response_code": response.status_code,
                    "error": response.text
                })
                
        except requests.exceptions.ConnectionError:
            result = "GPT2服务连接失败: 请确认服务是否正在运行"
            metadata.update({
                "status": "connection_error",
                "error": "Connection refused"
            })
            
        except Exception as e:
            result = f"调用GPT2服务失败: {str(e)}"
            metadata.update({
                "status": "error",
                "error": str(e)
            })
            
        finally:
            end_time = time.time()
            metadata.update({
                "end_time": end_time,
                "runtime": end_time - start_time
            })
            await self._track_run(message, result, metadata)
            
        return result

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