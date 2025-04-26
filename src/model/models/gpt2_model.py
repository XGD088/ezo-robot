"""
GPT2模型实现
"""
import aiohttp
from typing import Dict, Any

from src.model.base_model import APIModel
from src.model.config import DEFAULT_MODEL_CONFIG, GPT2_CONFIG
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("gpt2_model")

class GPT2Model(APIModel):
    """GPT2模型实现"""
    def __init__(self, service_url: str = None, config: Dict[str, Any] = None):
        """
        初始化GPT2模型
        
        Args:
            service_url: GPT2服务URL
            config: 模型配置
        """
        # 合并配置
        merged_config = DEFAULT_MODEL_CONFIG.copy()
        merged_config.update(config or {})
        
        super().__init__(merged_config)
        
        self.service_url = service_url or GPT2_CONFIG.get("default_service_url")
        if not self.service_url:
            raise ValueError("GPT2服务URL不能为空")
            
        logger.info(f"GPT2模型已初始化, 服务URL: {self.service_url}")
    
    async def _call_api(self, message: str) -> str:
        """
        调用GPT2 API
        
        Args:
            message: 用户输入消息
            
        Returns:
            API返回的响应文本
        """
        try:
            timeout = aiohttp.ClientTimeout(total=GPT2_CONFIG.get("timeout", 30))
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
                        
        except aiohttp.ClientError as e:
            logger.error(f"GPT2服务连接错误: {str(e)}")
            return f"GPT2服务连接失败: {str(e)}"
        except Exception as e:
            logger.error(f"GPT2服务调用错误: {str(e)}")
            return f"GPT2服务调用失败: {str(e)}" 