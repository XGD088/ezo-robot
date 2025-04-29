"""
模型管理器实现
"""
from typing import Dict, List, Optional, Any

from src.model.base_model import BaseModel
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("model_manager")

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