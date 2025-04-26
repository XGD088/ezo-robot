"""
回调和监控相关代码
"""
from typing import List, Dict, Any, Optional

from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler

from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("callbacks")

# 默认回调处理程序
console_handler = ConsoleCallbackHandler()
streaming_handler = StreamingStdOutCallbackHandler()

def get_callbacks(verbose: bool = True) -> List[Any]:
    """
    获取回调处理程序列表
    
    Args:
        verbose: 是否启用详细日志
        
    Returns:
        回调处理程序列表
    """
    callbacks = []
    if verbose:
        callbacks.append(console_handler)
    return callbacks

class TokenUsageTracker:
    """Token用量跟踪器"""
    @staticmethod
    async def track(func, *args, **kwargs):
        """
        跟踪函数调用的token用量
        
        Args:
            func: 要执行的异步函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果和token用量
        """
        with get_openai_callback() as cb:
            result = await func(*args, **kwargs)
            logger.info(f"Token使用情况: {cb}")
            return result, cb

async def safe_api_call(func, *args, **kwargs):
    """
    安全地调用API函数
    
    Args:
        func: 要执行的异步函数
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        函数执行结果或错误信息
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API调用错误: {str(e)}")
        return {"error": str(e)} 