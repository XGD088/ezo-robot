from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from dotenv import load_dotenv
from src.model import ModelManager, DeepSeekModel, GPT2Model
from src.utils.logger import setup_logger
import json

# 加载环境变量
load_dotenv()
logger = setup_logger("endpoints")

# 初始化 FastAPI 应用
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 初始化模型管理器
model_manager = ModelManager()

# 注册可用的模型
model_manager.register_model(
    "deepseek", 
    DeepSeekModel(os.getenv("DEEPSEEK_API_KEY"))
)

model_manager.register_model(
    "gpt2", 
    GPT2Model(os.getenv("GPT2_SERVICE_URL", "http://localhost:8018"))
)

class ChatInput(BaseModel):
    message: str
    model_name: str = "deepseek"  # 默认使用 DeepSeek 模型

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    """
    聊天接口
    :param data: 用户输入的消息和选择的模型
    :return: AI的回复
    """
    try:
        # 获取选择的模型
        model = model_manager.get_model(data.model_name)
        if not model:
            raise HTTPException(status_code=400, detail=f"模型 {data.model_name} 不存在")
        
        # 使用选择的模型生成响应
        async def generate():
            async for chunk in model.generate_response_stream(message=data.message):
                yield f"data: {json.dumps({'response': chunk})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"聊天接口错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """
    获取可用的模型列表
    """
    models = model_manager.list_models()
    logger.info(f"API返回的模型列表: {models}")
    return {
        "models": models
    }

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"}