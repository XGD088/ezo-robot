from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import httpx
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 FastAPI 应用
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url='https://tbnx.plus7.plus/v1',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    http_client=httpx.Client()
)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    """
    聊天接口
    :param data: 用户输入的消息
    :return: AI的回复
    """
    try:
        # 使用 DeepSeek API 处理消息
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答用户的问题。"},
                {"role": "user", "content": data.message}
            ],
            stream=False
        )
        
        return {
            "response": response.choices[0].message.content,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"}