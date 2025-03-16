# AI 聊天机器人

这是一个基于 DeepSeek Chat 和 FastAPI 构建的聊天机器人应用。

## 功能特点

- 使用 DeepSeek Chat 模型进行对话
- 基于 Gradio 的美观聊天界面
- 实时响应
- 历史消息记录
- 支持消息复制
- 提供示例问题

## 安装步骤

1. 克隆项目并安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
- 复制 `.env.example` 文件为 `.env`
- 在 `.env` 文件中填入你的 DeepSeek API 密钥

## 运行应用

1. 启动后端服务：
```bash
uvicorn src.api.endpoints:app --reload
```

2. 在新的终端窗口中启动前端应用：
```bash
python src/frontend/app.py
```

3. 在浏览器中访问：
- 前端界面：http://localhost:7860
- API 文档：http://localhost:8000/docs

## 技术栈

- FastAPI
- Gradio
- OpenAI Client
- DeepSeek Chat
- Python 3.8+ 