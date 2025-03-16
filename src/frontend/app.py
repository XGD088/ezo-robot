import gradio as gr
import requests
import json

def chat_with_bot(message, history):
    """
    与后端API交互的函数
    """
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": message}
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "抱歉，服务器出现错误，请稍后重试。"
    except Exception as e:
        return f"发生错误: {str(e)}"

# 创建Gradio界面
demo = gr.ChatInterface(
    chat_with_bot,
    chatbot=gr.Chatbot(
        height=600,
        show_label=False,
        show_share_button=False,
        show_copy_button=True,
        layout="bubble",
    ),
    title="🤖 AI 聊天助手",
    description="这是一个基于 GPT-4 和 LangChain 构建的聊天机器人。请输入您的问题，AI 会为您解答。",
    theme=gr.themes.Soft(),
    examples=[
        ["你好，请介绍一下你自己"],
        ["什么是人工智能？"],
        ["你能帮我写一个Python程序吗？"]
    ],
    retry_btn="重试",
    undo_btn="撤销",
    clear_btn="清除",
)

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    ) 