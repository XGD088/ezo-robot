import gradio as gr
import requests
import json
import sseclient

def get_available_models():
    """
    获取可用的模型列表
    """
    try:
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            print(f"获取模型列表成功: {response.json()}")
            return response.json()["models"]
        return ["qwen", "deepseek"]  # 默认模型列表，千问排在前面
    except Exception as e:
        print(f"获取模型列表失败: {str(e)}")
        return ["qwen", "deepseek"]  # 默认模型列表，千问排在前面

def chat_with_bot(message, history, model_name):
    """
    与后端API交互的函数
    """
    try:
        # 发送请求并获取SSE流
        response = requests.post(
            "http://localhost:8000/chat",
            json={
                "message": message,
                "model_name": model_name
            },
            stream=True,
            headers={'Accept': 'text/event-stream'}
        )
        
        if response.status_code == 200:
            # 创建SSE客户端
            client = sseclient.SSEClient(response)
            full_response = ""
            
            # 处理每个事件
            for event in client.events():
                if event.data:
                    try:
                        data = json.loads(event.data)
                        if "response" in data:
                            full_response += data["response"]
                            # 使用 yield 返回更新后的历史记录
                            yield history + [[message, full_response]]
                    except json.JSONDecodeError:
                        # 如果数据不是JSON格式，直接使用原始数据
                        full_response += event.data
                        yield history + [[message, full_response]]
        else:
            yield history + [[message, f"错误: {response.json()['detail']}"]]
    except Exception as e:
        yield history + [[message, f"发生错误: {str(e)}"]]

# 获取可用的模型列表
available_models = get_available_models()

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI 聊天助手")
    gr.Markdown("这是一个支持多模型的AI聊天助手。请选择要使用的模型，然后开始对话。")
    
    with gr.Row():
        # 设置默认选择为通义千问模型
        default_model = "qwen" if "qwen" in available_models else available_models[0]
        model_dropdown = gr.Dropdown(
            choices=available_models,
            value=default_model,
            label="选择模型"
        )
    
    chatbot = gr.Chatbot(
        height=600,
        show_label=False,
        show_share_button=False,
        show_copy_button=True,
        layout="bubble",
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="输入消息",
            placeholder="请输入您的问题...",
            lines=3
        )
        send = gr.Button("发送")
    
    with gr.Row():
        clear = gr.Button("清除对话")
        retry = gr.Button("重试")
        undo = gr.Button("撤销")
    
    gr.Examples(
        examples=[
            ["你好，请介绍一下你自己"],
            ["什么是人工智能？"],
            ["你能帮我写一个Python程序吗？"]
        ],
        inputs=msg
    )
    
    # 事件处理
    send.click(
        chat_with_bot,
        inputs=[msg, chatbot, model_dropdown],
        outputs=chatbot,
        show_progress=True  # 显示进度条
    ).then(
        lambda: "",
        None,
        msg
    )
    
    clear.click(lambda: None, None, chatbot)
    undo.click(lambda x: x[:-1], chatbot, chatbot)
    retry.click(
        lambda x, y, z: chat_with_bot(x[-1][0], x[:-1], z),
        inputs=[chatbot, msg, model_dropdown],
        outputs=chatbot,
        show_progress=True  # 显示进度条
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    ) 