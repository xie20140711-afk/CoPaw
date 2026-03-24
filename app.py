import gradio as gr
import os
import requests
import json

# 简化的 CoPaw 演示版本
# 使用 Hugging Face Inference API（免费）

HF_API_URL = "https://api-inference.huggingface.co/models/"
DEFAULT_MODEL = "microsoft/DialoGPT-medium"

def query_huggingface(prompt, model=DEFAULT_MODEL):
    """使用 Hugging Face Inference API"""
    try:
        # 如果有 HF Token，使用它获得更高配额
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        
        response = requests.post(
            f"{HF_API_URL}{model}",
            headers=headers,
            json={"inputs": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "无响应")
            return str(result)
        elif response.status_code == 429:
            return "🚦 系统繁忙，请稍后再试（Hugging Face API 限制）"
        else:
            return f"API 错误: {response.status_code}"
    except Exception as e:
        return f"请求出错: {str(e)}"

def chat(message, history):
    """聊天功能 - 轻量级版本"""
    if not message.strip():
        return "请输入消息"
    
    # 构建对话历史
    context = ""
    for h in history[-3:]:  # 只保留最近3轮对话
        if isinstance(h, tuple):
            context += f"User: {h[0]}\nAssistant: {h[1]}\n"
    
    prompt = f"{context}User: {message}\nAssistant:"
    
    # 调用 API
    response = query_huggingface(prompt)
    
    # 清理响应
    response = response.replace(prompt, "").strip()
    if not response:
        response = "我在思考中... 请稍后再试"
    
    return response

# 创建界面
with gr.Blocks(title="CoPaw Lite - AI 助手") as demo:
    gr.Markdown("""
    # 🤖 CoPaw Lite
    ### 轻量级 AI 助手演示
    
    ⚡ **快速启动** | 🆓 **免费使用** | 🚀 **基于 Hugging Face**
    """)
    
    chatbot = gr.ChatInterface(
        fn=chat,
        examples=[
            "你好，介绍一下自己",
            "讲个笑话",
            "用Python写个Hello World",
            "什么是机器学习？"
        ],
        title="开始对话",
        description="输入消息与 AI 助手交流"
    )
    
    gr.Markdown("""
    ---
    💡 **提示**: 如果收到 429 错误，说明 API 暂时繁忙，请稍等片刻再试。
    
    🔑 **提升体验**: 在 Settings > Secrets 中添加 `HF_TOKEN` 获得更高配额。
    获取 Token: https://huggingface.co/settings/tokens
    """)

# 启动
if __name__ == "__main__":
    # 获取端口（Render 使用 10000）
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )