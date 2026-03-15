"""
app.py - 阿里云百炼 DeepSeek V3.2 模型 Streamlit 交互界面
支持单轮对话、流式输出、参数调节、历史记录等功能
"""
import streamlit as st
import time
from datetime import datetime
from typing import Generator
import os

# 导入我们之前写好的工具类
from langchain_demo.langchain_util import get_deepseek_util, DeepSeekV32Util

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="DeepSeek V3.2 对话助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* 聊天消息样式 */
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 15px 15px 0 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 15px 15px 15px 0;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    
    .message-time {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    /* 侧边栏样式 */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #333;
    }
    
    /* 参数卡片样式 */
    .param-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    /* 按钮样式 */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* 成功提示 */
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* 警告提示 */
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
def init_session_state():
    """初始化所有会话状态变量"""

    # 聊天历史记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 模型实例（懒加载）
    if "util" not in st.session_state:
        st.session_state.util = None

    # 模型参数
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "model_name": "deepseek-v3.2"  # 根据你的实际模型名修改
        }

    # 对话统计
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "total_messages": 0,
            "total_tokens": 0,
            "session_start": datetime.now()
        }

    # 流式输出控制
    if "streaming" not in st.session_state:
        st.session_state.streaming = True

# ==================== 初始化模型 ====================
@st.cache_resource
def init_model():
    """初始化DeepSeek模型（缓存实例）"""
    try:
        util = get_deepseek_util()
        return util
    except Exception as e:
        st.error(f"❌ 模型初始化失败：{str(e)}")
        return None

# ==================== 检查API Key ====================
def check_api_key():
    """检查API Key是否配置"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        st.markdown("""
        <div class="warning-box">
            ⚠️ 未检测到 API Key！请在 <code>.env</code> 文件中设置 
            <code>DASHSCOPE_API_KEY=你的密钥</code>
        </div>
        """, unsafe_allow_html=True)
        return False

    # 显示API Key状态（隐藏部分字符）
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "已配置"
    st.markdown(f"""
    <div class="success-box">
        ✅ API Key 已配置：<code>{masked_key}</code>
    </div>
    """, unsafe_allow_html=True)
    return True

# ==================== 侧边栏 ====================
def render_sidebar():
    """渲染侧边栏配置"""

    with st.sidebar:
        st.image("https://img.alicdn.com/imgextra/i1/O1CN01HZCb7g1IhRKmQz2cY_!!6000000000927-2-tps-200-200.png",
                 width=100, caption="阿里云百炼")

        st.markdown("<div class='sidebar-header'>⚙️ 模型参数配置</div>",
                   unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='param-card'>", unsafe_allow_html=True)

            # 模型选择
            model_name = st.text_input(
                "🤖 模型名称",
                value=st.session_state.model_params["model_name"],
                help="请输入你在阿里云百炼平台部署的模型名称"
            )

            # 温度参数
            temperature = st.slider(
                "🌡️ 温度 (Temperature)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.model_params["temperature"],
                step=0.1,
                help="控制输出的随机性：较低的值更确定，较高的值更多样"
            )

            # 最大输出长度
            max_tokens = st.number_input(
                "📏 最大输出长度 (Max Tokens)",
                min_value=1,
                max_value=4096,
                value=st.session_state.model_params["max_tokens"],
                step=64,
                help="模型生成的最大token数量"
            )

            # 流式输出开关
            streaming = st.checkbox(
                "⚡ 流式输出",
                value=st.session_state.streaming,
                help="开启后可以看到逐字输出的效果"
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # 更新session state
            st.session_state.model_params.update({
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            st.session_state.streaming = streaming

        st.markdown("<div class='sidebar-header'>💬 对话管理</div>",
                   unsafe_allow_html=True)

        # 清空对话按钮
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # 示例提示词
        st.markdown("<div class='sidebar-header'>📝 示例提示词</div>",
                   unsafe_allow_html=True)

        examples = [
            "用Python写一个快速排序算法",
            "解释一下什么是区块链技术",
            "写一首关于春天的诗",
            "如何学习人工智能？给出学习路线图",
            "用一句话解释量子计算"
        ]

        for example in examples:
            if st.button(f"📌 {example}", use_container_width=True, key=f"example_{example[:10]}"):
                st.session_state.current_input = example

        # 显示统计信息
        st.markdown("<div class='sidebar-header'>📊 会话统计</div>",
                   unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("对话轮次", st.session_state.stats["total_messages"])
        with col2:
            duration = datetime.now() - st.session_state.stats["session_start"]
            st.metric("会话时长", f"{duration.seconds//60}分{duration.seconds%60}秒")

# ==================== 显示聊天历史 ====================
def display_chat_history():
    """显示所有聊天记录"""

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 显示时间戳
            st.caption(f"🕐 {msg['time']}")

# ==================== 处理用户输入 ====================
def process_user_input(prompt: str):
    """处理用户输入并生成回复"""

    # 添加用户消息到历史
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.stats["total_messages"] += 1

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 获取模型回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # 获取模型实例
            util = st.session_state.util or init_model()
            if not util:
                st.error("模型未初始化，请检查配置")
                return

            # 根据流式输出模式选择不同的处理方式
            if st.session_state.streaming:
                # 流式输出
                for chunk in util.stream_chat(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)  # 稍微减速，让效果更明显
                message_placeholder.markdown(full_response)
            else:
                # 非流式输出
                with st.spinner("🤔 思考中..."):
                    full_response = util.single_chat(prompt)
                message_placeholder.markdown(full_response)

            # 添加助手回复到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "time": datetime.now().strftime("%H:%M:%S")
            })

        except Exception as e:
            error_msg = f"❌ 出错了：{str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "time": datetime.now().strftime("%H:%M:%S")
            })

# ==================== 主页面 ====================
def main():
    """主函数"""

    # 初始化session state
    init_session_state()

    # 标题
    st.markdown("""
    <div class='main-title'>
        <h1>🤖 阿里云百炼 DeepSeek V3.2 对话助手</h1>
        <p>基于Streamlit构建的智能对话界面 | 支持流式输出、参数调节、历史记录</p>
    </div>
    """, unsafe_allow_html=True)

    # 检查API Key
    api_key_ok = check_api_key()

    # 渲染侧边栏
    render_sidebar()

    # 主对话区域
    if api_key_ok:
        # 初始化模型（如果还没初始化）
        if st.session_state.util is None:
            with st.spinner("🔄 正在初始化模型..."):
                st.session_state.util = init_model()

        # 显示聊天历史
        display_chat_history()

        # 聊天输入框
        prompt = st.chat_input("请输入你的问题...")

        # 如果有预设的输入（从示例按钮来）
        if hasattr(st.session_state, 'current_input') and st.session_state.current_input:
            prompt = st.session_state.current_input
            st.session_state.current_input = None

        # 处理用户输入
        if prompt:
            process_user_input(prompt)
            st.rerun()

    else:
        # 如果没有API Key，显示配置指南
        st.info("请在 `.env` 文件中配置你的阿里云百炼 API Key 后再使用本应用")

        with st.expander("📖 如何配置 API Key？"):
            st.markdown("""
            1. 登录[阿里云百炼控制台](https://bailian.console.aliyun.com/)
            2. 进入"模型部署"页面，找到你的 DeepSeek V3.2 模型
            3. 点击"查看调用信息"，复制你的 API Key
            4. 在项目根目录创建 `.env` 文件，添加:""")
# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()