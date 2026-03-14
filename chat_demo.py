# 导入核心依赖库
import streamlit as st
import requests
import json


# ===================== 核心函数：调用本地Ollama API（必须放在最前面） =====================
def call_ollama_model(prompt, model_name="deepseek-r1:1.5b", stream=True):
    """
    调用本地Ollama API，支持流式/非流式输出
    :param prompt: 用户输入的问题/提示词
    :param model_name: 本地Ollama模型名（需和ollama list显示的一致）
    :param stream: 是否流式输出（逐字显示，模拟打字效果）
    :return: 流式生成的回复片段（generator）/完整回复字符串
    """
    # Ollama默认本地API地址（端口11434，无需修改，除非手动改了Ollama端口）
    ollama_api_url = "http://localhost:11434/api/chat"

    # 构建Ollama API请求体（严格遵循Ollama的Chat API规范）
    request_body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": 0.7,  # 回复随机性：0(严谨)~1(创意)
        "top_p": 0.9  # 采样参数，保持默认即可
    }

    try:
        # 发送POST请求到Ollama本地服务
        response = requests.post(
            url=ollama_api_url,
            json=request_body,
            stream=stream,  # 流式输出必须开启stream参数
            timeout=300  # 超时时间：5分钟（适配大模型慢回复）
        )
        response.raise_for_status()  # 捕获HTTP错误（如404/500）

        # 流式输出（推荐：逐字显示，体验更好）
        if stream:
            for line in response.iter_lines():
                if line:  # 过滤空行
                    # 解析Ollama返回的逐行JSON
                    line_data = json.loads(line.decode("utf-8"))
                    # 提取回复内容
                    if "message" in line_data and "content" in line_data["message"]:
                        yield line_data["message"]["content"]
                    # 检测回复结束标记
                    if line_data.get("done", False):
                        break
        # 非流式输出（一次性返回完整回复）
        else:
            full_response = response.json()
            return full_response["message"]["content"]

    # 异常处理：覆盖所有常见错误场景
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ 无法连接到Ollama服务！请检查：\n1. Ollama是否已启动（sudo systemctl start ollama）\n2. 端口11434是否被占用")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ 请求超时！模型回复时间过长，可尝试：\n1. 切换更小的模型（如llama3:8b）\n2. 增加timeout参数值")
        return None
    except requests.exceptions.HTTPError as e:
        if "404" in str(e):
            st.error(f"❌ 模型 {model_name} 不存在！请先执行：ollama pull {model_name}")
        else:
            st.error(f"❌ HTTP请求错误：{str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ 未知错误：{str(e)}")
        return None


# ===================== 页面基础配置（自定义标题/图标/布局） =====================
st.set_page_config(
    page_title="本地Ollama聊天助手",  # 浏览器标签标题
    page_icon="💬",  # 浏览器标签图标（emoji/本地图片）
    layout="wide",  # 页面布局：wide（宽屏）/centered（居中）
    initial_sidebar_state="expanded"  # 侧边栏默认展开
)

# ===================== 初始化会话状态（保存聊天记录/加载状态） =====================
# 1. 聊天历史：存储{"role": "user/assistant", "content": 消息内容}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# 2. 加载状态：防止重复发送请求（输入框禁用）
if "loading" not in st.session_state:
    st.session_state.loading = False

# ===================== 侧边栏：模型选择/功能按钮 =====================
with st.sidebar:
    st.title("⚙️ 配置中心")

    # 模型选择框：必须和本地ollama list显示的模型名一致！
    # 示例：["llama3:8b", "qwen:7b", "gemma:7b", "mistral:7b"]
    model_name = st.selectbox(
        label="选择本地模型",
        options=["deepseek-r1:1.5b"],  # 🔴 替换成你的本地模型名
        index=0,
        help="需先通过 ollama pull 模型名 拉取，执行 ollama list 查看本地模型"
    )

    # 清空聊天记录按钮
    if st.button("🗑️ 清空聊天记录", type="primary", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()  # 刷新页面，立即清空

    # 显示本地Ollama提示
    st.divider()  # 分隔线
    st.info(
        """
        📌 操作提示：
        1. 确保Ollama已启动：sudo systemctl start ollama
        2. 拉取模型：ollama pull 模型名
        3. 查看本地模型：ollama list
        """
    )

# ===================== 主页面：聊天界面 =====================
st.title("💬 本地Ollama聊天助手")
st.caption(f"✅ 当前使用模型：{model_name} | 数据全程本地化，无隐私泄露")
st.divider()

# 渲染历史聊天记录
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入框：用户提问
if prompt := st.chat_input(
        placeholder="请输入你的问题...",
        disabled=st.session_state.loading,
        key="chat_input"
):
    # 步骤1：添加用户消息到历史并渲染
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 步骤2：调用Ollama模型生成回复（流式）
    st.session_state.loading = True
    # 终端打印日志（调试用）
    print(f"\n===== 调用模型：{model_name} | 用户提问：{prompt[:30]}... =====")

    with st.chat_message("assistant"):
        # 占位符：实时更新回复内容
        response_placeholder = st.empty()
        full_response = ""

        # 调用模型，逐字接收回复
        for chunk in call_ollama_model(prompt, model_name=model_name, stream=True):
            if chunk:
                full_response += chunk
                # 显示打字光标效果
                response_placeholder.markdown(full_response + "▌")

        # 最终渲染：去掉光标，显示完整回复
        response_placeholder.markdown(full_response)

    # 步骤3：添加助手回复到历史，重置加载状态
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    st.session_state.loading = False

# ===================== 页脚 =====================
st.divider()
st.caption("© 2026 本地Ollama聊天助手 | 基于Streamlit + Ollama构建")