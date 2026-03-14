"""
langchain_util.py - 封装阿里云百炼 deepseekv3.2 模型的通用调用工具
完全兼容版 - 自动适配不同版本的LangChain
"""
import os
import logging
from typing import Optional, Generator, Any
from dotenv import load_dotenv

# ===================== 兼容性导入 =====================
# 尝试多种导入方式，兼容不同版本的LangChain
try:
    # LangChain 0.3+ 版本
    from langchain_core.pydantic_v1 import SecretStr
except ImportError:
    try:
        # LangChain 0.2.x 版本
        from langchain_core.utils import SecretStr
    except ImportError:
        # 最后的备选方案：使用标准库
        from pydantic import SecretStr

# ChatOpenAI 导入（LangChain 统一接口）
from langchain_openai import ChatOpenAI

# 回调处理
try:
    from langchain_core.callbacks import StreamingStdOutCallbackHandler
except ImportError:
    from langchain_core.callbacks import StreamingStdOutCallbackHandler

# 提示词模板
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate

# ===================== 基础配置 =====================
# 加载.env文件中的环境变量
load_dotenv()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("langchain_util")

# ===================== 核心工具类 =====================
class DeepSeekV32Util:
    """阿里云百炼 deepseekv3.2 模型调用工具类"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "deepseek-v3.2",  # 替换成你在百炼里的准确模型名
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        # 获取API Key
        key_value = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key_value:
            raise ValueError("未配置DASHSCOPE_API_KEY！请在.env文件中设置")

        # 处理SecretStr（兼容不同版本）
        if isinstance(key_value, str):
            self.api_key = SecretStr(key_value)
        else:
            self.api_key = key_value

        # 阿里云百炼OpenAI兼容接口地址
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 初始化模型（不在这里初始化llm，而是在调用时动态创建）
        # 这样可以避免初始化时的兼容性问题

        logger.info(f"DeepSeekV3.2 工具类初始化成功，模型名：{model_name}")

    def _get_llm(self, streaming: bool = False) -> ChatOpenAI:
        """获取模型实例（延迟初始化）"""
        common_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
            "base_url": self.base_url,
        }

        if streaming:
            # 流式模型
            return ChatOpenAI(
                **common_params,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        else:
            # 非流式模型
            return ChatOpenAI(**common_params, streaming=False)

    def single_chat(self, prompt: str) -> str:
        """单轮对话：一次性返回完整结果"""
        try:
            logger.info(f"开始单轮调用，提问：{prompt[:50]}...")
            llm = self._get_llm(streaming=False)
            response = llm.invoke(prompt)

            # 兼容不同版本的返回格式
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)

            logger.info(f"单轮调用完成，回复长度：{len(result)}")
            return result
        except Exception as e:
            logger.error(f"单轮调用失败：{str(e)}", exc_info=True)
            return f"调用失败：{str(e)}"

    def stream_chat(self, prompt: str) -> Generator[str, None, None]:
        """流式对话：逐字返回结果，适配前端打字效果"""
        try:
            logger.info(f"开始流式调用，提问：{prompt[:50]}...")
            llm = self._get_llm(streaming=True)

            for chunk in llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str) and chunk:
                    yield chunk

            logger.info("流式调用完成")
        except Exception as e:
            logger.error(f"流式调用失败：{str(e)}", exc_info=True)
            yield f"调用失败：{str(e)}"

    def chat_with_template(self, template: str, **kwargs) -> str:
        """带提示词模板的调用，适配结构化输出场景"""
        try:
            # 动态获取输入变量
            input_vars = list(kwargs.keys())

            # 创建模板（兼容不同版本）
            try:
                prompt_template = PromptTemplate(
                    template=template,
                    input_variables=input_vars
                )
            except TypeError:
                # 某些版本可能不需要 input_variables
                prompt_template = PromptTemplate(template=template)

            # 格式化提示词
            try:
                prompt = prompt_template.format(**kwargs)
            except AttributeError:
                # 兼容旧版本
                prompt = template.format(**kwargs)

            logger.info(f"带模板调用，生成提示词：{prompt[:50]}...")

            llm = self._get_llm(streaming=False)
            response = llm.invoke(prompt)

            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"带模板调用失败：{str(e)}", exc_info=True)
            return f"调用失败：{str(e)}"

    def batch_chat(self, prompts: list) -> list:
        """批量对话（可选功能）"""
        results = []
        for prompt in prompts:
            try:
                result = self.single_chat(prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"批量调用中单条失败：{str(e)}")
                results.append(f"失败：{str(e)}")
        return results


# ===================== 快捷单例调用 =====================
def get_deepseek_util() -> DeepSeekV32Util:
    """快捷获取工具实例，避免重复初始化"""
    if not hasattr(get_deepseek_util, "_instance"):
        get_deepseek_util._instance = DeepSeekV32Util()
    return get_deepseek_util._instance


# ===================== 便捷函数 =====================
def simple_chat(prompt: str) -> str:
    """最简单的单轮对话函数"""
    return get_deepseek_util().single_chat(prompt)


def stream_chat(prompt: str) -> Generator[str, None, None]:
    """最简单的流式对话函数"""
    return get_deepseek_util().stream_chat(prompt)


# ===================== 本地测试 =====================
if __name__ == "__main__":
    print("="*50)
    print("开始测试阿里云百炼 DeepSeekV3.2 调用")
    print("="*50)

    try:
        # 获取工具实例
        util = get_deepseek_util()

        # 测试1：单轮调用
        print("\n【测试1】单轮调用测试")
        print("-"*30)
        response1 = util.single_chat("请用一句话解释什么是人工智能")
        print(f"回复：{response1}")

        # 测试2：带模板调用
        print("\n【测试2】带模板调用测试")
        print("-"*30)
        template = "请用{word_count}个字解释{concept}"
        response2 = util.chat_with_template(
            template,
            word_count=20,
            concept="深度学习"
        )
        print(f"回复：{response2}")

        # 测试3：流式调用
        print("\n【测试3】流式调用测试")
        print("-"*30)
        print("流式输出：", end="", flush=True)
        for chunk in util.stream_chat("请举一个人工智能应用的例子"):
            print(chunk, end="", flush=True)
        print("\n")

        # 测试4：简单函数调用
        print("\n【测试4】简单函数调用测试")
        print("-"*30)
        response4 = simple_chat("你好，请做个自我介绍")
        print(f"回复：{response4}")

        print("\n" + "="*50)
        print("✅ 所有测试完成！")
        print("="*50)

    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
        print("\n请检查：")
        print("1. .env 文件中是否有正确的 DASHSCOPE_API_KEY")
        print("2. 网络连接是否正常")
        print("3. 模型名称是否与百炼平台一致")