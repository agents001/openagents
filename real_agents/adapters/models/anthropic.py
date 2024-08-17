from typing import Any, Dict, List, Optional
from langchain_core.pydantic_v1 import Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
#from langchain.llms.anthropic import _AnthropicCommon
from langchain_community.llms.anthropic import _AnthropicCommon
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)


class ChatAnthropic(BaseChatModel, _AnthropicCommon):
    r"""Wrapper around Anthropic's large language model.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python
            import anthropic
            from langchain.llms import Anthropic
            model = ChatAnthropic(model="<model_name>", anthropic_api_key="my-api-key")
    """
    stop: Optional[List[str]] = None # stop 参数可以用来提供一组终止生成的字符串，但它不是调用该函数所必需的。如果用户没有提供这个参数，函数将自动将其视为 None，从而可能使用某种默认行为或设置。

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore    # 告诉 Pydantic 在输入数据中忽略未声明的额外字段。也就是说，如果传入的数据包含模型中未定义的字段，这些字段将被自动忽略，而不会导致错误。

    '''
    这个属性的主要目的是标识这个聊天模型的类型。
    当其他部分的代码需要知道正在使用的是哪种类型的语言模型时，可以通过访问 _llm_type 属性来获取这个信息。
    在这种情况下，它总是返回 "anthropic-chat"，表明这是一个 Anthropic 的聊天模型。
    '''
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

    '''
    将不同类型的消息统一转换为特定格式的文本.
    HUMAN_PROMPT 很可能是在这个类的某个父类中定义的。
    具体来说，ChatAnthropic 类继承自 _AnthropicCommon，这个类可能在其中定义了 HUMAN_PROMPT、AI_PROMPT 等常量.
    '''
    def _convert_one_message_to_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"{self.HUMAN_PROMPT} {message.content}"
        elif isinstance(message, AIMessage):
            message_text = f"{self.AI_PROMPT} {message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"{self.HUMAN_PROMPT} <admin>{message.content}</admin>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    '''
    接受一个参数 messages，这是一个 BaseMessage 对象的列表。
    使用列表推导式遍历 messages 列表中的每个消息。
    对每个消息调用 self._convert_one_message_to_text 方法，将单个消息转换为文本格式。
    使用 "".join() 方法将所有转换后的消息文本连接成一个单一的字符串。
    返回这个组合后的字符串。
    将一系列不同类型的消息（可能包括人类消息、AI消息、系统消息等）转换并组合成一个连续的文本字符串.
    '''
    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Format a list of strings into a single string with necessary newlines.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary newlines.
        """
        return "".join(self._convert_one_message_to_text(message) for message in messages)

    '''
    _convert_messages_to_prompt 用于将一系列消息转换为 Anthropic 模型可以理解的完整提示.
    首先检查 self.AI_PROMPT 是否存在。如果不存在，抛出 NameError，提示需要加载 anthropic 包。
    检查消息列表的最后一个元素是否为 AIMessage。如果不是，则在列表末尾添加一个空的 AIMessage。这确保了提示以 AI 的回应结束.
    调用 self._convert_messages_to_text 方法将消息列表转换为单一文本字符串。
    使用 rstrip() 方法去除文本末尾可能存在的空格。这是为了删除可能由 "Assistant: " 引起的尾随空格。
    返回处理后的文本字符串。
    '''
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a full prompt for the Anthropic model

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
        if not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if not isinstance(messages[-1], AIMessage):
            messages.append(AIMessage(content=""))
        text = self._convert_messages_to_text(messages)
        return text.rstrip()  # trim off the trailing ' ' that might come from the "Assistant: "
    
    '''
    这个 _generate 方法是生成聊天响应的核心方法。它的功能如下：
    将消息列表转换为提示字符串。
    准备参数字典，包含提示和默认参数。
    处理停止序列（stop sequences）:
    如果类有预定义的停止序列，将其与传入的停止序列合并。
    如果有停止序列，将其添加到参数中。

    根据是否启用流式处理（streaming）分两种情况：
    a. 如果启用流式处理：
    逐步接收完成的文本。
    每次接收新的文本片段时，如果有 run_manager，调用其 on_llm_new_token 方法。
    b. 如果不启用流式处理：
    一次性获取完整的响应。

    将生成的文本包装成 AIMessage 对象。
    返回 ChatResult 对象，其中包含 ChatGeneration 对象。
    '''

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params}
        if self.stop is not None:
            if stop is None:
                stop = self.stop
            else:
                stop.extend(self.stop)
        if stop:
            params["stop_sequences"] = stop

        if self.streaming:
            completion = ""
            stream_resp = self.client.completion_stream(**params)
            for data in stream_resp:
                delta = data["completion"][len(completion) :]
                completion = data["completion"]
                if run_manager:
                    run_manager.on_llm_new_token(
                        delta,
                    )
        else:
            response = self.client.completion(**params)
            completion = response["completion"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    '''
    这个 _agenerate 方法是 _generate 方法的异步版本。它的功能和结构与同步版本非常相似，但使用了异步编程的特性。以下是主要区别和功能：
    方法定义使用 async def，表示这是一个异步方法。
    使用 await 关键字来等待异步操作完成。
    异步流处理：
    使用 async for 循环来异步迭代流式响应。
    调用 await self.client.acompletion_stream() 获取异步流。
    非流式处理也使用异步调用：await self.client.acompletion()。
    如果有 run_manager，使用 await run_manager.on_llm_new_token() 异步调用回调。
    其他逻辑（如参数准备、消息转换等）与同步版本相同。

    这个异步方法允许在不阻塞主线程的情况下生成响应，特别适用于需要处理多个并发请求的场景，如web服务器。它提供了与同步版本相同的功能，但可以更有效地利用系统资源，尤其是在I/O密集型操作（如API调用）中。
    '''
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params}
        if stop:
            params["stop_sequences"] = stop

        if self.streaming:
            completion = ""
            stream_resp = await self.client.acompletion_stream(**params)
            async for data in stream_resp:
                delta = data["completion"][len(completion) :]
                completion = data["completion"]
                if run_manager:
                    await run_manager.on_llm_new_token(
                        delta,
                    )
        else:
            response = await self.client.acompletion(**params)
            completion = response["completion"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])
