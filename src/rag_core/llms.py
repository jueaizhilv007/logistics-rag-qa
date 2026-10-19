"""LLM 客户端：OpenAI 兼容协议的对话接口。

temperature 默认取端点的配置值（本项目为 0）：
政策问答要求同一问题得到同一答案，否则引用编号不稳定，溯源链路会失效。
"""

from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from rag_core.config import EndpointCfg
from rag_core.registry import register


@register("llm", "agicto")
class OpenAICompatibleLLM:
    """非流式与流式共用一套参数构造，避免两条路径行为不一致。"""

    def __init__(self, endpoint: EndpointCfg, api_key: str):
        self.client = OpenAI(base_url=endpoint.base_url, api_key=api_key or "sk-noop")
        self._model = endpoint.model
        self.temperature = endpoint.temperature
        self.max_tokens = endpoint.max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    def _params(self) -> dict:
        params = {"model": self._model, "temperature": self.temperature}
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        return params

    def chat(self, messages: list[dict], **kwargs) -> str:
        params = self._params()
        params.update(kwargs)
        resp = self.client.chat.completions.create(messages=messages, **params)
        return resp.choices[0].message.content or ""

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        params = self._params()
        params.update(kwargs)
        stream = self.client.chat.completions.create(
            messages=messages, stream=True, **params
        )
        for event in stream:
            choices = getattr(event, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0].delta, "content", None)
            if delta:
                yield delta


@register("llm", "local")
class LocalLLM(OpenAICompatibleLLM):
    """本地服务（Ollama / vLLM）实现，协议与云端一致。"""
