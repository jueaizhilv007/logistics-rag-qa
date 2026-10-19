"""向量化：OpenAI 兼容协议的 Embedding 客户端。

本地（Ollama / vLLM）与云端（agicto 等）走同一套协议，
差异只在 base_url / model / api_key，业务层不感知。

本模块被 import 时会向注册表注册实现，属于配置驱动路径的插件，
显式注入路径不使用本文件。
"""

from __future__ import annotations

from openai import OpenAI

from rag_core.config import EndpointCfg
from rag_core.core.logging import get_logger
from rag_core.registry import register


@register("embedding", "agicto")
class OpenAICompatibleEmbedding:
    """按批大小分批调用 embeddings 接口。

    分批是必要的：一次传入上千条文本会被服务端截断或超时，
    而失败时无法定位是哪一条出错，整批重跑成本高。
    """

    def __init__(self, endpoint: EndpointCfg, api_key: str):
        self.client = OpenAI(base_url=endpoint.base_url, api_key=api_key or "sk-noop")
        self.model = endpoint.model
        self.batch_size = endpoint.batch_size
        self._dim: int | None = None
        self.log = get_logger("rag_core.embedding")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
            self.log.debug("向量化批次 %d-%d 完成", i, i + len(batch))
        if self._dim is None and out:
            self._dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        vec = resp.data[0].embedding
        if self._dim is None:
            self._dim = len(vec)
        return vec

    @property
    def dimension(self) -> int:
        """未调用过时返回 0。

        维度不预先配置，而由首个 batch 的实际返回决定：
        换 embedding 模型时维度会变，写死会让建表与查询错位。
        """
        return self._dim or 0


@register("embedding", "local")
class LocalEmbedding(OpenAICompatibleEmbedding):
    """本地服务（Ollama / vLLM）实现，协议与云端一致。"""
