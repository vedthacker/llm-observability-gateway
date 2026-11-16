import pytest

from llmscope import ChatRequest, Gateway
from llmscope.cache import ResponseCache
from llmscope.clock import VirtualClock
from llmscope.gateway import GatewayConfig
from llmscope.providers import MockProvider
from llmscope.store import Store


@pytest.fixture()
def clock():
    return VirtualClock()


@pytest.fixture()
def make_gateway(clock):
    def make(cache=True, **cfg):
        provider = MockProvider(clock)
        c = ResponseCache(clock, semantic_threshold=0.85) if cache is True else (None if cache is False else cache)
        gw = Gateway(provider, Store(), clock, cache=c, config=GatewayConfig(**cfg))
        return gw, provider
    return make


def ask(gw, text, **kw):
    return gw.chat(ChatRequest([{"role": "user", "content": text}], **kw))
