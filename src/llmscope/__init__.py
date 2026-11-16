"""llmscope - observability, cost control and state-management optimisation for LLM applications."""

__version__ = "0.1.0"

from .gateway import ChatRequest, ChatResult, Gateway, GatewayError  # noqa: E402,F401
from .tracing import Tracer  # noqa: E402,F401
