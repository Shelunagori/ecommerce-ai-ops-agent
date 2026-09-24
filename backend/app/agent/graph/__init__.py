"""LangGraph StateGraph version of the commerce assistant (Step 6; policy RAG in Step 9).

Built alongside the Step-5 ``CommerceAssistant`` manual loop, which stays as the reference
implementation and parity oracle (compared through the test-only ``STEP5_PARITY_PROFILE``).
Production uses ``RAG_PROFILE``: prompt ``commerce-assistant-v2`` plus the RETRIEVE node for
``search_policy_knowledge``. No approvals/interrupts, write tools or durable checkpointing.
"""

from app.agent.graph.builder import build_commerce_graph
from app.agent.graph.runner import CommerceGraphAssistant
from app.agent.graph.state import CommerceGraphState, checkpoint_thread_key

__all__ = [
    "CommerceGraphAssistant",
    "CommerceGraphState",
    "build_commerce_graph",
    "checkpoint_thread_key",
]
