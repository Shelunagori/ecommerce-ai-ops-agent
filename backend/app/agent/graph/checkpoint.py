"""Ephemeral checkpointer factory, so code outside ``app/agent/graph`` (e.g. the RAG
evaluation harness) never imports LangGraph directly. In-memory only: nothing durable."""

from langgraph.checkpoint.memory import InMemorySaver


def ephemeral_checkpointer() -> InMemorySaver:
    return InMemorySaver()
