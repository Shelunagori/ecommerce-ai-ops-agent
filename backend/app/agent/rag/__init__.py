"""Policy knowledge for the LangGraph assistant (Step 9): the model-visible
``search_policy_knowledge`` capability, its model-facing context format, deterministic
final-answer grounding and the RAG/agent evaluation harness.

The capability is NOT a commerce tool: it is never registered in ``ToolExecutor``; the
graph's RETRIEVE node runs it with trusted tenant context and a fixed limit.
"""
