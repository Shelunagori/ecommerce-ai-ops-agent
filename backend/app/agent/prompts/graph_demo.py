"""System prompt ``commerce-assistant-v3-public-demo``: v2 (policy RAG) for the READ-ONLY
public demo (Supabase anonymous visitors).

The graph for this profile binds no action tools, so a write is impossible regardless of the
prompt; the rules below only make the model explain that restriction politely instead of
attempting one.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.agent.prompts import graph_rag as v2

PROMPT_ID = "commerce_assistant"
PROMPT_VERSION = "commerce-assistant-v3-public-demo"

READ_ONLY_RULES = """\

Public demo (read-only)
- This is a public, read-only demo session on synthetic data. You cannot change anything:
  no order cancellations, refunds, store credit or other updates are available here.
- If the user asks for a change, say that the public demo is read-only and that actions
  require a reviewer account with approval rights. You may still look up the relevant facts
  and policy so the user sees what would apply."""

SYSTEM_PROMPT = v2.SYSTEM_PROMPT.replace("\n\nGeneral\n", READ_ONLY_RULES + "\n\nGeneral\n")
assert SYSTEM_PROMPT != v2.SYSTEM_PROMPT  # noqa: S101 - import-time guard on the splice


def build_messages(text: str) -> list[BaseMessage]:
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)]
