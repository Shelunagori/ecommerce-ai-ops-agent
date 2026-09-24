"""Graph behaviour profiles. ONE graph implementation; the profile only selects the prompt
and whether the policy-knowledge capability (RETRIEVE node + grounding) is enabled.

* ``RAG_PROFILE`` - production default: ``commerce-assistant-v2`` + policy RAG.
* ``STEP5_PARITY_PROFILE`` - TEST-ORIENTED compatibility mode: ``commerce-assistant-v1``,
  no policy capability, no grounding. It exists so the Step-5 manual loop can be compared
  with the graph apples-to-apples. Not used by any production entry point.
"""

from dataclasses import dataclass
from types import ModuleType

from app.agent.prompts import assistant as v1_prompt
from app.agent.prompts import graph_rag as v2_prompt


@dataclass(frozen=True)
class GraphProfile:
    name: str
    prompt: ModuleType  # PROMPT_ID, PROMPT_VERSION, SYSTEM_PROMPT, build_messages
    policy_knowledge: bool

    @property
    def prompt_version(self) -> str:
        return self.prompt.PROMPT_VERSION


RAG_PROFILE = GraphProfile("rag", v2_prompt, policy_knowledge=True)
STEP5_PARITY_PROFILE = GraphProfile("step5-parity", v1_prompt, policy_knowledge=False)
