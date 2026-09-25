"""Grounding routing: deterministic commerce facts vs policy knowledge.

Regression (production, Cloudflare Workers AI): "Where is SHP-1003?" failed with
agent_grounding_error because the model wrote an invented ``policy://`` citation into a
commerce-only answer. Grounding never REQUIRES citations without retrieval; it rejects
citations that were not retrieved in this request. The fix keeps that rule and adds ONE
corrective model call for such an answer when no policy retrieval ran; RAG runs are
unchanged and fail closed.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agent.assistant import AssistantError, AssistantLimits
from app.agent.context import AgentContext
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.llm.provider import ChatModelProvider, ProviderInfo, RetryPolicy
from app.agent.prompts import graph_agent, graph_demo, graph_rag
from app.api.agent_runtime import AgentRuntime
from tests.assistant.fakes import ScriptedChatModel, ai_text, ai_tools, call
from tests.db.test_agent_stream import finished, parse_sse
from tests.db.test_provider_fallback_graph import Rig, Workers, cf_text, cf_tools
from tests.llm.test_cloudflare_provider import tool_call
from tests.rag.fakes import result as retrieval_result

V2 = "policy://delayed-shipment-compensation/v2#chunk-2"
INVENTED = "policy://shipment-tracking/v1#chunk-1"


class GeminiRig(Rig):
    def gemini_only(self, *steps):
        self.gemini = ScriptedChatModel(list(steps))
        provider = ChatModelProvider(
            self.gemini,  # type: ignore[arg-type]
            ProviderInfo(provider="gemini", model="gemini-3.8-flash"),
            RetryPolicy(max_retries=0),
            sleep=lambda _s: None,
        )
        self.assistant = CommerceGraphAssistant(
            provider,
            tools=self.tools,
            checkpointer=self.saver,
            retriever=self.retriever,
            profile=AGENT_PROFILE,
            actions=self.actions,
        )
        self.app.state.agent_runtime = AgentRuntime(self.assistant, self.actions)
        return self


@pytest.fixture
def rig(committed):
    return GeminiRig(committed)


def kinds(trace):
    return [(t["kind"], t["status"]) for t in trace]


# --- 1/3/10: Cloudflare commerce-only ------------------------------------------------------------
def test_cloudflare_commerce_only_answer_needs_no_policy_citation(rig, tenant_a):
    rig.install(
        Workers(
            cf_tools(tool_call("c1", "get_shipment", shipment_number="SHP-1003")),
            cf_text("Shipment SHP-1003 is delayed."),
        )
    )
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["answer"] == "Shipment SHP-1003 is delayed." and body["citations"] == []
    assert kinds(body["execution_trace"]) == [
        ("request", "completed"),
        ("model", "completed"),
        ("commerce_tool", "completed"),
        ("model", "completed"),
        ("response", "completed"),
    ]
    assert body["retrievals"] == []


def test_invented_citation_in_a_commerce_answer_gets_one_corrective_call(rig, tenant_a):
    """The production regression: the Cloudflare model decorates the shipment answer."""
    rig.install(
        Workers(
            cf_tools(tool_call("c1", "get_shipment", shipment_number="SHP-1003")),
            cf_text(f"Shipment SHP-1003 is delayed [{INVENTED}]."),
            cf_text("Shipment SHP-1003 is delayed."),
        )
    )
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["answer"] == "Shipment SHP-1003 is delayed." and body["citations"] == []
    assert INVENTED not in body["answer"]
    trace = body["execution_trace"]
    assert kinds(trace) == [
        ("request", "completed"),
        ("model", "completed"),
        ("commerce_tool", "completed"),
        ("model", "completed"),  # the rejected answer
        ("grounding", "rejected"),  # citation check: rejected, NOT a failed step
        ("model", "completed"),  # the corrected answer
        ("response", "completed"),
    ]
    check = trace[4]
    assert check["label"] == "Citation check" and check["metadata"] == {
        "citation_issue": "citation_not_retrieved"
    }
    assert not [t for t in trace if t["status"] == "failed"]
    assert body["model_calls"] == 3 and len(body["tool_calls"]) == 1  # tool not re-run
    # the corrective call saw the rejected answer + the application note, nothing else new
    from app.agent.graph.nodes import CITATION_CORRECTION_NOTE

    sent = rig.workers.body(2)["messages"]
    assert sent[-1] == {"role": "user", "content": CITATION_CORRECTION_NOTE}
    assert INVENTED in (sent[-2]["content"] or "")
    # neither the rejected answer nor the note is kept in the conversation
    history = rig.get(tenant_a, "/api/agent/threads/thread-1/messages").json()["messages"]
    assert [m["content"] for m in history] == [
        "Where is SHP-1003?",
        "Shipment SHP-1003 is delayed.",
    ]


def test_a_second_invented_citation_still_fails_closed(rig, tenant_a):
    rig.install(
        Workers(
            cf_tools(tool_call("c1", "get_shipment", shipment_number="SHP-1003")),
            cf_text(f"Delayed [{INVENTED}]."),
            cf_text(f"Still delayed [{INVENTED}]."),
        )
    )
    r = rig.post(tenant_a, "Where is SHP-1003?")
    assert (r.status_code, r.json()["error"]["code"]) == (502, "agent_grounding_error")
    assert len(rig.workers.requests) == 3  # exactly one corrective call


# --- 2: Gemini behaves identically --------------------------------------------------------------
def test_gemini_commerce_only_follows_the_same_rule(rig, tenant_a):
    rig.gemini_only(
        ai_tools(call("get_shipment", "g1", shipment_number="SHP-1003")),
        ai_text(f"Shipment SHP-1003 is delayed [{INVENTED}]."),
        ai_text("Shipment SHP-1003 is delayed."),
    )
    body = rig.post(tenant_a, "Where is SHP-1003?").json()
    assert body["answer"] == "Shipment SHP-1003 is delayed." and body["citations"] == []
    assert [
        t["metadata"].get("provider") for t in body["execution_trace"] if t["kind"] == "model"
    ] == ["gemini"] * 3
    assert isinstance(rig.gemini.invocations[2].messages[-1], HumanMessage)


# --- 4: not special-cased to shipments ------------------------------------------------------------
@pytest.mark.parametrize(
    ("tool", "args", "question"),
    [
        ("get_order", {"order_number": "ORD-1001"}, "Show me order ORD-1001"),
        ("get_customer", {"customer_code": "CUS-1001"}, "Who is CUS-1001?"),
        ("list_delayed_shipments", {}, "Which shipments are delayed?"),
    ],
)
def test_other_commerce_tools_follow_the_same_rule(rig, tenant_a, tool, args, question):
    rig.install(
        Workers(
            cf_tools(tool_call("c1", tool, **args)),
            cf_text(f"Here you go [{INVENTED}]."),
            cf_text("Here you go."),
        )
    )
    body = rig.post(tenant_a, question).json()
    assert body["answer"] == "Here you go." and body["tool_calls"][0]["tool"] == tool


# --- 5/6/7/8: RAG stays fail-closed ------------------------------------------------------------
def test_policy_only_request_still_retrieves_and_grounds(rig, tenant_b):
    rig.install(
        Workers(
            cf_tools(
                tool_call("r1", "search_policy_knowledge", query="delayed shipment compensation")
            ),
            cf_text(f"15% store credit [{V2}]."),
        )
    )
    body = rig.post(tenant_b, "What compensation applies to a delayed shipment?").json()
    assert [c["citation"] for c in body["citations"]] == [V2]
    assert ("grounding", "completed") in kinds(body["execution_trace"])
    assert ("retrieval", "completed") in kinds(body["execution_trace"])


def test_mixed_request_still_grounds_the_policy_part(rig, tenant_b):
    rig.install(
        Workers(
            cf_tools(tool_call("m1", "get_shipment", shipment_number="SHP-1003")),
            cf_tools(
                tool_call("m2", "search_policy_knowledge", query="delayed shipment compensation")
            ),
            cf_text(f"Delayed; 15% store credit [{V2}]."),
        )
    )
    body = rig.post(
        tenant_b, "Where is SHP-1003, and what compensation applies if it is delayed?"
    ).json()
    assert [k for k, _ in kinds(body["execution_trace"])] == [
        "request",
        "model",
        "commerce_tool",
        "model",
        "retrieval",
        "model",
        "grounding",
        "response",
    ]
    assert [c["citation"] for c in body["citations"]] == [V2]


@pytest.mark.parametrize(
    ("answer", "detail"),
    [
        ("15% store credit.", "citation_required"),  # policy retrieved, no citation
        (f"15% store credit [{INVENTED}].", "citation_not_retrieved"),  # wrong citation
    ],
)
def test_rag_answer_with_missing_or_invalid_citation_fails_without_a_second_chance(
    rig, tenant_b, answer, detail
):
    rig.install(
        Workers(
            cf_tools(tool_call("r1", "search_policy_knowledge", query="compensation")),
            cf_text(answer),
            cf_text(f"15% [{V2}]."),  # would be valid - must never be requested
        )
    )
    r = rig.post(tenant_b, "What compensation applies?")
    assert (r.status_code, r.json()["error"]["code"]) == (502, "agent_grounding_error")
    assert len(rig.workers.requests) == 2  # no corrective call on the RAG path


def test_mixed_request_with_an_invented_citation_fails_closed(rig, tenant_b):
    rig.install(
        Workers(
            cf_tools(tool_call("m1", "get_shipment", shipment_number="SHP-1003")),
            cf_tools(tool_call("m2", "search_policy_knowledge", query="compensation")),
            cf_text(f"Delayed; credit [{INVENTED}]."),
            cf_text("never requested"),
        )
    )
    r = rig.post(tenant_b, "Where is SHP-1003, and what compensation applies?")
    assert r.json()["error"]["code"] == "agent_grounding_error" and len(rig.workers.requests) == 3


def test_no_result_retrieval_keeps_its_fail_closed_rule(rig, tenant_b):
    rig.retriever.script.append(retrieval_result())
    rig.install(
        Workers(
            cf_tools(tool_call("n1", "search_policy_knowledge", query="warranty")),
            cf_text(f"Warranty is 2 years [{V2}]."),  # cites although nothing was found
            cf_text("never requested"),
        )
    )
    r = rig.post(tenant_b, "Warranty?")
    assert r.json()["error"]["code"] == "agent_grounding_error" and len(rig.workers.requests) == 2


def test_the_corrective_call_may_turn_into_a_real_policy_search(rig, tenant_b):
    """If the question needed policy after all, the model searches and normal RAG applies."""
    rig.install(
        Workers(
            cf_text(f"15% store credit [{V2}]."),  # answered from memory, no retrieval
            cf_tools(tool_call("r1", "search_policy_knowledge", query="compensation")),
            cf_text(f"15% store credit [{V2}]."),
        )
    )
    body = rig.post(tenant_b, "What compensation applies?").json()
    assert [c["citation"] for c in body["citations"]] == [V2]
    assert [k for k, s in kinds(body["execution_trace"])] == [
        "request",
        "model",
        "grounding",
        "model",
        "retrieval",
        "model",
        "grounding",
        "response",
    ]


def test_the_corrective_call_respects_the_round_limit(rig, tenant_a):
    rig.install(Workers(cf_text(f"Hello [{INVENTED}]."), cf_text("Hello.")))
    rig.assistant = CommerceGraphAssistant(
        rig.assistant._provider,
        tools=rig.tools,
        checkpointer=rig.saver,
        retriever=rig.retriever,
        profile=AGENT_PROFILE,
        actions=rig.actions,
        limits=AssistantLimits(max_model_rounds=1),
    )
    with pytest.raises(AssistantError) as info:
        rig.assistant.run("hi", AgentContext(tenant_a.tenant_id), thread_id="lim")
    assert info.value.code == "agent_grounding_error"
    assert len(rig.workers.requests) == 1


# --- 9: normalisation never makes a commerce answer "policy-grounded" -------------------------
def test_cloudflare_normalisation_never_marks_commerce_as_policy_grounded(rig, tenant_a):
    rig.install(
        Workers(
            cf_tools(tool_call("c1", "get_order", order_number="ORD-1001")),
            cf_text("ORD-1001 was delivered."),
        )
    )
    body = rig.post(tenant_a, "Show me order ORD-1001").json()
    assert body["citations"] == [] and body["retrievals"] == []
    assert not [t for t in body["execution_trace"] if t["kind"] in ("grounding", "retrieval")]
    sent = rig.workers.body(1)
    assert "policy://" not in str([m.get("content") for m in sent["messages"][1:]])


# --- 12: streaming and non-streaming agree -------------------------------------------------------
def test_stream_and_json_agree_for_the_corrected_commerce_answer(rig, tenant_a):
    script = (
        cf_tools(tool_call("c1", "get_shipment", shipment_number="SHP-1003")),
        cf_text(f"Shipment SHP-1003 is delayed [{INVENTED}]."),
        cf_text("Shipment SHP-1003 is delayed."),
    )
    rig.install(Workers(*script))
    plain = rig.post(tenant_a, "Where is SHP-1003?", thread="p").json()
    rig.install(Workers(*script))
    r = rig.client.post(
        "/api/agent/messages/stream",
        json={"text": "Where is SHP-1003?", "thread_id": "s"},
        headers={"X-Tenant-ID": str(tenant_a.tenant_id)},
    )
    ev = parse_sse(r.text)
    streamed = ev[-1]["response"]
    assert ev[-1]["type"] == "run_completed" and streamed["answer"] == plain["answer"]
    live = [
        (e["kind"], e["label"], e["status"], e.get("detail"), e["metadata"]) for e in finished(ev)
    ]
    final = [
        (t["kind"], t["label"], t["status"], t["detail"], t["metadata"])
        for t in streamed["execution_trace"]
    ]
    assert live == final  # the live trace converges on the final trace, correction included
    assert kinds(streamed["execution_trace"]) == kinds(plain["execution_trace"])
    assert not [e for e in ev if e["type"] == "step_failed"]


# --- prompt: the scope rule ships in both production prompts, v2 (RAG baseline) unchanged ------
def test_citation_scope_rule_is_in_the_production_prompts_only():
    assert graph_agent.PROMPT_VERSION == "commerce-assistant-v4"
    assert graph_demo.PROMPT_VERSION == "commerce-assistant-v4-public-demo"
    for prompt in (graph_agent.SYSTEM_PROMPT, graph_demo.SYSTEM_PROMPT):
        assert "Citations (scope)" in prompt and "contain no policy citation" in prompt
    assert "Citations (scope)" not in graph_rag.SYSTEM_PROMPT
    assert "Actions (approval required)" not in graph_demo.SYSTEM_PROMPT
