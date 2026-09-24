"""Agent/action trajectory evaluation on real PostgreSQL (committed writes) with a scripted
ORACLE model replaying the golden trajectories, plus adversarial variants. The oracle must
score 1.00 on every metric; each adversary must be caught by the metric it targets."""

from datetime import UTC, datetime, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select

import app.db.session as db_session_module
from app.actions.service import ActionService
from app.agent.action_eval import (
    DEFAULT_ACTION_CASES,
    METRIC_NAMES,
    ActionEvalEnv,
    action_cases_fingerprint,
    action_metrics,
    load_action_cases,
    oracle_script,
    run_action_case,
)
from app.agent.assistant import AssistantLimits
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.knowledge.evaluation import DEFAULT_CASES
from app.models import ActionRequest, Order, StoreCreditTransaction
from scripts.seed_demo import BLUEPEAK, NORTHSTAR, tenant_id_for
from tests.assistant.fakes import make_provider
from tests.db.conftest import fixed_clock
from tests.rag.fakes import FakeRetriever

TENANTS = {t.slug: tenant_id_for(t.slug) for t in (NORTHSTAR, BLUEPEAK)}
CASES = {c.id: c for c in load_action_cases()}


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)

    def __call__(self):
        return self.now


@pytest.fixture
def env(committed):
    clock = Clock()
    svc = ActionService(committed, clock=clock)
    tools = build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session, clock=fixed_clock)
    )
    saver = InMemorySaver()

    def make_assistant(_case, provider, retriever):
        return CommerceGraphAssistant(
            provider,
            tools=tools,
            limits=AssistantLimits(),
            checkpointer=saver,
            retriever=retriever,
            profile=AGENT_PROFILE,
            actions=svc,
        )

    def snapshot(tenant_id):
        with committed() as s:
            orders = dict(
                s.execute(
                    select(Order.order_number, Order.status).where(Order.tenant_id == tenant_id)
                ).all()
            )
            credits = s.scalar(
                select(func.count())
                .select_from(StoreCreditTransaction)
                .where(StoreCreditTransaction.tenant_id == tenant_id)
            )
            actions = s.scalar(
                select(func.count())
                .select_from(ActionRequest)
                .where(ActionRequest.tenant_id == tenant_id)
            )
        return {"orders": orders, "credits": credits, "actions": actions}

    def advance(seconds):
        clock.now = clock.now + timedelta(seconds=seconds)

    return ActionEvalEnv(make_assistant, FakeRetriever(), snapshot, advance, svc, TENANTS)


def oracle(case):
    return make_provider(*oracle_script(case))[0]


def test_fixture_is_separate_and_covers_the_required_scenarios():
    assert DEFAULT_ACTION_CASES != DEFAULT_CASES and len(action_cases_fingerprint()) == 64
    required = {
        "read-order",
        "policy-rag",
        "mixed-shipment-policy",
        "cancel-valid-approved",
        "cancel-shipped-refused",
        "cancel-duplicate",
        "credit-with-evidence-approved",
        "credit-rejected",
        "credit-expired",
        "wrong-tenant-order",
        "execute-without-approval",
        "malformed-write-args",
        "retrieval-failure-before-action",
        "stale-citation-action",
        "injection-tries-to-write",
        "resume-after-restart",
        "credit-without-evidence",
    }
    assert required <= set(CASES)


@pytest.mark.parametrize("case_id", list(CASES))
def test_oracle_trajectory(env, case_id):
    outcome = run_action_case(CASES[case_id], oracle(CASES[case_id]), env)
    assert outcome["passed"], {k: v for k, v in outcome.items() if v is not True}
    assert "answer" not in outcome  # structured outcomes only


def test_oracle_suite_scores_one(env, db_engine):
    from sqlalchemy import text

    with db_engine.connect() as conn:
        statuses = conn.execute(text("SELECT id, status FROM orders")).all()

    def reset():  # every case starts from the pristine seed
        with db_engine.begin() as conn:
            conn.execute(text("DELETE FROM audit_events"))
            conn.execute(text("DELETE FROM agent_runs"))
            conn.execute(text("DELETE FROM store_credit_transactions"))
            conn.execute(text("DELETE FROM action_requests"))
            for order_id, status in statuses:
                conn.execute(
                    text("UPDATE orders SET status=:s WHERE id=:i"), {"s": status, "i": order_id}
                )

    outcomes = []
    for case in CASES.values():
        outcomes.append(run_action_case(case, oracle(case), env))
        reset()
    metrics = action_metrics(outcomes)
    assert set(metrics) == {"cases", *METRIC_NAMES}
    assert all(metrics[m] == 1.0 for m in METRIC_NAMES), metrics


# --- adversaries: the metrics are not vacuous --------------------------------------------------
def test_wrong_order_is_caught_by_the_ordering_metric(env):
    case = CASES["mixed-shipment-policy"]
    swapped = case.model_copy(
        update={
            "turns": [
                case.turns[0].model_copy(
                    update={"reference": list(reversed(case.turns[0].reference))}
                )
            ]
        }
    )
    outcome = run_action_case(case, make_provider(*oracle_script(swapped))[0], env)
    assert outcome["capability_ordering_correct"] is False and outcome["passed"] is False


def test_skipping_the_proposal_is_caught(env):
    case = CASES["cancel-valid-approved"]
    lazy = case.model_copy(
        update={
            "turns": [
                case.turns[0].model_copy(
                    update={"reference": case.turns[0].reference[:1], "expect": "answered"}
                )
            ]
        }
    )
    outcome = run_action_case(case, make_provider(*oracle_script(lazy))[0], env)
    assert outcome["approval_required_detected"] is False
    assert outcome["action_outcome_correct"] is False and outcome["writes_ok"] is False


def test_wrong_amount_is_caught_by_argument_validity(env):
    case = CASES["credit-rejected"]
    turn = case.turns[0]
    bad_ref = [
        turn.reference[0],
        turn.reference[1].model_copy(
            update={"args": {**turn.reference[1].args, "amount": "20.00"}}
        ),
    ]
    wrong = case.model_copy(update={"turns": [turn.model_copy(update={"reference": bad_ref})]})
    outcome = run_action_case(case, make_provider(*oracle_script(wrong))[0], env)
    assert outcome["argument_valid"] is False
