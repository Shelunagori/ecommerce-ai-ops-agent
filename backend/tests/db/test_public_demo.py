"""Public "Try Live Demo" boundary: a VERIFIED Supabase anonymous JWT (``is_anonymous``
claim) maps to exactly one configured synthetic tenant, read-only. Permanent users keep the
``tenant_memberships`` path. Nothing the browser sends can grant or widen demo access."""

import time
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select, text

import app.db.session as db_session_module
from app.actions.capability import PROPOSE_CANCEL_ORDER
from app.actions.service import ActionService
from app.agent.graph import CommerceGraphAssistant
from app.agent.graph.profile import AGENT_PROFILE, PUBLIC_DEMO_PROFILE
from app.agent.tools import ToolDependencies, build_commerce_tools
from app.api.agent_runtime import AgentRuntime
from app.api.ratelimit import RateLimiter
from app.auth.jwt import JwtVerifier
from app.main import create_app
from app.models import ActionRequest, Order
from tests.assistant.fakes import ai_text, ai_tools, call, make_provider
from tests.conftest import make_settings
from tests.db.test_auth import ISS, OTHER_KEY, RSA_KEY, SUPABASE, resolver
from tests.rag.fakes import FakeRetriever

ANON = str(uuid.uuid4())
ANON_2 = str(uuid.uuid4())
PERMANENT = str(uuid.uuid4())


def token(sub, *, anonymous=True, key=RSA_KEY, **over):
    now = int(time.time())
    claims = {
        "sub": sub,
        "aud": "authenticated",
        "iss": ISS,
        "role": "authenticated",
        "iat": now,
        "exp": now + 600,
        **({"is_anonymous": anonymous} if anonymous is not None else {}),
        **over,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "rsa-1"})


def h(tok=None, tenant=None, **extra):
    headers = {**extra}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if tenant:
        headers["X-Tenant-ID"] = str(getattr(tenant, "tenant_id", tenant))
    return headers


class Harness:
    def __init__(self, committed, **settings):
        base = {
            "auth_mode": "supabase",
            "supabase_url": SUPABASE,
            "public_demo_enabled": True,
            "public_demo_tenant_slug": "bluepeak-retail",
        }
        self.app = create_app(make_settings(**{**base, **settings}))
        self.app.state.jwt_verifier = JwtVerifier(
            issuer=ISS, audience="authenticated", jwks_url=None, key_resolver=resolver
        )
        self.client = TestClient(self.app)
        self.actions = ActionService(committed)
        self.saver = InMemorySaver()
        self.tools = build_commerce_tools(
            ToolDependencies(session_scope=db_session_module.read_only_session)
        )
        self.script()

    def script(self, *steps):
        provider, self.model = make_provider(*steps)
        common = {"tools": self.tools, "checkpointer": self.saver, "retriever": FakeRetriever()}
        full = CommerceGraphAssistant(
            provider, profile=AGENT_PROFILE, actions=self.actions, **common
        )
        demo = CommerceGraphAssistant(provider, profile=PUBLIC_DEMO_PROFILE, **common)
        self.app.state.agent_runtime = AgentRuntime(full, self.actions, public_demo=demo)

    def post(self, headers, text_="hi", thread="t1"):
        return self.client.post(
            "/api/agent/messages", json={"text": text_, "thread_id": thread}, headers=headers
        )


@pytest.fixture
def demo(committed, tenant_a, tenant_b, db_engine):
    with db_engine.begin() as conn:  # permanent user: approver of BOTH tenants
        for tenant in (tenant_a, tenant_b):
            conn.execute(
                text(
                    "INSERT INTO tenant_memberships (id, tenant_id, user_subject, role) "
                    "VALUES (gen_random_uuid(), :t, :s, 'approver')"
                ),
                {"t": tenant.tenant_id, "s": PERMANENT},
            )
    return Harness(committed)


# --- 1/2: verified anonymous JWT -> exactly the configured tenant, role member ---------------
def test_anonymous_sees_only_the_demo_tenant_as_member(demo, tenant_b):
    body = demo.client.get("/api/me", headers=h(token(ANON))).json()
    assert body["public_demo"] is True
    assert [(m["slug"], m["role"]) for m in body["memberships"]] == [("bluepeak-retail", "member")]
    assert body["memberships"][0]["tenant_id"] == str(tenant_b.tenant_id)
    # No membership row was created for the anonymous subject.
    assert demo.client.get("/api/customers", headers=h(token(ANON))).status_code == 200


# --- 3: another tenant can never be selected ------------------------------------------------
def test_anonymous_cannot_select_another_tenant(demo, committed, tenant_a):
    for tenant in (tenant_a, uuid.uuid4()):
        r = demo.client.get("/api/customers", headers=h(token(ANON), tenant))
        assert (r.status_code, r.json()["error"]["code"]) == (403, "tenant_forbidden")
    with committed() as s:  # no membership row is ever created for an anonymous subject
        sql = text("SELECT count(*) FROM tenant_memberships WHERE user_subject = :s")
        assert s.scalar(sql, {"s": ANON}) == 0


# --- 4/5: no approvals, no action endpoints, no write capability ----------------------------
def test_anonymous_cannot_approve_reject_or_list_actions(demo, tenant_b):
    fake_id = uuid.uuid4()
    for method, path in (
        ("post", f"/api/agent/actions/{fake_id}/approve"),
        ("post", f"/api/agent/actions/{fake_id}/reject"),
        ("get", f"/api/agent/actions/{fake_id}"),
        ("get", "/api/agent/actions"),
    ):
        kwargs = {"json": {}} if method == "post" else {}
        r = getattr(demo.client, method)(path, headers=h(token(ANON)), **kwargs)
        assert (r.status_code, r.json()["error"]["code"]) == (403, "public_demo_read_only"), path


def test_anonymous_runs_without_write_capabilities(demo, committed, tenant_b):
    demo.script(ai_text("The public demo is read-only, so I cannot cancel orders."))
    r = demo.post(h(token(ANON)), "Cancel ORD-1004")
    assert r.status_code == 200 and r.json()["action"] is None
    assert PROPOSE_CANCEL_ORDER not in demo.model.bound  # the capability is never offered
    assert r.json()["prompt_version"] == PUBLIC_DEMO_PROFILE.prompt_version
    with committed() as s:
        assert s.scalar(select(func.count()).select_from(ActionRequest)) == 0


def test_a_proposal_from_the_model_cannot_become_a_pending_action(demo, committed, tenant_b):
    # Even if a (misbehaving) model emits the proposal call, it is not a bound capability:
    # it is refused as an unknown tool, nothing is persisted, nothing is written.
    demo.script(
        ai_tools(call(PROPOSE_CANCEL_ORDER, order_number="ORD-1004", reason="x")),
        ai_text("The public demo is read-only."),
    )
    r = demo.post(h(token(ANON)), "Cancel ORD-1004")
    body = r.json()
    assert r.status_code == 200 and body["action"] is None
    assert [(t["tool"], t["outcome"]) for t in body["tool_calls"]] == [
        (PROPOSE_CANCEL_ORDER, "unknown_tool")
    ]
    with committed() as s:
        assert s.scalar(select(func.count()).select_from(ActionRequest)) == 0
        order = select(Order.status).where(
            Order.tenant_id == tenant_b.tenant_id, Order.order_number == "ORD-1004"
        )
        assert s.scalar(order) == "confirmed"  # BluePeak seed status, unchanged


# --- 6: permanent users keep tenant_memberships (and the full HITL capability) ---------------
def test_permanent_user_still_uses_memberships(demo, tenant_a, tenant_b):
    body = demo.client.get("/api/me", headers=h(token(PERMANENT, anonymous=False))).json()
    assert body["public_demo"] is False
    assert {(m["slug"], m["role"]) for m in body["memberships"]} == {
        ("northstar-commerce", "approver"),
        ("bluepeak-retail", "approver"),
    }
    stranger = token(str(uuid.uuid4()), anonymous=False)  # permanent, no membership
    assert demo.client.get("/api/me", headers=h(stranger)).json()["memberships"] == []
    assert demo.client.get("/api/customers", headers=h(stranger)).status_code == 403
    demo.script(ai_tools(call(PROPOSE_CANCEL_ORDER, "p1", order_number="ORD-1004", reason="x")))
    r = demo.post(h(token(PERMANENT, anonymous=False), tenant_a), "Cancel ORD-1004")
    assert r.json()["action"]["status"] == "pending_approval"


def test_missing_is_anonymous_claim_is_a_permanent_user(demo):
    body = demo.client.get("/api/me", headers=h(token(str(uuid.uuid4()), anonymous=None))).json()
    assert body["public_demo"] is False and body["memberships"] == []


# --- 7: forged anonymity never grants access -------------------------------------------------
@pytest.mark.parametrize(
    "forged",
    [
        {"X-Is-Anonymous": "true"},
        {"X-Public-Demo": "true"},
        {"X-Supabase-Is-Anonymous": "1"},
    ],
)
def test_forged_headers_do_not_grant_demo_access(demo, forged):
    stranger = token(str(uuid.uuid4()), anonymous=False)
    body = demo.client.get("/api/me", headers=h(stranger, **forged)).json()
    assert body["public_demo"] is False and body["memberships"] == []
    assert demo.client.get("/api/customers", headers=h(stranger, **forged)).status_code == 403
    assert demo.client.get("/api/me", headers=forged).status_code == 401  # no token at all


def test_forged_body_field_is_rejected(demo):
    r = demo.client.post(
        "/api/agent/messages",
        json={"text": "hi", "thread_id": "t", "is_anonymous": True},
        headers=h(token(str(uuid.uuid4()), anonymous=False)),
    )
    assert r.status_code in (403, 422)  # extra fields are forbidden; no membership anyway


@pytest.mark.parametrize("value", ["true", 1, "yes"])
def test_only_a_boolean_true_claim_is_anonymous(demo, value):
    body = demo.client.get("/api/me", headers=h(token(str(uuid.uuid4()), anonymous=value))).json()
    assert body["public_demo"] is False


def test_unsigned_anonymous_claim_is_rejected(demo):
    bad = token(ANON, key=OTHER_KEY)  # is_anonymous=true but not signed by the project
    assert demo.client.get("/api/me", headers=h(bad)).status_code == 401


# --- 8/13: tenant from the verified principal; demo disabled / misconfigured fails closed ----
def test_demo_disabled_refuses_anonymous_users(committed, tenant_b):
    off = Harness(committed, public_demo_enabled=False)
    r = off.client.get("/api/me", headers=h(token(ANON)))
    assert (r.status_code, r.json()["error"]["code"]) == (403, "public_demo_disabled")
    assert off.client.get("/api/customers", headers=h(token(ANON))).status_code == 403


def test_missing_demo_tenant_fails_safely(committed):
    bad = Harness(committed, public_demo_tenant_slug="no-such-tenant")
    r = bad.client.get("/api/customers", headers=h(token(ANON)))
    assert (r.status_code, r.json()["error"]["code"]) == (503, "public_demo_unavailable")


def test_demo_runtime_without_read_only_assistant_fails_closed(demo, tenant_b):
    runtime = demo.app.state.agent_runtime
    demo.app.state.agent_runtime = AgentRuntime(runtime.assistant, runtime.actions)
    r = demo.post(h(token(ANON)), "hi")
    assert (r.status_code, r.json()["error"]["code"]) == (503, "public_demo_unavailable")


# --- conversation state is per anonymous subject -------------------------------------------
def test_anonymous_visitors_never_share_conversations(demo):
    demo.script(ai_text("hello one"), ai_text("hello two"))
    demo.post(h(token(ANON)), "I am visitor one", thread="same")
    demo.post(h(token(ANON_2)), "I am visitor two", thread="same")
    one = demo.client.get("/api/agent/threads/same/messages", headers=h(token(ANON))).json()
    two = demo.client.get("/api/agent/threads/same/messages", headers=h(token(ANON_2))).json()
    assert [m["content"] for m in one["messages"]] == ["I am visitor one", "hello one"]
    assert [m["content"] for m in two["messages"]] == ["I am visitor two", "hello two"]


# --- abuse protection -----------------------------------------------------------------------
def test_public_demo_has_a_stricter_budget(demo, tenant_b):
    demo.app.state.rate_limiter = RateLimiter(100)
    demo.app.state.public_demo_rate_limiter = RateLimiter(2)
    demo.app.state.public_demo_global_rate_limiter = RateLimiter(3)
    demo.script(*(ai_text(f"a{i}") for i in range(10)))
    assert demo.post(h(token(ANON)), "1").status_code == 200
    assert demo.post(h(token(ANON)), "2").status_code == 200
    r = demo.post(h(token(ANON)), "3")
    assert (r.status_code, r.json()["error"]["code"]) == (429, "rate_limited")
    # A fresh anonymous identity only gets what is left of the global demo budget.
    assert demo.post(h(token(ANON_2)), "4").status_code == 200
    assert demo.post(h(token(ANON_2)), "5").status_code == 429
    # Permanent users are not charged against the public-demo budgets.
    assert demo.post(h(token(PERMANENT, anonymous=False), tenant_b), "6").status_code == 200


def test_public_demo_principal_never_has_approval_or_action_capability():
    """Second layer behind the endpoint refusal: the principal itself carries no approval
    right even if a role were ever mis-assigned."""
    from app.auth.principal import Membership, Principal
    from app.core.tenant import TenantContext

    tenant = uuid.uuid4()
    membership = Membership(tenant, "bluepeak-retail", "BluePeak", "approver")
    p = Principal("anon", "supabase", TenantContext(tenant), "approver", (membership,), True)
    assert (p.can_approve, p.can_use_actions) == (False, False)
    permanent = Principal("u", "supabase", TenantContext(tenant), "approver", (membership,))
    assert (permanent.can_approve, permanent.can_use_actions) == (True, True)


# --- durable per-visitor message budget ---------------------------------------------------------
LIMIT_MESSAGE = "Public demo limit reached. Please start a reviewer session or try again later."


def budgeted(committed, limit):
    harness = Harness(committed, public_demo_message_budget=limit)
    harness.app.state.public_demo_rate_limiter = RateLimiter(1000)
    harness.app.state.public_demo_global_rate_limiter = RateLimiter(1000)
    return harness


def test_anonymous_visitor_has_a_durable_message_budget(committed, tenant_b, demo):
    """(`demo` inserts the permanent reviewer's memberships.)"""
    demo = budgeted(committed, 2)
    demo.script(*(ai_text(f"a{i}") for i in range(10)))
    assert demo.post(h(token(ANON)), "1").status_code == 200
    assert demo.post(h(token(ANON)), "2", thread="t2").status_code == 200  # new thread: same budget
    r = demo.post(h(token(ANON)), "3", thread="t3")
    assert (r.status_code, r.json()["error"]["code"]) == (429, "public_demo_limit_reached")
    assert r.json()["error"]["message"] == LIMIT_MESSAGE
    # the streaming endpoint is refused BEFORE streaming, with the same JSON error
    s = demo.client.post(
        "/api/agent/messages/stream", json={"text": "4", "thread_id": "t4"}, headers=h(token(ANON))
    )
    assert s.status_code == 429 and s.headers["content-type"].startswith("application/json")
    # durable: a brand-new app instance (restart / second replica) still refuses this visitor
    again = budgeted(committed, 2)
    again.script(ai_text("x"))
    assert again.post(h(token(ANON)), "5", thread="t5").status_code == 429
    # another anonymous visitor has their own budget; a reviewer is never budgeted
    assert again.post(h(token(ANON_2)), "6").status_code == 200
    reviewer = demo.post(h(token(PERMANENT, anonymous=False), tenant_b), "7", thread="r")
    assert reviewer.status_code == 200


def test_a_failed_run_gives_its_budget_unit_back(committed):
    from app.agent.llm.errors import LLMUnavailableError

    demo = budgeted(committed, 1)
    demo.script(LLMUnavailableError(), LLMUnavailableError(), ai_text("recovered"))
    failed = demo.post(h(token(ANON)), "1")
    assert failed.status_code == 503  # the model call failed: it does not count
    assert demo.post(h(token(ANON)), "2", thread="t2").status_code == 200
    assert demo.post(h(token(ANON)), "3", thread="t3").status_code == 429


def test_budget_is_keyed_by_the_verified_subject_only(committed):
    demo = budgeted(committed, 1)
    demo.script(*(ai_text(f"a{i}") for i in range(5)))
    assert demo.post(h(token(ANON)), "1").status_code == 200
    forged = h(token(ANON), **{"X-Demo-Budget": "0", "X-Anonymous-Id": str(uuid.uuid4())})
    assert demo.post(forged, "2", thread="t2").status_code == 429
    with committed() as session:
        stored = session.execute(text("SELECT subject_hash, messages FROM public_demo_usage")).all()
    assert [m for _, m in stored] == [1] and ANON not in stored[0][0]  # digest, not the subject


def test_concurrent_reservations_cannot_overshoot_the_limit(committed):
    from concurrent.futures import ThreadPoolExecutor

    from app.auth.demo_budget import PublicDemoBudget

    budget = PublicDemoBudget(committed)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: budget.reserve("visitor-x", 3), range(12)))
    assert results.count(True) == 3 and budget.used("visitor-x") == 3
    budget.release("visitor-x")
    assert budget.used("visitor-x") == 2
    assert budget.reserve("anyone", 0) is True  # 0 = budget disabled


def test_reviewer_accounts_are_never_budgeted(committed, tenant_b, demo):
    harness = budgeted(committed, 1)
    harness.script(*(ai_text(f"r{i}") for i in range(4)))
    reviewer = h(token(PERMANENT, anonymous=False), tenant_b)
    for i in range(3):  # three messages with a budget of one
        assert harness.post(reviewer, str(i), thread=f"r{i}").status_code == 200
    with committed() as session:
        assert session.execute(text("SELECT count(*) FROM public_demo_usage")).scalar() == 0
