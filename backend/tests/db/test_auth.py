"""Trusted identity / tenant boundary (Phase 7): Supabase Auth JWT verification with
deterministic local keys (no network, no Supabase secret), server-side memberships, and the
demo-header mode refused in production."""

import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.auth.errors import InvalidTokenError
from app.auth.jwt import JwtVerifier
from app.main import create_app
from tests.conftest import make_settings

SUPABASE = "https://demo-project.supabase.co"
ISS = f"{SUPABASE}/auth/v1"
RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
EC_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
KEYS = {"rsa-1": RSA_KEY.public_key(), "ec-1": EC_KEY.public_key()}
ALICE, BOB, MALLORY = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())


def token(sub=ALICE, *, key=RSA_KEY, alg="RS256", kid="rsa-1", **over):
    now = int(time.time())
    claims = {
        "sub": sub,
        "aud": "authenticated",
        "iss": ISS,
        "role": "authenticated",
        "iat": now,
        "exp": now + 600,
        **over,
    }
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid})


def resolver(tok):
    kid = jwt.get_unverified_header(tok).get("kid")
    if kid not in KEYS:
        raise jwt.PyJWKClientError("unknown kid")
    return KEYS[kid]


def verifier(**kw):
    return JwtVerifier(
        issuer=ISS, audience="authenticated", jwks_url=None, key_resolver=resolver, **kw
    )


# --- verifier ----------------------------------------------------------------------
@pytest.mark.parametrize("args", [{}, {"key": EC_KEY, "alg": "ES256", "kid": "ec-1"}])
def test_valid_asymmetric_tokens(args):
    assert verifier().verify(token(**args))["sub"] == ALICE


@pytest.mark.parametrize(
    "bad",
    [
        {"exp": int(time.time()) - 3600},  # expired
        {"iss": "https://evil.supabase.co/auth/v1"},
        {"aud": "anon"},
        {"role": "anon"},
        {"role": "service_role"},
        {"sub": None},
        {"key": OTHER_KEY},  # wrong signature
        {"kid": "unknown"},
    ],
)
def test_invalid_tokens_are_rejected(bad):
    with pytest.raises(InvalidTokenError):
        verifier().verify(token(**bad))


def test_alg_none_and_hs256_confusion_are_rejected():
    now = int(time.time())
    claims = {
        "sub": ALICE,
        "aud": "authenticated",
        "iss": ISS,
        "role": "authenticated",
        "iat": now,
        "exp": now + 60,
    }
    none_token = jwt.encode(claims, None, algorithm="none")
    # HS256 when no shared secret is configured (algorithm-confusion attempt)
    hs_token = jwt.encode(claims, "x" * 32, algorithm="HS256")
    for bad in (none_token, hs_token, "not-a-jwt"):
        with pytest.raises(InvalidTokenError):
            verifier().verify(bad)


def test_legacy_hs256_only_with_an_explicit_secret():
    secret = "s" * 40
    tok = token(key=secret, alg="HS256", kid="x")
    assert verifier(hs256_secret=secret).verify(tok)["sub"] == ALICE
    with pytest.raises(InvalidTokenError):
        verifier().verify(tok)


def test_jwks_client_path_without_network(monkeypatch):
    """The real PyJWKClient path, with the JWKS document served locally."""
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(RSA_KEY.public_key(), as_dict=True)
    jwk.update({"kid": "rsa-1", "use": "sig", "alg": "RS256"})
    fetched = []
    monkeypatch.setattr(
        jwt.PyJWKClient, "fetch_data", lambda self: fetched.append(self.uri) or {"keys": [jwk]}
    )
    v = JwtVerifier(issuer=ISS, audience="authenticated", jwks_url=f"{ISS}/.well-known/jwks.json")
    assert fetched == []  # construction is network-free
    assert v.verify(token())["sub"] == ALICE
    assert fetched == [f"{ISS}/.well-known/jwks.json"]


# --- API boundary ----------------------------------------------------------------------
@pytest.fixture
def api(committed, tenant_a, tenant_b, db_engine):
    with db_engine.begin() as conn:
        for tenant, sub, role in (
            (tenant_a, ALICE, "approver"),
            (tenant_b, ALICE, "member"),
            (tenant_a, BOB, "member"),
        ):
            conn.execute(
                text(
                    "INSERT INTO tenant_memberships (id, tenant_id, user_subject, role) "
                    "VALUES (gen_random_uuid(), :t, :s, :r)"
                ),
                {"t": tenant.tenant_id, "s": sub, "r": role},
            )
    app = create_app(make_settings(auth_mode="supabase", supabase_url=SUPABASE))
    app.state.jwt_verifier = verifier()
    return TestClient(app)


def h(tok=None, tenant=None):
    headers = {}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if tenant:
        headers["X-Tenant-ID"] = str(tenant.tenant_id)
    return headers


def test_missing_or_bad_credentials(api, tenant_a):
    assert (
        api.get("/api/customers", headers=h(tenant=tenant_a)).json()["error"]["code"]
        == "auth_required"
    )
    r = api.get(
        "/api/customers",
        headers={"Authorization": "Basic abc", "X-Tenant-ID": str(tenant_a.tenant_id)},
    )
    assert r.status_code == 401
    r = api.get("/api/customers", headers=h(token(exp=int(time.time()) - 999), tenant_a))
    assert (r.status_code, r.json()["error"]["code"]) == (401, "auth_invalid")


def test_member_can_use_an_allowed_tenant(api, tenant_a, tenant_b):
    assert api.get("/api/customers", headers=h(token(), tenant_a)).status_code == 200
    assert api.get("/api/customers", headers=h(token(), tenant_b)).status_code == 200


def test_client_tenant_is_never_trusted_without_membership(api, tenant_b):
    r = api.get("/api/customers", headers=h(token(BOB), tenant_b))  # BOB is not a BluePeak member
    assert (r.status_code, r.json()["error"]["code"]) == (403, "tenant_forbidden")
    r = api.get("/api/customers", headers={**h(token(BOB)), "X-Tenant-ID": str(uuid.uuid4())})
    assert r.status_code == 403  # non-existent tenant looks identical


def test_tenant_selection(api):
    assert api.get("/api/customers", headers=h(token(BOB))).status_code == 200  # single membership
    r = api.get("/api/customers", headers=h(token(ALICE)))  # two memberships, none selected
    assert (r.status_code, r.json()["error"]["code"]) == (400, "tenant_selection_required")
    r = api.get("/api/customers", headers=h(token(MALLORY)))  # no membership at all
    assert r.status_code == 403


def test_me_lists_only_memberships(api):
    body = api.get("/api/me", headers=h(token(ALICE))).json()
    assert body["subject"] == ALICE and body["auth_mode"] == "supabase"
    assert {(m["slug"], m["role"]) for m in body["memberships"]} == {
        ("northstar-commerce", "approver"),
        ("bluepeak-retail", "member"),
    }
    assert api.get("/api/me", headers=h(token(MALLORY))).json()["memberships"] == []
    assert api.get("/api/me").status_code == 401


def test_member_role_cannot_approve(api, tenant_a):
    r = api.post(
        f"/api/agent/actions/{uuid.uuid4()}/approve", json={}, headers=h(token(BOB), tenant_a)
    )
    assert (r.status_code, r.json()["error"]["code"]) == (403, "action_forbidden")


def test_demo_header_mode_is_refused_in_production(committed, tenant_a):
    app = create_app(make_settings(app_env="production", auth_mode="demo"))
    r = TestClient(app).get("/api/customers", headers={"X-Tenant-ID": str(tenant_a.tenant_id)})
    assert (r.status_code, r.json()["error"]["code"]) == (503, "auth_not_configured")


def test_threads_are_per_user_inside_a_tenant(api, committed, tenant_a):
    from langgraph.checkpoint.memory import InMemorySaver

    import app.db.session as db_session_module
    from app.actions.service import ActionService
    from app.agent.graph import CommerceGraphAssistant
    from app.agent.graph.profile import AGENT_PROFILE
    from app.agent.tools import ToolDependencies, build_commerce_tools
    from app.api.agent_runtime import AgentRuntime
    from tests.assistant.fakes import ai_text, make_provider
    from tests.rag.fakes import FakeRetriever

    provider, _ = make_provider(ai_text("Hi Alice"), ai_text("Hi Bob"))
    tools = build_commerce_tools(
        ToolDependencies(session_scope=db_session_module.read_only_session)
    )
    svc = ActionService(committed)
    api.app.state.agent_runtime = AgentRuntime(
        CommerceGraphAssistant(
            provider,
            tools=tools,
            checkpointer=InMemorySaver(),
            retriever=FakeRetriever(),
            profile=AGENT_PROFILE,
            actions=svc,
        ),
        svc,
    )
    for who in (ALICE, BOB):
        r = api.post(
            "/api/agent/messages",
            json={"text": f"I am {who}", "thread_id": "shared"},
            headers=h(token(who), tenant_a),
        )
        assert r.status_code == 200, r.text
    bob = api.get("/api/agent/threads/shared/messages", headers=h(token(BOB), tenant_a)).json()[
        "messages"
    ]
    assert [m["content"] for m in bob] == [f"I am {BOB}", "Hi Bob"]


def test_grant_membership_script(committed, tenant_b, capsys):
    from scripts import grant_membership

    assert (
        grant_membership.main(
            ["--tenant", "bluepeak-retail", "--subject", MALLORY, "--role", "approver"]
        )
        == 0
    )
    assert (
        grant_membership.main(
            ["--tenant", "bluepeak-retail", "--subject", MALLORY, "--role", "member"]
        )
        == 0
    )
    assert grant_membership.main(["--tenant", "nope", "--subject", MALLORY]) == 1
    with committed() as s:
        rows = s.execute(
            text("SELECT role FROM tenant_memberships WHERE user_subject=:s"), {"s": MALLORY}
        ).all()
    assert [r[0] for r in rows] == ["member"]
