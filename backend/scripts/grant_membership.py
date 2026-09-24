"""Grant (or change) a user's access to a tenant: the server-side tenant boundary for
``AUTH_MODE=supabase``.

    uv run python -m scripts.grant_membership --tenant northstar-commerce \\
        --subject <supabase-user-uuid> --role approver

``--subject`` is the Supabase Auth user id (the verified JWT ``sub``). Roles: ``member``
(chat, read) or ``approver`` (may also approve/reject action requests). Idempotent.
"""

import argparse
import sys
import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.session import unit_of_work
from app.models import Tenant, TenantMembership


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grant_membership")
    parser.add_argument("--tenant", required=True, help="tenant slug")
    parser.add_argument("--subject", required=True, help="Supabase user id (JWT sub)")
    parser.add_argument("--role", choices=("member", "approver"), default="member")
    ns = parser.parse_args(argv)
    if not 1 <= len(ns.subject) <= 128:
        parser.error("subject must be 1-128 characters")
    with unit_of_work() as session:
        tenant_id = session.scalar(select(Tenant.id).where(Tenant.slug == ns.tenant))
        if tenant_id is None:
            print("Unknown tenant.", file=sys.stderr)
            return 1
        stmt = insert(TenantMembership).values(
            id=uuid.uuid4(), tenant_id=tenant_id, user_subject=ns.subject, role=ns.role
        )
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["tenant_id", "user_subject"], set_={"role": ns.role}
            )
        )
    print(f"membership granted: {ns.tenant} -> {ns.role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
