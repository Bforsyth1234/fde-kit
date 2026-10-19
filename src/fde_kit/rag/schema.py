"""Schema, roles and Row-Level Security for multi-tenant retrieval.

Isolation is enforced by Postgres, not by application code. Every tenant-scoped table has RLS
enabled AND forced, with one policy: a row is visible (and writable) only when its `tenant_id`
equals the transaction-local setting `app.tenant_id`. If nothing set it, the comparison is
against '' or NULL and matches nothing, so a forgotten tenant fails closed: zero rows, and
inserts are rejected by WITH CHECK.

RLS has three well-known bypasses, and each is a configuration mistake rather than a bug:

1. Superusers ignore RLS entirely, FORCE or not. The default `docker compose` user is one.
2. Roles with BYPASSRLS ignore it.
3. A table's owner ignores it unless the table has FORCE ROW LEVEL SECURITY.

So there are two roles. `fde_rag_owner` (NOLOGIN) owns the tables; `fde_rag_app` (LOGIN) is what
the application connects as, with DML grants and nothing else. `check_posture` verifies all of
this on the live connection, and `TenantStore` refuses to start if any check fails. A store that
silently runs as a superuser would pass every functional test and isolate nothing.

`migrate` is idempotent and runs with an admin URL (the compose superuser is fine for that).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg
from psycopg import sql

EMBED_DIM = 384
OWNER_ROLE = "fde_rag_owner"
APP_ROLE = "fde_rag_app"
TENANT_SETTING = "app.tenant_id"

TENANT_TABLES = ("rag_tenants", "rag_documents", "rag_chunks", "rag_cache")

_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def validate_tenant_id(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not _TENANT_ID.match(tenant_id):
        raise ValueError(
            f"invalid tenant id {tenant_id!r}: lowercase letters, digits, '-' and '_', "
            "1-63 chars, starting with a letter or digit"
        )
    return tenant_id


DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_tenants (
    tenant_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id      BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source),
    -- Target for the composite foreign key below.
    UNIQUE (tenant_id, doc_id)
);

-- Token count for BM25 length normalisation. IMMUTABLE so it can back a generated column.
CREATE OR REPLACE FUNCTION rag_token_count(content TEXT) RETURNS INT
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT COALESCE(sum(array_length(positions, 1)), 0)::int
    FROM unnest(to_tsvector('english', content))
$$;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id    BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    doc_id      BIGINT NOT NULL,
    ord         INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    n_tokens    INT GENERATED ALWAYS AS (rag_token_count(content)) STORED,
    -- A chunk can only point at a document of its own tenant. Referential checks run outside
    -- RLS, so this is enforced by the key itself, not by the policy.
    FOREIGN KEY (tenant_id, doc_id)
        REFERENCES rag_documents (tenant_id, doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rag_cache (
    tenant_id   TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    query_hash  TEXT NOT NULL,
    query       TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    params      JSONB NOT NULL,
    result      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, query_hash)
);

CREATE INDEX IF NOT EXISTS rag_chunks_tenant_idx ON rag_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS rag_chunks_tsv_idx ON rag_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS rag_cache_embedding_idx
    ON rag_cache USING hnsw (embedding vector_cosine_ops);
"""


def _policy_sql(table: str) -> str:
    # current_setting(..., true) returns NULL instead of raising when the setting was never
    # defined, and '' after a transaction-local set has ended. Neither matches a real tenant id.
    return f"""
ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
ALTER TABLE {table} FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON {table};
CREATE POLICY tenant_isolation ON {table}
    USING (tenant_id = current_setting('{TENANT_SETTING}', true))
    WITH CHECK (tenant_id = current_setting('{TENANT_SETTING}', true));
"""


def migrate(admin_url: str, app_password: str = "fde_rag_app") -> None:
    """Create roles, tables, indexes and policies. Safe to run repeatedly."""
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{OWNER_ROLE}') THEN
        CREATE ROLE {OWNER_ROLE} NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
        CREATE ROLE {APP_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT;
    END IF;
END $$;
"""
        )
        conn.execute(
            sql.SQL("ALTER ROLE {} WITH LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}").format(
                sql.Identifier(APP_ROLE), sql.Literal(app_password)
            )
        )
        with conn.transaction():
            conn.execute(DDL)
            for table in TENANT_TABLES:
                conn.execute(f"ALTER TABLE {table} OWNER TO {OWNER_ROLE}")
                conn.execute(_policy_sql(table))
                conn.execute(
                    f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}"
                )
            conn.execute(
                f"GRANT USAGE ON SCHEMA public TO {APP_ROLE};"
                f"GRANT USAGE, SELECT ON SEQUENCE rag_documents_doc_id_seq, "
                f"rag_chunks_chunk_id_seq TO {APP_ROLE};"
            )
            # Creating tenants is an operator action, not something a tenant session can do.
            conn.execute(f"REVOKE INSERT, UPDATE, DELETE ON rag_tenants FROM {APP_ROLE}")


def create_tenant(admin_url: str, tenant_id: str, name: str = "") -> None:
    validate_tenant_id(tenant_id)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO rag_tenants (tenant_id, name) VALUES (%s, %s) "
            "ON CONFLICT (tenant_id) DO UPDATE SET name = EXCLUDED.name",
            (tenant_id, name or tenant_id),
        )


def drop_all(admin_url: str) -> None:
    """Tests only: remove every rag_* table (roles are left in place)."""
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS rag_cache, rag_chunks, rag_documents, rag_tenants CASCADE"
        )


def app_url_from_admin(admin_url: str, app_password: str = "fde_rag_app") -> str:
    """Same host and database as `admin_url`, connecting as the app role."""
    info = psycopg.conninfo.conninfo_to_dict(admin_url)
    info["user"] = APP_ROLE
    info["password"] = app_password
    return psycopg.conninfo.make_conninfo(**info)


# --------------------------------------------------------------------------- posture


@dataclass
class PostureProblem:
    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


class InsecureConnection(RuntimeError):
    """The connection would let RLS be bypassed. Raised by TenantStore at startup."""

    def __init__(self, problems: list[PostureProblem]):
        self.problems = problems
        super().__init__(
            "refusing to serve tenants on this connection:\n  "
            + "\n  ".join(str(p) for p in problems)
        )


def check_posture(conn: psycopg.Connection) -> list[PostureProblem]:
    """Every way this connection could see across tenants. Empty list means none found."""
    problems: list[PostureProblem] = []
    row = conn.execute(
        "SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    assert row is not None
    user, is_super, bypass = row
    if is_super:
        problems.append(
            PostureProblem("superuser", f"connected as {user}, a superuser; RLS does not apply")
        )
    if bypass:
        problems.append(PostureProblem("bypassrls", f"role {user} has BYPASSRLS"))

    rows = conn.execute(
        """
        SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
               pg_has_role(current_user, c.relowner, 'USAGE') AS acts_as_owner
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = ANY(%s)
        """,
        (list(TENANT_TABLES),),
    ).fetchall()
    found = {r[0] for r in rows}
    for missing in sorted(set(TENANT_TABLES) - found):
        problems.append(PostureProblem("missing_table", f"{missing} not found; run migrate"))
    for name, rls, force, acts_as_owner in rows:
        if not rls:
            problems.append(PostureProblem("rls_disabled", f"{name} has RLS disabled"))
        if not force:
            problems.append(
                PostureProblem("rls_not_forced", f"{name} lacks FORCE ROW LEVEL SECURITY")
            )
        if acts_as_owner:
            problems.append(
                PostureProblem("owner", f"{user} owns {name} or can act as its owner")
            )
    return problems
