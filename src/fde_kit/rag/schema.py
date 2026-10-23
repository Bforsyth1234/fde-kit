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

Beyond row visibility:

- Documents, chunks, cache and the fact tables are partitioned by tenant, one partition each,
  created by `create_tenant`. Shared indexes make a query's cost depend on other tenants' data,
  which is observable as latency; tests/test_rag_isolation.py measures it.
- Ids are random UUIDs, not shared sequences.
- The app role never logs bound parameters (document text, questions) to the server log.
- `check_posture` also rejects any permissive policy other than tenant_isolation as written
  here, on parents and partitions: permissive policies are OR-ed, so one extra `USING (true)`
  would open every tenant's rows.

`migrate` is idempotent and runs with an admin URL (the compose superuser is fine for that).
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

import psycopg
from psycopg import sql

from ..config import get_settings

EMBED_DIM = 384
OWNER_ROLE = "fde_rag_owner"
APP_ROLE = "fde_rag_app"
TENANT_SETTING = "app.tenant_id"

TENANT_TABLES = (
    "rag_tenants",
    "rag_documents",
    "rag_chunks",
    "rag_cache",
    "rag_entities",
    "rag_extraction_jobs",
    "rag_facts",
    "rag_dictionary",
)
META_TABLE = "rag_meta"

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

-- One row of deployment facts the posture check compares against. Readable by the app role.
CREATE TABLE IF NOT EXISTS rag_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rag_tenants (
    tenant_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Partitioned by tenant, one partition per tenant (created by `create_tenant`). Each tenant's
-- rows, B-trees, GIN and HNSW indexes are physically its own, so the work a query does, and so
-- its latency, can't depend on another tenant's data. Ids are random UUIDs, not a shared
-- sequence, so the gap between two of your ids says nothing about anyone else's inserts.
CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id      UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Provenance (PR 3). A source is versioned: a re-ingest with a different sha256 inserts
    -- version max+1 and points the old row's superseded_by at it. Old rows stay so that an old
    -- citation's doc_id + content_hash still resolves; their chunks do not (see store.py).
    content_hash  TEXT NOT NULL DEFAULT '',
    version       INT NOT NULL DEFAULT 1,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_by UUID,
    body          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant_id, doc_id),
    CONSTRAINT rag_documents_source_version_key UNIQUE (tenant_id, source, version)
) PARTITION BY LIST (tenant_id);

-- Older schemas predate the provenance columns; adding on the parent cascades to every
-- partition. The unique-constraint swap (tenant_id, source) -> (tenant_id, source, version)
-- is done in `migrate`, which looks the old constraint's name up rather than guessing it.
ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;
ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS superseded_by UUID;
ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS body TEXT NOT NULL DEFAULT '';

-- Token count for BM25 length normalisation. IMMUTABLE so it can back a generated column.
CREATE OR REPLACE FUNCTION rag_token_count(content TEXT) RETURNS INT
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT COALESCE(sum(array_length(positions, 1)), 0)::int
    FROM unnest(to_tsvector('english', content))
$$;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id    UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL,
    doc_id      UUID NOT NULL,
    ord         INT NOT NULL,
    content     TEXT NOT NULL,
    -- sha256 of `content` at ingest, computed by the store. A citation carries it, so a reader
    -- can prove what text a Source pointed at even after the document is re-ingested.
    content_hash TEXT NOT NULL DEFAULT '',
    -- Offsets of the chunk body inside rag_documents.body (NULL when not locatable; a
    -- title-prefixed chunk is located by its body after the title line).
    char_start  INT,
    char_end    INT,
    embedding   vector({EMBED_DIM}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    n_tokens    INT GENERATED ALWAYS AS (rag_token_count(content)) STORED,
    PRIMARY KEY (tenant_id, chunk_id),
    -- A chunk can only point at a document of its own tenant. Referential checks run outside
    -- RLS, so this is enforced by the key itself, not by the policy.
    FOREIGN KEY (tenant_id, doc_id)
        REFERENCES rag_documents (tenant_id, doc_id) ON DELETE CASCADE
) PARTITION BY LIST (tenant_id);

CREATE TABLE IF NOT EXISTS rag_cache (
    tenant_id   TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    query_hash  TEXT NOT NULL,
    query       TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    params      JSONB NOT NULL,
    result      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, query_hash)
) PARTITION BY LIST (tenant_id);

-- Older schemas predate content_hash; adding on the parent cascades to every partition.
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS char_start INT;
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS char_end INT;

-- The fact layer (PR 3): harmonised numbers extracted from the documents, with provenance.
-- Composite foreign keys everywhere: a fact can only point at its own tenant's entity, document
-- and chunk, exactly as rag_chunks -> rag_documents. The constraint, not the policy, enforces it.
CREATE TABLE IF NOT EXISTS rag_entities (
    entity_id       UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,                  -- 'company', 'property', 'drug', ...
    canonical_name  TEXT NOT NULL,
    aliases         TEXT[] NOT NULL DEFAULT '{{}}',
    PRIMARY KEY (tenant_id, entity_id),
    UNIQUE (tenant_id, kind, canonical_name)
) PARTITION BY LIST (tenant_id);

CREATE TABLE IF NOT EXISTS rag_extraction_jobs (
    extraction_job_id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,                  -- 'seed', 'adapter:edgar', 'llm'
    model           TEXT NOT NULL DEFAULT '',
    prompt_sha256   TEXT NOT NULL DEFAULT '',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, extraction_job_id)
) PARTITION BY LIST (tenant_id);

CREATE TABLE IF NOT EXISTS rag_facts (
    fact_id         UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    entity_id       UUID NOT NULL,
    field           TEXT NOT NULL,                  -- key into rag_dictionary
    period_start    DATE,
    period_end      DATE,
    value_num       NUMERIC,                        -- exactly one of value_num / value_text
    value_text      TEXT,
    unit            TEXT NOT NULL DEFAULT '',
    doc_id          UUID,                           -- provenance: the document it came from
    chunk_id        UUID,                           -- and the chunk, when known
    conflict_group_id UUID,                         -- NULL = no disagreement on record
    extraction_job_id UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, fact_id),
    FOREIGN KEY (tenant_id, entity_id)
        REFERENCES rag_entities (tenant_id, entity_id) ON DELETE CASCADE,
    -- SET NULL names the column (PG 15+): a plain SET NULL would also null tenant_id, which is
    -- NOT NULL and the partition key, so deleting a cited document would fail instead.
    FOREIGN KEY (tenant_id, doc_id)
        REFERENCES rag_documents (tenant_id, doc_id) ON DELETE SET NULL (doc_id),
    FOREIGN KEY (tenant_id, chunk_id)
        REFERENCES rag_chunks (tenant_id, chunk_id) ON DELETE SET NULL (chunk_id),
    FOREIGN KEY (tenant_id, extraction_job_id)
        REFERENCES rag_extraction_jobs (tenant_id, extraction_job_id)
        ON DELETE SET NULL (extraction_job_id),
    CHECK ((value_num IS NULL) <> (value_text IS NULL))
) PARTITION BY LIST (tenant_id);
CREATE INDEX IF NOT EXISTS rag_facts_lookup_idx
    ON rag_facts (tenant_id, entity_id, field, period_start, period_end);

CREATE TABLE IF NOT EXISTS rag_dictionary (
    tenant_id       TEXT NOT NULL REFERENCES rag_tenants (tenant_id) ON DELETE CASCADE,
    field           TEXT NOT NULL,
    definition      TEXT NOT NULL,
    unit            TEXT NOT NULL DEFAULT '',
    allowed_values  TEXT[] NOT NULL DEFAULT '{{}}',
    notes           TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (tenant_id, field)
) PARTITION BY LIST (tenant_id);

-- Defined on the parents, built per partition.
CREATE INDEX IF NOT EXISTS rag_chunks_doc_idx ON rag_chunks (tenant_id, doc_id);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS rag_chunks_tsv_idx ON rag_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS rag_cache_embedding_idx
    ON rag_cache USING hnsw (embedding vector_cosine_ops);
"""

PARTITIONED_TABLES = (
    "rag_documents",
    "rag_chunks",
    "rag_cache",
    "rag_entities",
    "rag_extraction_jobs",
    "rag_facts",
    "rag_dictionary",
)
# Referencing side first, so partitions can be detached and dropped in this order.
_DROP_ORDER = (
    "rag_facts",
    "rag_dictionary",
    "rag_extraction_jobs",
    "rag_entities",
    "rag_cache",
    "rag_chunks",
    "rag_documents",
)
DOC_SOURCE_VERSION_KEY = "rag_documents_source_version_key"
LOG_PARAM_SETTINGS = ("log_parameter_max_length", "log_parameter_max_length_on_error")
# How Postgres deparses the policy condition; posture compares the live policies against it.
EXPECTED_QUAL = f"(tenant_id = current_setting('{TENANT_SETTING}'::text, true))"


def partition_name(table: str, tenant_id: str, *, key: str | None = None) -> str:
    """Stable, length-safe partition name (tenant ids can be 63 chars; so can identifiers).

    By default a truncated SHA-256 of the tenant id, which anyone holding a list of candidate
    tenant ids can match against names in `pg_class` or the server log. With a deployment key
    (`FDE_RAG_PARTITION_KEY` in settings or `.env`, or `key=`) the name is an HMAC instead, and
    the catalog reveals nothing about who the tenants are.

    The key is part of the schema: `migrate` records a fingerprint of it in `rag_meta`, and
    `check_posture` refuses a connection whose key doesn't match, because under a different
    key every partition lookup would miss and `create_tenant` would try to make a second
    partition for the same tenant id. The key is not rotatable without renaming partitions.
    """
    secret = key if key is not None else get_settings().rag_partition_key
    if secret:
        digest = hmac.new(secret.encode(), tenant_id.encode(), hashlib.sha256).hexdigest()[:16]
    else:
        digest = hashlib.sha256(tenant_id.encode()).hexdigest()[:16]
    return f"{table}_t_{digest}"


def key_fingerprint(key: str) -> str:
    """What gets stored to detect a key mismatch. Not the key, not reversible to it."""
    return hashlib.sha256(b"fde-rag-partition-key:" + key.encode()).hexdigest()[:32]


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
        # Never write bound parameter values to the server log for this role. With the default
        # (-1, unlimited), turning on slow-query logging puts every document's text, every
        # embedding and every user question into the Postgres log, which usually ships to a
        # shared log system. Per-role settings take precedence over postgresql.conf.
        # log_parameter_max_length is superuser-only, so the app can't undo it;
        # log_parameter_max_length_on_error is user-settable, so TenantStore.session also sets
        # it to 0 at the start of every transaction.
        for guc in LOG_PARAM_SETTINGS:
            conn.execute(f"ALTER ROLE {APP_ROLE} SET {guc} = 0")
        old = conn.execute(
            "SELECT relname FROM pg_class WHERE relname = ANY(%s) AND relkind = 'r'",
            (list(PARTITIONED_TABLES),),
        ).fetchall()
        if old:
            raise RuntimeError(
                f"{', '.join(r[0] for r in old)} exist as unpartitioned tables from an older "
                "schema. Drop them (fde_kit.rag.schema.drop_all) and re-ingest."
            )
        with conn.transaction():
            conn.execute(DDL)
            _swap_document_unique(conn)
            for table in TENANT_TABLES:
                conn.execute(f"ALTER TABLE {table} OWNER TO {OWNER_ROLE}")
                conn.execute(_policy_sql(table))
                conn.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}")
            conn.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
            # Creating tenants is an operator action, not something a tenant session can do.
            conn.execute(f"REVOKE INSERT, UPDATE, DELETE ON rag_tenants FROM {APP_ROLE}")
            conn.execute(f"GRANT SELECT ON rag_meta TO {APP_ROLE}")
            fp = key_fingerprint(get_settings().rag_partition_key)
            existing = conn.execute(
                "SELECT value FROM rag_meta WHERE key = 'partition_key_fingerprint'"
            ).fetchone()
            if existing and existing[0] != fp:
                raise RuntimeError(
                    "FDE_RAG_PARTITION_KEY does not match the key this database was migrated "
                    "with; existing partitions were named under the other key"
                )
            conn.execute(
                "INSERT INTO rag_meta (key, value) VALUES ('partition_key_fingerprint', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (fp,),
            )


def _swap_document_unique(conn: psycopg.Connection) -> None:
    """UNIQUE (tenant_id, source) -> UNIQUE (tenant_id, source, version) on rag_documents.

    Schemas from before PR 3 carry the two-column constraint under whatever name Postgres gave
    it; look it up by definition instead of guessing. Fresh schemas get the three-column
    constraint from the DDL, under its explicit name, and this is a no-op."""
    old = conn.execute(
        """
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'rag_documents'::regclass AND contype = 'u'
          AND pg_get_constraintdef(oid) = 'UNIQUE (tenant_id, source)'
        """
    ).fetchall()
    for (name,) in old:
        conn.execute(
            sql.SQL("ALTER TABLE rag_documents DROP CONSTRAINT IF EXISTS {}").format(
                sql.Identifier(name)
            )
        )
    have_new = conn.execute(
        "SELECT 1 FROM pg_constraint WHERE conrelid = 'rag_documents'::regclass AND conname = %s",
        (DOC_SOURCE_VERSION_KEY,),
    ).fetchone()
    if have_new is None:
        conn.execute(
            f"ALTER TABLE rag_documents ADD CONSTRAINT {DOC_SOURCE_VERSION_KEY} "
            "UNIQUE (tenant_id, source, version)"
        )


def create_tenant(admin_url: str, tenant_id: str, name: str = "") -> None:
    """Register a tenant and create its partitions. Idempotent.

    Each partition gets the parent's forced policy and the app role's grants: the store queries
    a tenant's partitions by name (so planning never looks at other tenants), and the policy
    means a session pointed at the wrong partition still sees nothing."""
    validate_tenant_id(tenant_id)
    with psycopg.connect(admin_url, autocommit=True) as conn, conn.transaction():
        conn.execute(
            "INSERT INTO rag_tenants (tenant_id, name) VALUES (%s, %s) "
            "ON CONFLICT (tenant_id) DO UPDATE SET name = EXCLUDED.name",
            (tenant_id, name or tenant_id),
        )
        for table in PARTITIONED_TABLES:
            part = sql.Identifier(partition_name(table, tenant_id))
            conn.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES IN ({})").format(
                    part, sql.Identifier(table), sql.Literal(tenant_id)
                )
            )
            conn.execute(
                sql.SQL("ALTER TABLE {} OWNER TO {}").format(part, sql.Identifier(OWNER_ROLE))
            )
            # Never store the plaintext tenant_id in a table comment: pg_description is global.
            conn.execute(sql.SQL("COMMENT ON TABLE {} IS NULL").format(part))
            name = partition_name(table, tenant_id)  # [a-z0-9_] only
            conn.execute(_policy_sql(name))
            conn.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {name} TO {APP_ROLE}")


def drop_tenant(admin_url: str, tenant_id: str) -> None:
    """Offboard a tenant: drop its partitions (rows, indexes and files go with them) and the
    tenant row. Idempotent. What this does not reach: WAL already written, replicas' WAL
    archives, and backups. See README "Backups and logs"."""
    validate_tenant_id(tenant_id)
    with psycopg.connect(admin_url, autocommit=True) as conn, conn.transaction():
        for table in _DROP_ORDER:
            part = partition_name(table, tenant_id)
            exists = conn.execute("SELECT to_regclass(%s)", (part,)).fetchone()
            if exists and exists[0]:
                conn.execute(
                    sql.SQL("ALTER TABLE {} DETACH PARTITION {}").format(
                        sql.Identifier(table), sql.Identifier(part)
                    )
                )
                conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(part)))
        conn.execute("DELETE FROM rag_tenants WHERE tenant_id = %s", (tenant_id,))


def drop_all(admin_url: str) -> None:
    """Tests only: remove every rag_* table (roles are left in place)."""
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS "
            + ", ".join((*_DROP_ORDER, "rag_tenants", META_TABLE))
            + " CASCADE"
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

    for guc in LOG_PARAM_SETTINGS:
        value = conn.execute("SELECT current_setting(%s)", (guc,)).fetchone()
        if not value or value[0] != "0":
            problems.append(
                PostureProblem(
                    "params_logged",
                    f"{guc} = {value[0] if value else '?'}; the server log can record query "
                    "parameters (document text, questions). Run migrate, or set it to 0.",
                )
            )

    stored = conn.execute(
        "SELECT value FROM rag_meta WHERE key = 'partition_key_fingerprint'"
    ).fetchone()
    if stored is None:
        problems.append(PostureProblem("partition_key", "rag_meta has no key record; run migrate"))
    elif stored[0] != key_fingerprint(get_settings().rag_partition_key):
        problems.append(
            PostureProblem(
                "partition_key",
                "FDE_RAG_PARTITION_KEY differs from the key the schema was migrated with; "
                "partition names would not resolve",
            )
        )

    # Every policy on every RAG table and partition. Permissive policies are OR-ed together, so
    # one extra `USING (true)` policy, added for debugging say, opens every tenant's rows even
    # though tenant_isolation is still there. The only permissive policy allowed is ours,
    # exactly as migrate wrote it; restrictive policies only narrow access and are fine.
    for table, name, permissive, cmd, roles, qual, check in conn.execute(
        """
        SELECT pol.tablename, pol.policyname, pol.permissive, pol.cmd, pol.roles::text[],
               pol.qual, pol.with_check
        FROM pg_policies pol
        WHERE pol.schemaname = 'public' AND (pol.tablename = ANY(%s) OR pol.tablename IN (
            SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent WHERE p.relname = ANY(%s)))
        """,
        (list(TENANT_TABLES), list(PARTITIONED_TABLES)),
    ).fetchall():
        if permissive != "PERMISSIVE":
            continue
        ok = (
            name == "tenant_isolation"
            and cmd == "ALL"
            and roles == ["public"]
            and qual == EXPECTED_QUAL
            and check == EXPECTED_QUAL
        )
        if not ok:
            problems.append(
                PostureProblem(
                    "policy",
                    f"{table} has permissive policy {name} ({cmd}, roles {roles}) with "
                    f"USING {qual} / WITH CHECK {check}; only tenant_isolation as written by "
                    "migrate is allowed",
                )
            )

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
    for (name,) in conn.execute(
        "SELECT relname FROM pg_class WHERE relname = ANY(%s) AND relkind <> 'p'",
        (list(PARTITIONED_TABLES),),
    ).fetchall():
        problems.append(
            PostureProblem(
                "not_partitioned",
                f"{name} is not partitioned by tenant; shared indexes make one tenant's query "
                "cost depend on other tenants' data",
            )
        )
    for part, rls, force, has_policy, acts_as_owner in conn.execute(
        """
        SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
               EXISTS (SELECT 1 FROM pg_policy pol WHERE pol.polrelid = c.oid),
               pg_has_role(current_user, c.relowner, 'USAGE')
        FROM pg_inherits i
        JOIN pg_class c ON c.oid = i.inhrelid JOIN pg_class p ON p.oid = i.inhparent
        WHERE p.relname = ANY(%s)
        """,
        (list(PARTITIONED_TABLES),),
    ).fetchall():
        # The store queries partitions directly, so each one needs its own forced policy.
        if not (rls and force and has_policy):
            problems.append(
                PostureProblem(
                    "partition_rls",
                    f"partition {part} lacks an enabled, forced tenant policy; direct "
                    "queries to it would not be filtered",
                )
            )
        if acts_as_owner:
            problems.append(PostureProblem("owner", f"{user} owns partition {part}"))
    with_policy = {
        r[0]
        for r in conn.execute(
            "SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation' "
            "AND tablename = ANY(%s)",
            (list(TENANT_TABLES),),
        ).fetchall()
    }
    for name, rls, force, acts_as_owner in rows:
        if name not in with_policy:
            problems.append(PostureProblem("policy", f"{name} has no tenant_isolation policy"))
        if not rls:
            problems.append(PostureProblem("rls_disabled", f"{name} has RLS disabled"))
        if not force:
            problems.append(
                PostureProblem("rls_not_forced", f"{name} lacks FORCE ROW LEVEL SECURITY")
            )
        if acts_as_owner:
            problems.append(PostureProblem("owner", f"{user} owns {name} or can act as its owner"))
    return problems
