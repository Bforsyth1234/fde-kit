"""Provenance, facts and read-only SQL (PR 3), against a real Postgres.

Same gate and fixtures as tests/test_rag_isolation.py: FDE_TEST_PG_URL must point at an admin
URL on a server with pgvector >= 0.8, or everything here skips. The schema is recreated from
scratch for the module and seeded with the demo lease corpus; the entities, facts and
dictionary rows are inserted by the tests themselves (demo_corpus grows them in 1-C).
"""

from __future__ import annotations

import os
from datetime import date

import psycopg
import pytest

from fde_kit.rag import (
    DictEntry,
    FactRow,
    HashEmbedder,
    SqlRejected,
    TenantStore,
    app_url_from_admin,
    check_posture,
    create_tenant,
    migrate,
)
from fde_kit.rag.demo_corpus import TENANTS, seed
from fde_kit.rag.schema import (
    DOC_SOURCE_VERSION_KEY,
    PARTITIONED_TABLES,
    TENANT_TABLES,
    drop_all,
    drop_tenant,
    partition_name,
)
from fde_kit.rag.store import content_hash

PG_URL = os.environ.get("FDE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="FDE_TEST_PG_URL not set")

NEW_TABLES = ("rag_entities", "rag_extraction_jobs", "rag_facts", "rag_dictionary")
HARBOR = TENANTS["harbor"]["name"]


@pytest.fixture(scope="module")
def admin_url() -> str:
    assert PG_URL
    with psycopg.connect(PG_URL) as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error as exc:
            pytest.skip(f"pgvector unavailable: {exc}")
        ver = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    if ver and tuple(int(x) for x in ver[0].split(".")[:2]) < (0, 8):
        pytest.skip(f"pgvector {ver[0]} < 0.8 (no iterative index scans)")
    drop_all(PG_URL)
    migrate(PG_URL)
    for tid, t in TENANTS.items():
        create_tenant(PG_URL, tid, t["name"])
    return PG_URL


@pytest.fixture(scope="module")
def app_url(admin_url: str) -> str:
    return app_url_from_admin(admin_url)


@pytest.fixture(scope="module")
def store(app_url: str):
    with TenantStore(app_url, HashEmbedder()) as s:
        seed(s)
        yield s


@pytest.fixture(scope="module")
def entities(store: TenantStore) -> dict[str, str]:
    """One property_manager entity per tenant, aliased by tenant id and a short name."""
    ids: dict[str, str] = {}
    for tid, t in TENANTS.items():
        with store.session(tid) as s:
            ids[tid] = s.upsert_entity("property_manager", t["name"], [tid, t["name"].split()[0]])
    return ids


# ------------------------------------------------------------------ schema


def test_new_tables_are_partitioned_with_a_partition_per_tenant(admin_url: str, store):
    assert set(NEW_TABLES) <= set(TENANT_TABLES) and set(NEW_TABLES) <= set(PARTITIONED_TABLES)
    with psycopg.connect(admin_url) as conn:
        kinds = dict(
            conn.execute(
                "SELECT relname, relkind FROM pg_class WHERE relname = ANY(%s)", (list(NEW_TABLES),)
            ).fetchall()
        )
        assert kinds == {t: "p" for t in NEW_TABLES}
        for tid in TENANTS:
            for table in NEW_TABLES:
                part = partition_name(table, tid)
                assert conn.execute("SELECT to_regclass(%s)", (part,)).fetchone() != (None,)
        cols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'rag_documents'"
            ).fetchall()
        }
        assert {"content_hash", "version", "fetched_at", "superseded_by", "body"} <= cols
        chunk_cols = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'rag_chunks'"
            ).fetchall()
        }
        assert {"char_start", "char_end"} <= chunk_cols
        defs = [
            r[0]
            for r in conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'rag_documents'::regclass AND contype = 'u'"
            ).fetchall()
        ]
        assert defs == ["UNIQUE (tenant_id, source, version)"]


def test_app_role_posture_is_clean_with_the_new_tables(app_url: str, store):
    with psycopg.connect(app_url) as conn:
        assert check_posture(conn) == []


def test_migrate_is_idempotent_and_swaps_a_legacy_unique_constraint(admin_url: str, store):
    """A pre-PR-3 database has UNIQUE (tenant_id, source) under a generated name. migrate
    finds it by definition, drops it, and adds the versioned one; running again is a no-op."""
    migrate(admin_url)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f"ALTER TABLE rag_documents DROP CONSTRAINT {DOC_SOURCE_VERSION_KEY}")
        conn.execute("ALTER TABLE rag_documents ADD UNIQUE (tenant_id, source)")
        legacy = conn.execute(
            "SELECT conname FROM pg_constraint WHERE conrelid = 'rag_documents'::regclass "
            "AND contype = 'u'"
        ).fetchall()
        assert len(legacy) == 1 and legacy[0][0] != DOC_SOURCE_VERSION_KEY
    migrate(admin_url)
    with psycopg.connect(admin_url) as conn:
        rows = conn.execute(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'rag_documents'::regclass AND contype = 'u'"
        ).fetchall()
    assert rows == [(DOC_SOURCE_VERSION_KEY, "UNIQUE (tenant_id, source, version)")]
    with psycopg.connect(app_url_from_admin(admin_url)) as conn:
        assert check_posture(conn) == []


def test_drop_tenant_removes_all_seven_partitions(admin_url: str, store: TenantStore):
    create_tenant(admin_url, "leaver")
    with store.session("leaver") as s:
        doc = s.ingest("leaver/contract.md", "CONTRACT\n\nConfidential terms.")
        eid = s.upsert_entity("company", "Leaver Inc", ["leaver"])
        s.add_facts([FactRow("leaver", "late_fee", 10, "USD", doc_id=doc)])
        s.set_dictionary([DictEntry("late_fee", "Late fee", "USD")])
        assert eid
    drop_tenant(admin_url, "leaver")
    drop_tenant(admin_url, "leaver")
    with psycopg.connect(admin_url) as conn:
        for table in PARTITIONED_TABLES:
            gone = conn.execute("SELECT to_regclass(%s)", (partition_name(table, "leaver"),))
            assert gone.fetchone() == (None,), table
        assert conn.execute(
            "SELECT count(*) FROM rag_tenants WHERE tenant_id = 'leaver'"
        ).fetchone() == (0,)


# ------------------------------------------------------------------ versioning


V1 = "NOTICE\n\nThe zebrafish clause: pay by the fifth.\n\nSecond paragraph of the first draft."
V2 = "NOTICE\n\nThe quokka clause: pay by the third.\n\nSecond paragraph of the second draft."
SRC = "harbor/versioned.md"


def _doc_ids(store: TenantStore, source: str) -> list[tuple[str, int, str | None]]:
    with store.session("harbor") as s:
        return s.conn.execute(
            "SELECT doc_id::text, version, superseded_by::text FROM rag_documents "
            "WHERE source = %s ORDER BY version",
            (source,),
        ).fetchall()


def test_ingest_is_idempotent_by_hash(store: TenantStore):
    with store.session("harbor") as s:
        first = s.ingest_versioned(SRC, V1, title="Versioned")
        assert first.changed and first.version == 1 and first.content_hash == content_hash(V1)
        again = s.ingest(SRC, V1, title="Versioned")
        assert again == first.doc_id
        assert s.last_ingest is not None and not s.last_ingest.changed
        assert s.last_ingest.doc_id == first.doc_id and s.last_ingest.version == 1
        assert s.last_ingest.content_hash == first.content_hash
    assert len(_doc_ids(store, SRC)) == 1


def test_changed_text_makes_a_new_version_and_retires_the_old_chunks(store: TenantStore):
    with store.session("harbor") as s:
        old = s.ingest_versioned(SRC, V1, title="Versioned")
        assert not old.changed
        assert any(h.doc_id == old.doc_id for h in s.search("zebrafish", k=3, mode="keyword"))
        s.cached_search("zebrafish")  # warm the cache so we can see it invalidated

        new = s.ingest_versioned(SRC, V2, title="Versioned")
        assert new.changed and new.version == 2 and new.doc_id != old.doc_id
        assert new.content_hash == content_hash(V2)

        rows = s.conn.execute(
            "SELECT doc_id::text, version, superseded_by::text, body, content_hash "
            "FROM rag_documents WHERE source = %s ORDER BY version",
            (SRC,),
        ).fetchall()
        assert [(r[0], r[1], r[2]) for r in rows] == [
            (old.doc_id, 1, new.doc_id),
            (new.doc_id, 2, None),
        ]
        assert rows[0][3] == V1 and rows[1][3] == V2  # old body kept as provenance

        n_old = s.conn.execute(
            "SELECT count(*) FROM rag_chunks WHERE doc_id = %s", (old.doc_id,)
        ).fetchone()
        assert n_old == (0,)
        n_new = s.conn.execute(
            "SELECT count(*) FROM rag_chunks WHERE doc_id = %s", (new.doc_id,)
        ).fetchone()
        assert n_new and n_new[0] > 0

        hits = s.search("zebrafish", k=5, mode="keyword")
        assert all(h.doc_id != old.doc_id for h in hits)
        hits = s.search("quokka", k=3, mode="keyword")
        assert hits and hits[0].doc_id == new.doc_id and hits[0].content_hash
        assert s.conn.execute("SELECT count(*) FROM rag_cache").fetchone() == (0,)

        # A third identical ingest after the change is a no-op on version 2.
        again = s.ingest_versioned(SRC, V2, title="Versioned")
        assert (again.doc_id, again.version, again.changed) == (new.doc_id, 2, False)


def test_chunks_carry_offsets_into_the_body(store: TenantStore):
    with store.session("harbor") as s:
        rows = s.conn.execute(
            "SELECT c.content, c.char_start, c.char_end, d.body FROM rag_chunks c "
            "JOIN rag_documents d USING (doc_id) WHERE d.source = %s ORDER BY c.ord",
            (SRC,),
        ).fetchall()
        assert rows
        for content, start, end, body in rows:
            assert start is not None and end is not None and start < end
            assert content.endswith(body[start:end])
            assert content.startswith("Versioned\n\n")
        # The demo corpus (title-prefixed, template leases) locates every chunk.
        unlocated = s.conn.execute(
            "SELECT count(*) FROM rag_chunks WHERE char_start IS NULL"
        ).fetchone()
        assert unlocated == (0,)


def test_fetch_slices_and_reports_supersession(store: TenantStore):
    (old_id, _, _), (new_id, _, _) = _doc_ids(store, SRC)
    with store.session("harbor") as s:
        old = s.fetch(old_id)
        assert old.text == V1 and old.version == 1 and old.superseded_by == new_id
        assert old.content_hash == content_hash(V1) and old.source == SRC
        assert old.char_start == 0 and old.char_end == len(V1) and old.next is None
        assert old.total_chars == len(V1) and old.title == "Versioned"

        head = s.fetch(new_id, 0, 10)
        assert head.text == V2[:10] and head.next == 10 and head.superseded_by is None
        assert head.version == 2 and head.total_chars == len(V2)
        rest = s.fetch(new_id, head.next)
        assert rest.text == V2[10:] and rest.next is None and rest.char_end == len(V2)

        # Clamped, never raises for a range outside the body.
        tail = s.fetch(new_id, 10_000, 20_000)
        assert tail.text == "" and tail.char_start == tail.char_end == len(V2)
        assert tail.next is None
        inverted = s.fetch(new_id, 20, 5)
        assert inverted.char_start == 20 and inverted.char_end == 20 and inverted.text == ""

        with pytest.raises(KeyError):
            s.fetch("00000000-0000-0000-0000-000000000000")
        with pytest.raises(KeyError):
            s.fetch("not-a-uuid")


def test_deleting_a_document_cascades_to_chunks_and_nulls_fact_provenance(
    store: TenantStore, entities: dict[str, str]
):
    """ON DELETE SET NULL (doc_id): only the provenance column is cleared, not tenant_id."""
    (_, _, _), (new_id, _, _) = _doc_ids(store, SRC)
    with store.session("harbor") as s:
        chunk = s.conn.execute(
            "SELECT chunk_id::text FROM rag_chunks WHERE doc_id = %s LIMIT 1", (new_id,)
        ).fetchone()
        assert chunk
        (fid,) = s.add_facts(
            [FactRow("harbor", "notice_days", 60, "days", doc_id=new_id, chunk_id=chunk[0])]
        )
        fact = {f.fact_id: f for f in s.facts(field="notice_days")}[fid]
        assert fact.doc_id == new_id and fact.chunk_id == chunk[0] and fact.doc_source == SRC
        s.conn.execute("DELETE FROM rag_documents WHERE source = %s", (SRC,))
        fact = {f.fact_id: f for f in s.facts(field="notice_days")}[fid]
        assert fact.doc_id is None and fact.chunk_id is None and fact.doc_source is None
        assert fact.value == 60
        s.conn.execute("DELETE FROM rag_facts WHERE fact_id = %s::uuid", (fid,))
    assert _doc_ids(store, SRC) == []


# ------------------------------------------------------------------ entities


def test_entities_resolve_by_canonical_name_or_alias_case_insensitively(
    store: TenantStore, entities: dict[str, str]
):
    with store.session("harbor") as s:
        eid = entities["harbor"]
        assert s.resolve_entity(HARBOR) == eid
        assert s.resolve_entity(HARBOR.upper()) == eid
        assert s.resolve_entity("harbor") == eid
        assert s.resolve_entity("HARBOR") == eid
        assert s.resolve_entity("Harbor") == eid  # the short-name alias
        assert s.resolve_entity("summit") is None  # another tenant's alias: unknown here
        assert s.resolve_entity("nobody") is None

        # Upsert by (kind, canonical_name) keeps the id and replaces the aliases.
        same = s.upsert_entity("property_manager", HARBOR, ["hpm"])
        assert same == eid
        assert s.resolve_entity("hpm") == eid and s.resolve_entity("harbor") is None
        s.upsert_entity("property_manager", HARBOR, ["harbor", "Harbor"])
        assert s.resolve_entity("harbor") == eid

        # Same name, different kind is a different entity.
        other = s.upsert_entity("building", HARBOR)
        assert other != eid
        s.conn.execute("DELETE FROM rag_entities WHERE entity_id = %s", (other,))


def test_add_facts_rejects_unknown_entities_and_bad_values(store: TenantStore, entities):
    with store.session("harbor") as s:
        with pytest.raises(ValueError, match="unknown entity"):
            s.add_facts([FactRow("Nobody Ltd", "late_fee", 1)])
        with pytest.raises(ValueError):
            s.add_facts([FactRow("harbor", "late_fee", None)])  # type: ignore[arg-type]
        assert s.facts(entity="Nobody Ltd") == []


# ------------------------------------------------------------------ facts and conflicts


def _clear_facts(s) -> None:
    """`demo_corpus.seed` loads 18 facts and a dictionary per tenant. Tests that assert exact
    contents start from a clean slate; inside a tenant session the policy is the WHERE clause."""
    s.conn.execute("DELETE FROM rag_facts")
    s.conn.execute("DELETE FROM rag_dictionary")


def test_add_facts_conflict_rule(store: TenantStore, entities: dict[str, str]):
    period = {"period_start": date(2026, 1, 1), "period_end": None}
    with store.session("harbor") as s:
        _clear_facts(s)
        lease = s.conn.execute(
            "SELECT doc_id::text FROM rag_documents WHERE source LIKE 'harbor/leases/%' "
            "ORDER BY source LIMIT 1"
        ).fetchone()
        assert lease
        job = s.start_extraction_job("seed", model="none", prompt_sha256="abc")

        # Different value for the same (entity, field, period): one shared group.
        (a,) = s.add_facts(
            [FactRow("harbor", "late_fee", 75, "USD", doc_id=lease[0], **period)],
            extraction_job_id=job,
        )
        assert [f.conflict_group_id for f in s.facts(field="late_fee")] == [None]
        (b,) = s.add_facts([FactRow("harbor", "late_fee", 85, "USD", **period)])
        groups = {f.fact_id: f.conflict_group_id for f in s.facts(field="late_fee")}
        assert groups[a] is not None and groups[a] == groups[b]
        group = groups[a]

        # A third, different value joins the existing group rather than making a new one.
        (c,) = s.add_facts([FactRow("harbor", "late_fee", 95, "USD", **period)])
        groups = {f.fact_id: f.conflict_group_id for f in s.facts(field="late_fee")}
        assert groups == {a: group, b: group, c: group}

        # Same value, same key: a second row, but no conflict group.
        (g1,) = s.add_facts([FactRow("harbor", "grace_days", 5, "days", **period)])
        (g2,) = s.add_facts([FactRow("harbor", "grace_days", 5, "days", **period)])
        grace = s.facts(entity=HARBOR, field="grace_days")
        assert {f.fact_id for f in grace} == {g1, g2}
        assert [f.conflict_group_id for f in grace] == [None, None]
        assert [f.value for f in grace] == [5, 5]

        # A different period is a different key: no conflict with the 2026 rows.
        (d,) = s.add_facts(
            [FactRow("harbor", "late_fee", 65, "USD", period_start=date(2025, 1, 1))]
        )
        by_id = {f.fact_id: f for f in s.facts(field="late_fee")}
        assert by_id[d].conflict_group_id is None
        assert by_id[d].period_start == date(2025, 1, 1) and by_id[d].period_end is None

        # Text values work the same way, and CHECK keeps exactly one of num/text.
        (p1,) = s.add_facts([FactRow("harbor", "pet_deposit", "not applicable")])
        (p2,) = s.add_facts([FactRow("harbor", "pet_deposit", "waived")])
        pets = {f.fact_id: f for f in s.facts(field="pet_deposit")}
        assert pets[p1].value == "not applicable" and pets[p2].value == "waived"
        assert pets[p1].conflict_group_id == pets[p2].conflict_group_id is not None
        with pytest.raises(psycopg.errors.CheckViolation), s.conn.transaction():
            s.conn.execute(
                "INSERT INTO rag_facts (tenant_id, entity_id, field) VALUES (%s, %s, 'x')",
                ("harbor", entities["harbor"]),
            )

        # Output shape and provenance.
        fa = by_id[a]
        assert fa.entity == HARBOR and fa.entity_id == entities["harbor"]
        assert fa.value == 75 and isinstance(fa.value, int) and fa.unit == "USD"
        assert fa.doc_id == lease[0] and fa.doc_source and fa.doc_source.startswith("harbor/")
        assert fa.doc_hash and len(fa.doc_hash) == 64
        assert fa.extraction_job_id == job and fa.chunk_id is None
        src = fa.to_source("harbor")
        assert src.source_id == a and src.uri == fa.doc_source and src.doc_id == lease[0]
        assert src.content_hash == fa.doc_hash and src.span == f"fact:{a}"
        assert src.title == f"{HARBOR} late_fee" and src.tenant_id == "harbor"
        fb = by_id[b]
        assert fb.doc_id is None and fb.doc_source is None and fb.doc_hash is None
        assert fb.to_source().uri == f"fact:{b}"

        # Filters: by alias, by field, both; unknown entity is empty.
        assert {f.fact_id for f in s.facts(entity="harbor")} >= {a, b, c, d, g1, g2, p1, p2}
        assert {f.fact_id for f in s.facts(entity="HARBOR", field="late_fee")} == {a, b, c, d}
        assert s.facts(entity="summit") == []


def test_facts_cannot_point_at_another_tenants_document(store: TenantStore, entities):
    """The composite FK, not the policy, keeps a fact's provenance inside its tenant."""
    with store.session("summit") as s:
        summit_doc = s.conn.execute("SELECT doc_id::text FROM rag_documents LIMIT 1").fetchone()
    assert summit_doc
    with pytest.raises(psycopg.errors.ForeignKeyViolation), store.session("harbor") as s:
        s.add_facts([FactRow("harbor", "late_fee", 1, doc_id=summit_doc[0])])
    with store.session("harbor") as s:
        assert all(f.value != 1 for f in s.facts(field="late_fee"))


# ------------------------------------------------------------------ dictionary


def test_dictionary_upserts_by_field(store: TenantStore):
    with store.session("harbor") as s:
        s.conn.execute("DELETE FROM rag_dictionary")  # seed loaded one; assert exact contents
        s.set_dictionary(
            [
                DictEntry("late_fee", "Fee charged after the grace period", "USD"),
                DictEntry("grace_days", "Days after the due date before a late fee", "days"),
            ]
        )
        s.set_dictionary(
            [
                DictEntry(
                    "late_fee",
                    "Late payment fee",
                    "USD",
                    allowed_values=("flat", "percent"),
                    notes="revised",
                )
            ]
        )
        s.set_dictionary([])
        got = s.dictionary()
    assert got == [
        DictEntry("grace_days", "Days after the due date before a late fee", "days"),
        DictEntry("late_fee", "Late payment fee", "USD", ("flat", "percent"), "revised"),
    ]
    with store.session("summit") as s:
        # Dictionaries are per tenant: summit keeps its seeded entries, never harbor's edits.
        summit = {e.field: e for e in s.dictionary()}
        assert summit and summit["late_fee"].notes != "revised"


# ------------------------------------------------------------------ query()


def test_query_returns_columns_rows_and_truncation(store: TenantStore, entities):
    with store.session("harbor") as s:
        r = s.query("SELECT g AS n, g * 2 AS twice FROM generate_series(1, 10) g ORDER BY g")
        assert r.columns == ["n", "twice"] and len(r.rows) == 10 and not r.truncated
        assert r.rows[0] == (1, 2) and r.elapsed_ms >= 0

        r = s.query("SELECT g FROM generate_series(1, 10) g ORDER BY g", max_rows=3)
        assert r.rows == [(1,), (2,), (3,)] and r.truncated

        r = s.query("SELECT g FROM generate_series(1, 3) g", max_rows=3)
        assert len(r.rows) == 3 and not r.truncated

        # LIKE patterns with '%' are not mistaken for parameter placeholders.
        r = s.query("SELECT count(*) FROM rag_documents WHERE source LIKE 'harbor/leases/%'")
        assert r.rows[0][0] > 0

        # Facts come back with plain Python values, uuids as strings, numerics as numbers.
        r = s.query(
            "SELECT f.fact_id, e.canonical_name, f.value_num, f.conflict_group_id "
            "FROM rag_facts f JOIN rag_entities e ON e.entity_id = f.entity_id "
            "WHERE f.field = 'late_fee' ORDER BY f.value_num"
        )
        assert r.columns == ["fact_id", "canonical_name", "value_num", "conflict_group_id"]
        assert all(isinstance(row[0], str) and len(row[0]) == 36 for row in r.rows)
        assert [row[2] for row in r.rows] == [65, 75, 85, 95]
        assert isinstance(r.rows[0][2], int)
        # SELECT * through a join with duplicate column names still works.
        r = s.query("SELECT * FROM rag_facts f JOIN rag_entities e USING (tenant_id, entity_id)")
        assert "canonical_name" in r.columns and r.rows


def test_query_rejections_do_not_poison_the_session(store: TenantStore):
    with store.session("harbor") as s:
        with pytest.raises(SqlRejected) as exc:
            s.query("DELETE FROM rag_facts")
        assert "SELECT" in exc.value.reason

        with pytest.raises(SqlRejected) as exc:
            s.query("SELECT no_such_column FROM rag_facts")
        assert "no_such_column" in exc.value.reason and "\n" not in exc.value.reason

        with pytest.raises(SqlRejected):
            s.query("SELECT * FROM no_such_table")

        # The session's transaction is still usable afterwards.
        assert s.query("SELECT 1 AS one").rows == [(1,)]
        assert s.count_chunks() > 0


def test_query_makes_the_transaction_read_only(store: TenantStore, entities):
    """The boundary behind the keyword filter: an UPDATE fed straight to the connection after a
    query() call fails, because the transaction is read-only and Postgres won't switch back."""
    with store.session("harbor") as s:
        assert s.query("SELECT count(*) FROM rag_facts").rows[0][0] > 0
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction), s.conn.transaction():
            s.conn.execute("UPDATE rag_facts SET unit = 'tampered'")
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction), s.conn.transaction():
            s.conn.execute("DELETE FROM rag_documents")
        # And it cannot be switched back ("cannot set transaction read-write mode inside a
        # read-only transaction"), not even at the top level.
        with pytest.raises(psycopg.errors.ActiveSqlTransaction), s.conn.transaction():
            s.conn.execute("SET LOCAL transaction_read_only = off")
        with pytest.raises(psycopg.errors.ActiveSqlTransaction), s.conn.transaction():
            s.conn.execute("SET transaction_read_only = off")
        # Reads continue to work.
        assert s.query("SELECT 1 AS one").rows == [(1,)]
    # And a fresh session can write again (the setting was transaction-local).
    with store.session("harbor") as s:
        n = s.conn.execute("UPDATE rag_facts SET unit = unit WHERE field = 'late_fee'").rowcount
        assert n >= 1
        assert all(f.unit != "tampered" for f in s.facts())


def test_query_statement_timeout(store: TenantStore):
    slow = "SELECT count(*) FROM generate_series(1, 1000000) a, generate_series(1, 1000000) b"
    with store.session("harbor") as s:
        with pytest.raises(SqlRejected) as exc:
            s.query(slow, timeout_ms=50)
        assert "timeout" in exc.value.reason.lower()
        # pg_sleep never gets that far: the guard refuses pg_ functions outright.
        with pytest.raises(SqlRejected):
            s.query("SELECT pg_sleep(5)")
        # The session's statement timeout is back to what it was.
        assert s.conn.execute("SHOW statement_timeout").fetchone() == ("0",)
        assert s.query("SELECT 1 AS one").rows == [(1,)]


# ------------------------------------------------------------------ cross-tenant


def test_query_sees_only_the_sessions_tenant(store: TenantStore, entities: dict[str, str]):
    with store.session("summit") as s:
        _clear_facts(s)
        s.add_facts([FactRow("summit", "late_fee", 50, "USD")])
    with store.session("oakline") as s:
        _clear_facts(s)
        s.add_facts([FactRow("oakline", "late_fee", 5, "percent_of_rent")])
    with store.session("summit") as s:
        r = s.query("SELECT * FROM rag_facts")
        tenants = {row[r.columns.index("tenant_id")] for row in r.rows}
        assert tenants == {"summit"}
        assert [row[r.columns.index("value_num")] for row in r.rows] == [50]
        for table in ("rag_entities", "rag_documents", "rag_dictionary", "rag_extraction_jobs"):
            r = s.query(f"SELECT DISTINCT tenant_id FROM {table}")
            assert {row[0] for row in r.rows} <= {"summit"}, table
        # A WHERE clause naming another tenant is filtered to nothing, not an error.
        r = s.query("SELECT count(*) FROM rag_facts WHERE tenant_id = 'harbor'")
        assert r.rows == [(0,)]
        r = s.query(
            "SELECT value_num FROM rag_facts WHERE tenant_id = 'summit' "
            "UNION ALL SELECT value_num FROM rag_facts WHERE tenant_id = 'harbor'"
        )
        assert [row[0] for row in r.rows] == [50]


def test_fetch_of_another_tenants_document_is_unknown(store: TenantStore):
    with store.session("harbor") as s:
        harbor_doc = s.conn.execute(
            "SELECT doc_id::text FROM rag_documents ORDER BY source LIMIT 1"
        ).fetchone()
    assert harbor_doc
    with store.session("harbor") as s:
        assert s.fetch(harbor_doc[0]).source.startswith("harbor/")
    with store.session("summit") as s:
        with pytest.raises(KeyError):
            s.fetch(harbor_doc[0])
        r = s.query(f"SELECT count(*) FROM rag_documents WHERE doc_id = '{harbor_doc[0]}'")
        assert r.rows == [(0,)]


# --- precision-aware conflicts (PR 4)

FY2019 = {"period_start": date(2019, 1, 1), "period_end": date(2019, 12, 31)}


@pytest.fixture
def company(store: TenantStore, entities):
    """A company entity in harbor, removed with its facts afterwards."""
    with store.session("harbor") as s:
        eid = s.upsert_entity("company", "Vertex Test", ["VRTX-T"])
    yield "Vertex Test"
    with store.session("harbor") as s:
        s.conn.execute("DELETE FROM rag_facts WHERE entity_id = %s::uuid", (eid,))
        s.conn.execute("DELETE FROM rag_entities WHERE entity_id = %s::uuid", (eid,))


def test_precision_only_difference_forms_no_group(store: TenantStore, company: str):
    """Vertex FY2019 revenue: exact in two 10-Ks, to $100K in a third. Not a conflict."""
    with store.session("harbor") as s:
        ids = s.add_facts(
            [
                FactRow(company, "revenue", 4_162_821_000, "USD", **FY2019, precision=1000),
                FactRow(company, "revenue", 4_162_821_000, "USD", **FY2019, precision=1000),
                FactRow(company, "revenue", 4_162_800_000, "USD", **FY2019, precision=100_000),
            ]
        )
        got = {f.fact_id: f for f in s.facts(entity=company, field="revenue")}
        assert set(got) == set(ids)
        assert [got[i].conflict_group_id for i in ids] == [None, None, None]
        # The same pair with no precision is PR 3's exact rule: a conflict.
        a, b = s.add_facts(
            [
                FactRow(company, "revenue_exact", 4_162_821_000, "USD", **FY2019),
                FactRow(company, "revenue_exact", 4_162_800_000, "USD", **FY2019),
            ]
        )
        exact = {f.fact_id: f.conflict_group_id for f in s.facts(field="revenue_exact")}
        assert exact[a] is not None and exact[a] == exact[b]


def test_restatement_chain_then_rounded_repeat_is_one_group(store: TenantStore, company: str):
    """Pfizer-style FY2019 revenue: 51,750M, recast to 41,172M, then 40,905M, all in millions.
    Then a repeat of 41,172M rounded to hundreds of millions: it agrees with 41,172M but not
    with the others, and joins the same group (see the add_facts docstring)."""
    with store.session("harbor") as s:
        chain = s.add_facts(
            [
                FactRow(company, "revenue", v * 1_000_000, "USD", **FY2019, precision=1_000_000)
                for v in (51_750, 41_172, 40_905)
            ]
        )
        groups = {f.fact_id: f.conflict_group_id for f in s.facts(field="revenue")}
        assert set(groups) == set(chain)
        (group,) = set(groups.values())
        assert group is not None

        (repeat,) = s.add_facts(
            [FactRow(company, "revenue", 41_200_000_000, "USD", **FY2019, precision=10**8)]
        )
        groups = {f.fact_id: f.conflict_group_id for f in s.facts(field="revenue")}
        assert groups == {**{f: group for f in chain}, repeat: group}

        # Another period of the same field is a different key: untouched.
        (other,) = s.add_facts(
            [
                FactRow(
                    company,
                    "revenue",
                    41_908_000_000,
                    "USD",
                    period_start=date(2020, 1, 1),
                    period_end=date(2020, 12, 31),
                    precision=1_000_000,
                )
            ]
        )
        assert {f.fact_id: f.conflict_group_id for f in s.facts(field="revenue")}[other] is None


def test_precision_and_source_ref_round_trip(store: TenantStore, company: str):
    ref = "us-gaap:Revenues@0000875320-22-000007"
    with store.session("harbor") as s:
        a, b, c = s.add_facts(
            [
                FactRow(
                    company,
                    "revenue",
                    4_162_800_000,
                    "USD",
                    **FY2019,
                    precision=100_000,
                    source_ref=ref,
                ),
                FactRow(company, "revenue", 4_162_821_000, "USD", **FY2019),
                FactRow(company, "auditor", "Ernst & Young LLP", **FY2019),
            ]
        )
        got = {f.fact_id: f for f in s.facts(entity="VRTX-T")}
        assert got[a].precision == 100_000 and isinstance(got[a].precision, int)
        assert got[a].source_ref == ref and got[a].value == 4_162_800_000
        assert got[b].precision is None and got[b].source_ref == ""
        assert got[c].precision is None and got[c].source_ref == ""
        assert all(f.conflict_group_id is None for f in got.values())
        assert got[a].to_source("harbor").span == f"fact:{a}"
    with store.session("harbor") as s:
        r = s.query(
            "SELECT f.fact_id, f.precision, f.source_ref FROM rag_facts f "
            "JOIN rag_entities e ON e.entity_id = f.entity_id "
            "WHERE e.canonical_name = 'Vertex Test' ORDER BY f.created_at, f.fact_id"
        )
        assert r.columns == ["fact_id", "precision", "source_ref"]
        rows = {row[0]: row[1:] for row in r.rows}
        assert rows == {a: (100_000, ref), b: (None, ""), c: (None, "")}


def test_add_facts_rejects_bad_precision(store: TenantStore, company: str):
    with store.session("harbor") as s:
        with pytest.raises(ValueError):
            s.add_facts([FactRow(company, "revenue", 1, precision=0)])
        with pytest.raises(ValueError):
            s.add_facts([FactRow(company, "auditor", "EY", precision=1000)])
        assert s.facts(entity=company) == []


def test_migrate_adds_precision_columns_to_a_pr3_schema(admin_url: str, store):
    """A PR 3 database lacks precision and source_ref. migrate adds them on the parent and
    they cascade to every existing partition, with the default filled in."""
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute("ALTER TABLE rag_facts DROP COLUMN precision, DROP COLUMN source_ref")
    migrate(admin_url)
    with psycopg.connect(admin_url) as conn:
        for table in ("rag_facts", *(partition_name("rag_facts", t) for t in TENANTS)):
            cols = dict(
                conn.execute(
                    "SELECT column_name, is_nullable || ':' || data_type "
                    "FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name IN ('precision', 'source_ref')",
                    (table,),
                ).fetchall()
            )
            assert cols == {"precision": "YES:numeric", "source_ref": "NO:text"}, table
        assert conn.execute(
            "SELECT count(*) FROM rag_facts WHERE source_ref <> ''"
        ).fetchone() == (0,)
    with psycopg.connect(app_url_from_admin(admin_url)) as conn:
        assert check_posture(conn) == []
