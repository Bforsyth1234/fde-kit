"""`add_facts` as one batch: equivalence with the row-at-a-time implementation it replaced.

Same gate as tests/test_rag_facts.py: FDE_TEST_PG_URL must be an admin URL on a server with
pgvector >= 0.8, or everything here skips.

The old implementation is kept below as `reference_add_facts`, verbatim apart from taking the
session as an argument (and noting, without changing anything, when its group choice depends
on heap order). The property test runs it and the batch version over the same seeded random
batches into two fresh tenants and compares the resulting rows, the conflict-group partition
and the returned id order.
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
import pytest

from fde_kit.rag import FactRow, HashEmbedder, TenantStore, app_url_from_admin, values_agree
from fde_kit.rag.schema import create_tenant, drop_tenant, migrate
from fde_kit.rag.store import TenantSession

PG_URL = os.environ.get("FDE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="FDE_TEST_PG_URL not set")

REF, NEW = "afb-ref", "afb-new"
LOCK_A, LOCK_B = "afb-lock-a", "afb-lock-b"
TENANTS = (REF, NEW, LOCK_A, LOCK_B)

# kind, canonical name, aliases. "Shared" is an alias of two entities (the canonical_name
# order breaks the tie); "pfizer" is Pfizer Inc's alias and, in another case, a canonical name
# of its own (a canonical match wins over an alias match).
ENTITIES = [
    ("company", "Vertex Pharmaceuticals", ["Vertex", "VRTX", "Shared"]),
    ("company", "Pfizer Inc", ["PFE", "pfizer", "Shared"]),
    ("company", "PFIZER", ["pfe-legacy"]),
    ("county", "Hartford County, CT", ["Hartford"]),
]
NAMES = [
    "Vertex Pharmaceuticals",
    "vertex",
    "VRTX",
    "Shared",
    "shared",
    "Pfizer Inc",
    "PFE",
    "pfizer",
    "PFIZER",
    "pfe-legacy",
    "HARTFORD",
    "Hartford County, CT",
]
FIELDS = ["revenue", "units", "auditor"]
PERIODS: list[tuple[Any, Any]] = [
    (None, None),
    (date(2019, 1, 1), date(2019, 12, 31)),
    (date(2020, 1, 1), None),
    ("2019-01-01", "2019-12-31"),  # the same key as a str, through the ::date cast
    (datetime(2020, 1, 1, 0, 0), None),  # and as a naive datetime
]
BIG = 4_162_821_000
NUMBERS: list[Any] = [
    BIG,
    BIG,
    4_162_800_000,
    4_163_000_000,
    BIG + 400,
    float(BIG),
    Decimal("4162821000.00"),
    51_750_000_000,
    41_172_000_000,
    41_200_000_000,
    10,
    10,
    Decimal("10.3"),
    Decimal("10.30"),
    10.4,
    0.1 + 0.2,  # 0.30000000000000004: stored as 0.3 (float8 -> numeric keeps 15 digits)
    Decimal("0.3"),
    1 / 3,
    -1850,
    -1900,
    1850,
]
PRECISIONS: list[Any] = [None, None, None, 1, 1000, 1e6, 10**8, 100, Decimal("0.5"), 1.09]
TEXTS = ["Ernst & Young LLP", "EY", "ey", "KPMG"]


def reference_add_facts(
    s: TenantSession,
    rows: list[FactRow],
    *,
    extraction_job_id: str | None = None,
    ambiguous: list[int] | None = None,
) -> list[str]:
    """The pre-batch `TenantSession.add_facts`, row at a time. `ambiguous` (test-only)
    collects the indexes of rows whose differing candidates carried two or more distinct
    groups: there the group taken is the first in heap order, which the batch version can
    only model for facts on record before the call."""
    out: list[str] = []
    for i, r in enumerate(rows):
        entity_id = s.resolve_entity(r.entity)
        if entity_id is None:
            raise ValueError(f"unknown entity {r.entity!r} for tenant {s.tenant_id}")
        if isinstance(r.value, bool) or not isinstance(r.value, int | float | Decimal | str):
            raise ValueError(f"fact value must be a number or a string, got {r.value!r}")
        if isinstance(r.value, Decimal) and not r.value.is_finite():
            raise ValueError(f"fact value must be finite, got {r.value!r}")
        if r.precision is not None:
            if isinstance(r.value, str):
                raise ValueError(f"precision applies to numbers only, got {r.value!r}")
            if isinstance(r.precision, bool) or not r.precision > 0:
                raise ValueError(f"precision must be positive, got {r.precision!r}")
        num = r.value if isinstance(r.value, int | float | Decimal) else None
        txt = r.value if isinstance(r.value, str) else None
        candidates = s.conn.execute(
            s.sql("""
            SELECT fact_id::text, conflict_group_id::text, value_num, value_text, precision
            FROM @facts
            WHERE entity_id = %(e)s::uuid AND field = %(f)s::text
              AND period_start IS NOT DISTINCT FROM %(ps)s::date
              AND period_end IS NOT DISTINCT FROM %(pe)s::date
            """),
            {"e": entity_id, "f": r.field, "ps": r.period_start, "pe": r.period_end},
        ).fetchall()
        differing = [
            (fid, grp)
            for fid, grp, vnum, vtxt, prec in candidates
            if not values_agree(r.value, r.precision, vnum if vnum is not None else vtxt, prec)
        ]
        if ambiguous is not None and len({g for _, g in differing if g}) > 1:
            ambiguous.append(i)
        group: str | None = None
        if differing:
            group = next((g for _, g in differing if g), None) or str(uuid.uuid4())
            s.conn.execute(
                s.sql(
                    "UPDATE @facts SET conflict_group_id = %s::uuid "
                    "WHERE fact_id = ANY(%s::uuid[]) "
                    "AND conflict_group_id IS DISTINCT FROM %s::uuid"
                ),
                (group, [f for f, _ in differing], group),
            )
        row = s.conn.execute(
            s.sql("""
            INSERT INTO @facts (tenant_id, entity_id, field, period_start, period_end,
                                value_num, value_text, unit, doc_id, chunk_id,
                                conflict_group_id, extraction_job_id, precision, source_ref)
            VALUES (%s, %s::uuid, %s, %s::date, %s::date, %s::numeric, %s::text, %s,
                    %s::uuid, %s::uuid, %s::uuid, %s::uuid, %s::numeric, %s)
            RETURNING fact_id::text
            """),
            (
                s.tenant_id,
                entity_id,
                r.field,
                r.period_start,
                r.period_end,
                num,
                txt,
                r.unit,
                r.doc_id,
                r.chunk_id,
                group,
                extraction_job_id,
                r.precision,
                r.source_ref,
            ),
        ).fetchone()
        assert row is not None
        out.append(row[0])
    return out


# ------------------------------------------------------------------ fixtures


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
    migrate(PG_URL)
    for tid in TENANTS:
        drop_tenant(PG_URL, tid)
        create_tenant(PG_URL, tid)
    yield PG_URL
    for tid in TENANTS:
        drop_tenant(PG_URL, tid)


@pytest.fixture(scope="module")
def store(admin_url: str):
    with TenantStore(app_url_from_admin(admin_url), HashEmbedder()) as s:
        for tid in (REF, NEW):
            with s.session(tid) as sess:
                for kind, name, aliases in ENTITIES:
                    sess.upsert_entity(kind, name, aliases)
                sess.ingest("afb/a.md", "Document A.\n\nRevenue was reported.")
                sess.ingest("afb/b.md", "Document B.\n\nRevenue was restated.")
        yield s


# ------------------------------------------------------------------ helpers


def _random_rows(rng: random.Random) -> list[dict[str, Any]]:
    """Abstract rows (doc by source, not id), so the same batch can go to either tenant."""
    out = []
    for _ in range(rng.randint(1, 25)):
        field = rng.choice(FIELDS)
        text = field == "auditor" or rng.random() < 0.05
        value = rng.choice(TEXTS) if text else rng.choice(NUMBERS)
        ps, pe = rng.choice(PERIODS)
        out.append(
            {
                "entity": rng.choice(NAMES),
                "field": field,
                "value": value,
                "unit": rng.choice(["USD", "USD", ""]),
                "period_start": ps,
                "period_end": pe,
                "doc": rng.choice([None, None, "afb/a.md", "afb/b.md"]),
                "precision": None if text else rng.choice(PRECISIONS),
                "source_ref": rng.choice(["", "", "us-gaap:Revenues@0000875320-22-000007"]),
            }
        )
    return out


def _materialize(s: TenantSession, abstract: list[dict[str, Any]]) -> list[FactRow]:
    docs = dict(s.conn.execute("SELECT source, doc_id::text FROM rag_documents").fetchall())
    return [
        FactRow(
            a["entity"],
            a["field"],
            a["value"],
            a["unit"],
            period_start=a["period_start"],
            period_end=a["period_end"],
            doc_id=docs[a["doc"]] if a["doc"] else None,
            precision=a["precision"],
            source_ref=a["source_ref"],
        )
        for a in abstract
    ]


def _snapshot(s: TenantSession, labels: dict[str, tuple[int, int]]):
    """Every fact as a tenant-independent row keyed by its label (call, index in the call),
    plus the conflict-group partition over labels."""
    rows = s.conn.execute(
        """
        SELECT f.fact_id::text, e.canonical_name, e.kind, f.field, f.period_start,
               f.period_end, f.value_num::text, f.value_text, f.unit, f.precision::text,
               f.source_ref, d.source, f.extraction_job_id IS NOT NULL,
               f.conflict_group_id::text
        FROM rag_facts f
        JOIN rag_entities e ON e.entity_id = f.entity_id
        LEFT JOIN rag_documents d ON d.doc_id = f.doc_id
        """
    ).fetchall()
    assert len(rows) == len(labels)
    by_label = {}
    groups: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for fid, *data, grp in rows:
        by_label[labels[fid]] = tuple(data)
        if grp is not None:
            groups[grp].add(labels[fid])
    return by_label, {frozenset(g) for g in groups.values()}


# ------------------------------------------------------------------ tests


def test_resolve_entities_matches_resolve_entity(store: TenantStore):
    with store.session(NEW) as s:
        names = [*NAMES, "nobody", "VERTEX PHARMACEUTICALS", "shared ", ""]
        batch = s._resolve_entities(names)
        one_by_one = {n: s.resolve_entity(n) for n in names}
        assert batch == {n: e for n, e in one_by_one.items() if e is not None}
        # The ties resolve as documented: canonical order for "Shared", canonical wins for
        # "pfizer" (the PFIZER entity, not Pfizer Inc's alias).
        canon = dict(
            s.conn.execute("SELECT entity_id::text, canonical_name FROM rag_entities").fetchall()
        )
        assert canon[batch["shared"]] == "Pfizer Inc"
        assert canon[batch["pfizer"]] == "PFIZER"
        assert s._resolve_entities([]) == {}


SEEDS = range(200)


def test_batch_add_facts_equals_row_at_a_time(store: TenantStore):
    """Seeded random batches, 1-4 sequential calls each, into two fresh tenants: the old
    implementation into REF, the batch one into NEW. Same rows, same group partition, same
    returned-id order, every seed. Where the old code's choice of group depended on heap
    order (two groups on one key, see the add_facts docstring), the partition is compared
    only if the reference made no such choice."""
    stats = {
        "rows": 0,
        "grouped": 0,
        "regrouped_existing": 0,
        "ambiguous_seeds": 0,
        "ambiguous_but_equal": 0,
    }
    for seed in SEEDS:
        rng = random.Random(seed)
        for tid in (REF, NEW):
            with store.session(tid) as s:
                s.conn.execute("DELETE FROM rag_facts")
        labels = {REF: {}, NEW: {}}
        ambiguous: list[int] = []
        for call in range(rng.randint(1, 4)):
            abstract = _random_rows(rng)
            with_job = rng.random() < 0.3
            for tid in (REF, NEW):
                with store.session(tid) as s:
                    rows = _materialize(s, abstract)
                    job = s.start_extraction_job("test") if with_job else None
                    if tid == REF:
                        ids = reference_add_facts(
                            s, rows, extraction_job_id=job, ambiguous=ambiguous
                        )
                    else:
                        before = dict(
                            s.conn.execute(
                                "SELECT fact_id::text, conflict_group_id::text FROM rag_facts"
                            ).fetchall()
                        )
                        ids = s.add_facts(rows, extraction_job_id=job)
                        after = dict(
                            s.conn.execute(
                                "SELECT fact_id::text, conflict_group_id::text FROM rag_facts"
                                " WHERE fact_id = ANY(%s::uuid[])",
                                (list(before),),
                            ).fetchall()
                        )
                        stats["regrouped_existing"] += sum(
                            before[f] != after[f] for f in before
                        )
                    assert len(ids) == len(rows) == len(set(ids))
                    labels[tid].update({fid: (call, i) for i, fid in enumerate(ids)})
                    stats["rows"] += len(rows) if tid == NEW else 0
        with store.session(REF) as s:
            ref_rows, ref_groups = _snapshot(s, labels[REF])
        with store.session(NEW) as s:
            new_rows, new_groups = _snapshot(s, labels[NEW])
        # Rows by label: the returned ids map to the same rows, in the same order.
        assert new_rows == ref_rows, f"seed {seed}"
        grouped = set().union(*ref_groups) if ref_groups else set()
        assert (set().union(*new_groups) if new_groups else set()) == grouped, f"seed {seed}"
        stats["grouped"] += len(grouped)
        if ambiguous:
            stats["ambiguous_seeds"] += 1
            stats["ambiguous_but_equal"] += new_groups == ref_groups
        else:
            assert new_groups == ref_groups, f"seed {seed}"
    print(f"\nadd_facts equivalence over {len(SEEDS)} seeds: {stats}")
    # The generator must actually exercise the rule, not just insert lone facts.
    assert stats["grouped"] > stats["rows"] // 5
    assert stats["regrouped_existing"] > 50
    assert stats["ambiguous_seeds"] < len(SEEDS) // 10


TWO_GROUPS = [
    # 10 and 10.3 (exact) disagree: one group. 10.4 at p=1 agrees with both, ungrouped. Then
    # 10 at p=1.09 agrees with 10 and 10.3 but not 10.4: a second group on the same key.
    (Decimal(10), None),
    (Decimal("10.4"), Decimal(1)),
    (Decimal("10.3"), None),
    (Decimal(10), Decimal("1.09")),
]


@pytest.mark.parametrize(
    "last",
    [
        pytest.param((Decimal("9.3"), None), id="differs-from-both-groups-agrees-with-one"),
        pytest.param((Decimal("10.4"), Decimal("1.09")), id="agrees-with-the-ungrouped"),
    ],
)
@pytest.mark.parametrize("split", [[1, 1, 1, 1, 1], [4, 1], [5]], ids=str)
def test_two_groups_on_one_key(store: TenantStore, last, split):
    """The one case where the group a fact takes depends on candidate order (see the
    add_facts docstring): a value that differs from members of two groups. Here the result
    matches the old code whether the history arrives row by row, as one prior call, or in
    the same batch."""
    values = [*TWO_GROUPS, last]
    flagged: list[int] = []
    snaps = []
    for tid in (REF, NEW):
        labels: dict[str, tuple[int, int]] = {}
        with store.session(tid) as s:
            s.conn.execute("DELETE FROM rag_facts")
        at = 0
        for n in split:
            rows = [
                FactRow("Hartford", "units", v, precision=p, period_start=date(2025, 1, 1))
                for v, p in values[at : at + n]
            ]
            with store.session(tid) as s:
                if tid == REF:
                    ids = reference_add_facts(s, rows, ambiguous=flagged)
                else:
                    ids = s.add_facts(rows)
            labels.update({fid: (0, at + i) for i, fid in enumerate(ids)})
            at += n
        with store.session(tid) as s:
            snaps.append(_snapshot(s, labels))
    assert flagged  # the reference did have to choose between two groups
    assert snaps[0] == snaps[1]


def test_errors_match_and_nothing_is_written(store: TenantStore):
    """Same exception and message as the old code for the first bad row, raised before any
    row is written (the old code had written the rows before it)."""
    ok = FactRow("VRTX", "revenue", 1, "USD")
    bad_batches = [
        [ok, FactRow("Nobody Ltd", "revenue", 1)],
        [ok, FactRow("VRTX", "revenue", None)],  # type: ignore[arg-type]
        [ok, FactRow("VRTX", "revenue", True)],
        [ok, FactRow("VRTX", "revenue", Decimal("NaN"))],
        [ok, FactRow("VRTX", "auditor", "EY", precision=1000)],
        [ok, FactRow("VRTX", "revenue", 5, precision=0)],
        [ok, FactRow("VRTX", "revenue", 5, precision=True)],
        # Unknown entity is checked before the value, per row, in row order.
        [ok, FactRow("Nobody Ltd", "revenue", None), FactRow("VRTX", "x", 1, precision=-1)],  # type: ignore[arg-type]
        [ok, FactRow("VRTX", "revenue", None), FactRow("Nobody Ltd", "x", 1)],  # type: ignore[arg-type]
    ]
    for rows in bad_batches:
        messages = []
        for tid, fn in ((REF, reference_add_facts), (NEW, TenantSession.add_facts)):
            with store.session(tid) as s:
                s.conn.execute("DELETE FROM rag_facts")
                with pytest.raises(ValueError) as exc:
                    fn(s, rows)
                messages.append(str(exc.value).replace(tid, "<tenant>"))
                if tid == NEW:
                    assert s.conn.execute("SELECT count(*) FROM rag_facts").fetchone() == (0,)
        assert messages[0] == messages[1]
    with store.session(NEW) as s:
        assert s.add_facts([]) == []


def test_concurrent_add_facts_into_one_tenant_see_each_other(store: TenantStore, admin_url):
    """Two transactions add disagreeing values for one key. Without the advisory lock each
    misses the other's uncommitted row and neither is grouped; with it the second waits for
    the first to commit, sees its row, and both end up in one group. Another tenant's load
    does not wait."""
    with store.session(LOCK_A) as s:
        s.upsert_entity("company", "Acme")
    with store.session(LOCK_B) as s:
        s.upsert_entity("company", "Acme")
    first_in = threading.Event()
    release = threading.Event()
    done: dict[str, float] = {}

    def first() -> None:
        with store.session(LOCK_A) as s:
            s.add_facts([FactRow("Acme", "revenue", 100, period_start=date(2024, 1, 1))])
            first_in.set()
            release.wait(10)
        done["first"] = time.monotonic()

    def second() -> None:
        with store.session(LOCK_A) as s:
            s.add_facts([FactRow("Acme", "revenue", 200, period_start=date(2024, 1, 1))])
        done["second"] = time.monotonic()

    t1 = threading.Thread(target=first)
    t1.start()
    assert first_in.wait(10)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.5)
    assert "second" not in done  # blocked on the lock
    # Another tenant is not serialized behind LOCK_A.
    with store.session(LOCK_B) as s:
        s.add_facts([FactRow("Acme", "revenue", 1)])
    release.set()
    t1.join(10)
    t2.join(10)
    assert done["second"] >= done["first"]
    with store.session(LOCK_A) as s:
        groups = [f.conflict_group_id for f in s.facts(field="revenue")]
    assert len(groups) == 2 and groups[0] is not None and groups[0] == groups[1]
