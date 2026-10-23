"""Pure tests for the demo corpus's PR 3 additions: the late-fee addendum, entities, fact rows,
the dictionary and the new eval questions. No Postgres."""

from __future__ import annotations

from datetime import date

import pytest

from fde_kit.rag.demo_corpus import (
    ADDENDUM_EFFECTIVE,
    ADDENDUM_LATE_FEES,
    TENANTS,
    addendum_source,
    dictionary,
    documents,
    entities,
    fact_rows,
    facts,
    questions,
)

TIDS = list(TENANTS)


def _addendum_doc(tid: str):
    docs = [d for d in documents(tid) if d.source == addendum_source(tid)]
    assert len(docs) == 1, f"{tid} should have exactly one late-fee addendum"
    return docs[0]


@pytest.mark.parametrize("tid", TIDS)
def test_every_tenant_has_the_addendum(tid: str):
    doc = _addendum_doc(tid)
    assert doc.title == "Addendum: late fee revision (2026)"
    assert doc.metadata["type"] == "addendum"
    assert ADDENDUM_LATE_FEES[tid] in doc.text
    assert "January 1, 2026" in doc.text
    # The lease texts are untouched: they still carry the original number, not the revised one.
    leases = [d for d in documents(tid) if d.metadata.get("type") == "lease"]
    assert leases and all(TENANTS[tid]["late_fee"] in d.text for d in leases)


def test_addendum_is_one_shared_template():
    """Substitute each tenant's name and number out and the texts must be identical, the same
    property the leases have. That is what keeps the isolation suite's leak plausible."""
    templates = set()
    for tid in TIDS:
        text = _addendum_doc(tid).text
        assert text.count(TENANTS[tid]["name"]) == 1
        assert text.count(ADDENDUM_LATE_FEES[tid]) == 1
        templates.add(
            text.replace(TENANTS[tid]["name"], "{name}").replace(ADDENDUM_LATE_FEES[tid], "{fee}")
        )
    assert len(templates) == 1


def test_other_tenants_numbers_do_not_appear_in_the_addendum():
    """The shared-template isolation test asserts other tenants' lease late fees are absent from
    this tenant's hits; the addendum must not sneak one in."""
    for tid in TIDS:
        text = _addendum_doc(tid).text
        for other in TIDS:
            if other != tid:
                assert TENANTS[other]["late_fee"] not in text
                assert ADDENDUM_LATE_FEES[other] not in text


@pytest.mark.parametrize("tid", TIDS)
def test_late_fee_rows_collide_on_period(tid: str):
    rows = [r for _src, r in fact_rows(tid) if r.field == "late_fee"]
    assert len(rows) == 2
    a, b = rows
    assert a.entity == b.entity == TENANTS[tid]["name"]
    assert a.period_start == b.period_start == ADDENDUM_EFFECTIVE == date(2026, 1, 1)
    assert a.period_end is None and b.period_end is None
    assert a.unit == b.unit
    assert a.value != b.value
    assert all(r.doc_id is None for r in rows)


def test_late_fee_values_and_units():
    by_tid = {tid: [r for r in facts(tid) if r.field == "late_fee"] for tid in TIDS}
    assert [(r.value, r.unit) for r in by_tid["harbor"]] == [(75, "USD"), (85, "USD")]
    assert [(r.value, r.unit) for r in by_tid["summit"]] == [(50, "USD"), (60, "USD")]
    assert [(r.value, r.unit) for r in by_tid["oakline"]] == [
        (5, "percent_of_rent"),
        (6, "percent_of_rent"),
    ]


@pytest.mark.parametrize("tid", TIDS)
def test_fact_sources_are_real_documents(tid: str):
    sources = {d.source for d in documents(tid)}
    pairs = fact_rows(tid)
    assert all(src in sources for src, _row in pairs)
    addendum_rows = [r for src, r in pairs if src == addendum_source(tid)]
    assert [r.field for r in addendum_rows] == ["late_fee"]
    assert facts(tid) == [r for _src, r in pairs]


@pytest.mark.parametrize("tid", TIDS)
def test_monthly_rent_rows_match_units_and_carry_periods(tid: str):
    units = TENANTS[tid]["units"]
    rows = [r for r in facts(tid) if r.field == "monthly_rent"]
    assert len(rows) == len(units) == 12
    for row, (unit, _name, rent, _start, _end) in zip(rows, units, strict=True):
        assert row.entity == unit
        assert row.unit == "USD"
        assert row.value == float(rent.replace("$", "").replace(",", ""))
        assert isinstance(row.period_start, date) and isinstance(row.period_end, date)
        assert row.period_start < row.period_end
        assert row.period_start.day == 1


@pytest.mark.parametrize("tid", TIDS)
def test_scalar_lease_facts(tid: str):
    t = TENANTS[tid]
    by_field = {r.field: r for r in facts(tid) if r.field != "monthly_rent"}
    assert by_field["grace_days"].value == int(t["grace_days"])
    assert by_field["grace_days"].unit == "days"
    assert by_field["notice_days"].value == int(t["notice_days"])
    assert by_field["notice_days"].unit == "days"
    assert by_field["security_deposit_months"].unit == "months"
    pet = by_field["pet_deposit"]
    if t["pet_deposit"] == "not applicable":
        assert pet.value == "not applicable"
    else:
        assert pet.value == float(t["pet_deposit"].lstrip("$")) and pet.unit == "USD"


def test_security_deposit_months():
    months = {
        tid: next(r for r in facts(tid) if r.field == "security_deposit_months") for tid in TIDS
    }
    assert {tid: r.value for tid, r in months.items()} == {"harbor": 1.5, "summit": 1, "oakline": 2}


def test_dictionary_covers_every_field():
    entries = {e.field: e for e in dictionary()}
    used = {r.field for tid in TIDS for r in facts(tid)}
    assert used <= set(entries), used - set(entries)
    assert len(entries) == len(dictionary()), "one entry per field"
    assert all(e.definition for e in entries.values())
    assert "percent_of_rent" in entries["late_fee"].notes and "USD" in entries["late_fee"].notes
    for name, unit in [("grace_days", "days"), ("monthly_rent", "USD"), ("notice_days", "days")]:
        assert entries[name].unit == unit


@pytest.mark.parametrize("tid", TIDS)
def test_entities_one_manager_and_one_unit_per_lease(tid: str):
    t = TENANTS[tid]
    ents = entities(tid)
    managers = [e for e in ents if e[0] == "property_manager"]
    units = [e for e in ents if e[0] == "unit"]
    assert len(managers) == 1 and len(units) == len(t["units"])
    assert managers[0][1] == t["name"]
    assert managers[0][2] == [tid, t["name"].split()[0]]
    assert [u[1] for u in units] == [u[0] for u in t["units"]]
    refs = {d.metadata["ref"] for d in documents(tid) if d.metadata.get("type") == "lease"}
    assert {u[2][0] for u in units} == refs
    # Every fact names an entity that exists.
    names = {e[1] for e in ents}
    assert {r.entity for r in facts(tid)} <= names


def test_questions_keep_old_ids_and_add_new_ones():
    qs = {q.id: q for q in questions()}
    for tid in TIDS:
        for old in ("late-fee", "grace", "notice", "pets", "deposit", "extra"):
            assert f"{tid}-{old}" in qs
        texts = {d.source: d.text for d in documents(tid)}
        grace = qs[f"{tid}-grace-days"]
        assert grace.query == "What is the grace period in days?"
        assert grace.answer == f"{TENANTS[tid]['grace_days']} days"
        assert any(grace.answer in texts[s] for s in grace.sources)
        rev = qs[f"{tid}-addendum-late-fee"]
        assert rev.query == "What is the revised late fee under the 2026 addendum?"
        assert rev.answer == ADDENDUM_LATE_FEES[tid]
        assert rev.sources == (addendum_source(tid),)
        assert rev.answer in texts[addendum_source(tid)]
        # The revised number must not be answerable from another tenant's documents.
        for other in TIDS:
            if other != tid:
                assert not any(rev.answer in d.text for d in documents(other))
