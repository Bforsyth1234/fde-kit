"""A synthetic corpus built to make tenant leaks *plausible*, not obvious.

Three property managers use the same residential lease template. The clauses are word-for-word
identical across tenants; only the numbers differ (late fee, grace period, pet deposit, notice
period). So a leaked chunk doesn't look like a leak. Ask Summit "what is the late fee?" and a
chunk from Harbor reads as a perfectly good answer, just the wrong one: $75 instead of $50.

That is what makes the corpus useful for the isolation tests, and it makes the retrieval eval
honest too: within a tenant, twelve leases share the same text, so "rent for unit 2B" can only be
answered by the chunk that carries that unit, and pure vector search struggles to tell them
apart.

Each tenant also has one document the others don't (a maintenance policy, a parking addendum,
pool rules), so a question about it from another tenant has a correct answer of "nothing".

All names, addresses and numbers are invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Doc:
    source: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Question:
    id: str
    tenant: str
    query: str
    answer: str
    """A string that must appear in a relevant chunk."""
    sources: tuple[str, ...] = ()
    """If set, the relevant chunk must also come from one of these documents."""
    category: str = "policy"


TENANTS: dict[str, dict] = {
    "harbor": {
        "name": "Harbor Property Management",
        "city": "New Haven, CT",
        "late_fee": "$75",
        "grace_days": "5",
        "pet_deposit": "$400",
        "pet_rule": "Up to two cats or one dog under 40 lbs",
        "notice_days": "60",
        "deposit": "one and one-half months' rent",
    },
    "summit": {
        "name": "Summit Rentals",
        "city": "Hartford, CT",
        "late_fee": "$50",
        "grace_days": "3",
        "pet_deposit": "$250",
        "pet_rule": "One pet of any kind under 25 lbs",
        "notice_days": "30",
        "deposit": "one month's rent",
    },
    "oakline": {
        "name": "Oakline Homes",
        "city": "Stamford, CT",
        "late_fee": "5% of the monthly rent",
        "grace_days": "7",
        "pet_deposit": "not applicable",
        "pet_rule": "No pets, except service and assistance animals",
        "notice_days": "45",
        "deposit": "two months' rent",
    },
}


_FIRST = ["Dana", "Marcus", "Priya", "Tomas", "Alicia", "Ben", "Chen", "Rosa", "Gregor", "Imani",
          "Hiro", "Sofia", "Kwame", "Lena", "Rafael", "Ines", "Owen", "Mira", "Jonas", "Talia",
          "Eamon", "Noor", "Viktor", "Ama", "Felix", "Yara", "Desmond", "Kiri", "Arjun", "Elsa",
          "Mateo", "Zoe", "Idris", "Hana", "Lucas", "Freya"]
_LAST = ["Whitfield", "Oyelaran", "Raman", "Brennan", "Fernsby", "Okafor", "Liwei", "Delgado",
         "Lindqvist", "Castellanos", "Tanabe", "Marchetti", "Mensah", "Vogel", "Ibarra", "Soto",
         "Pryce", "Halvorsen", "Achterberg", "Novak", "Quill", "Haddad", "Petrov", "Boateng",
         "Ashby", "Kaur", "Moreau", "Tane", "Iyer", "Strand", "Rossi", "Kimura", "Bello",
         "Sato", "Ferreira", "Nyberg"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]
_LAST_DAY = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_UNIT_NAMES = {
    "harbor": [f"Unit {f}{c}" for f in (1, 2, 3, 4) for c in "ABC"],
    "summit": [f"Apartment {f}0{n}" for f in (1, 2, 3, 4) for n in (1, 2, 3)],
    "oakline": [f"Townhome {n}" for n in range(1, 7)] + [f"Flat {n}" for n in range(10, 16)],
}
_BASE_RENT = {"harbor": 1800, "summit": 1350, "oakline": 2600}


def _units(tid: str, offset: int) -> list[tuple[str, str, str, str, str]]:
    out = []
    for i, unit in enumerate(_UNIT_NAMES[tid]):
        n = offset + i
        name = f"{_FIRST[n]} {_LAST[(n * 7) % len(_LAST)]}"
        rent = _BASE_RENT[tid] + ((n * 37) % 23) * 25
        m = (n * 5) % 12
        start = f"{_MONTHS[m]} 1, 2026"
        end_m = (m - 1) % 12
        end_y = 2027 if m else 2026
        end = f"{_MONTHS[end_m]} {_LAST_DAY[end_m]}, {end_y}"
        out.append((unit, name, f"${rent:,}", start, end))
    return out


for _i, _tid in enumerate(TENANTS):
    TENANTS[_tid]["units"] = _units(_tid, _i * 12)


_PREFIX = {"harbor": "HPM", "summit": "SR", "oakline": "OH"}


def lease_ref(tid: str, i: int) -> str:
    """A reference number like HPM-26-4317. Opaque on purpose: embeddings can't read meaning
    into it, which is exactly the kind of query keyword search exists for."""
    return f"{_PREFIX[tid]}-26-{(i * 7919 + 1301) % 9000 + 1000}"


def _lease(
    t: dict, unit: str, tenant_name: str, rent: str, start: str, end: str, ref: str
) -> str:
    return f"""RESIDENTIAL LEASE AGREEMENT

Lease reference: {ref}

This lease is between {t["name"]} ("Landlord") and {tenant_name} ("Tenant") for {unit}, {t["city"]}.

1. TERM

The lease term begins on {start} and ends on {end}.

2. RENT

Monthly rent for {unit} is {rent}, due on the first day of each month.

3. LATE FEES

Rent received more than {t["grace_days"]} days after the due date incurs a late fee of {t["late_fee"]}. Repeated late payment may be treated as a material breach.

4. SECURITY DEPOSIT

Tenant pays a security deposit equal to {t["deposit"]} before move-in. The deposit is returned within 30 days of move-out, less any lawful deductions.

5. PETS

Pet policy: {t["pet_rule"]}. Pet deposit: {t["pet_deposit"]}.

6. NOTICE TO VACATE

Tenant must give written notice at least {t["notice_days"]} days before the end of the term if Tenant does not intend to renew.

7. MAINTENANCE AND REPAIRS

Tenant reports needed repairs in writing. Landlord makes repairs within a reasonable time. Tenant is responsible for damage caused by Tenant or guests.
"""


_EXTRAS: dict[str, Doc] = {
    "harbor": Doc(
        "harbor/policies/maintenance.md",
        "Harbor Property Management: Maintenance and Emergency Policy",
        """MAINTENANCE AND EMERGENCY POLICY

Routine requests go through the resident portal and are answered within two business days.

EMERGENCIES

For a gas smell, flooding, no heat below 55 degrees, or a lockout, call the 24-hour emergency line at 203-555-0148. A technician is dispatched within four hours.

PEST CONTROL

Quarterly pest treatment is included at no charge.
""",
        {"type": "policy"},
    ),
    "summit": Doc(
        "summit/addenda/parking.md",
        "Summit Rentals: Parking Addendum",
        """PARKING ADDENDUM

Each unit may hold one parking permit. The permit fee is $35 per month, added to rent.

GUEST PARKING

Guests may park in marked visitor spaces for up to 48 hours. Vehicles left longer are towed at the owner's expense.

WINTER PARKING BAN

During a declared snow emergency, vehicles must be moved to Lot C by 8 p.m.
""",
        {"type": "addendum"},
    ),
    "oakline": Doc(
        "oakline/policies/pool.md",
        "Oakline Homes: Pool and Clubhouse Rules",
        """POOL AND CLUBHOUSE RULES

The pool is open from Memorial Day to Labor Day, 9 a.m. to 9 p.m.

GUESTS

Each household may bring up to four guests. Residents must accompany guests at all times.

CLUBHOUSE RESERVATIONS

The clubhouse can be reserved for private events for a $150 refundable deposit.
""",
        {"type": "policy"},
    ),
}


def _slug(unit: str) -> str:
    return unit.lower().replace(" ", "-")


def documents(tenant: str) -> list[Doc]:
    t = TENANTS[tenant]
    docs = [
        Doc(
            f"{tenant}/leases/{_slug(unit)}.md",
            f"Lease: {unit}, {name} ({t['name']})",
            _lease(t, unit, name, rent, start, end, lease_ref(tenant, i)),
            {"type": "lease", "unit": unit, "ref": lease_ref(tenant, i)},
        )
        for i, (unit, name, rent, start, end) in enumerate(t["units"])
    ]
    docs.append(_EXTRAS[tenant])
    return docs


def questions() -> list[Question]:
    qs: list[Question] = []
    for tid, t in TENANTS.items():
        leases = tuple(f"{tid}/leases/{_slug(u[0])}.md" for u in t["units"])
        # Paraphrased on purpose: none of these reuse the clause's own wording.
        qs += [
            Question(f"{tid}-late-fee", tid, "How much extra do I owe if my rent is late?",
                     t["late_fee"], leases),
            Question(f"{tid}-grace", tid, "Is there any slack before a late payment is penalised?",
                     f"more than {t['grace_days']} days", leases),
            Question(f"{tid}-notice", tid,
                     "If I'm not renewing, how far ahead do I have to tell the landlord?",
                     f"{t['notice_days']} days", leases),
            Question(f"{tid}-pets", tid, "Can I keep a dog in the apartment?", t["pet_rule"],
                     leases),
            Question(f"{tid}-deposit", tid, "How much money do I put down before moving in?",
                     t["deposit"], leases),
        ]
        for unit, name, rent, _start, end in t["units"][::3]:
            src = (f"{tid}/leases/{_slug(unit)}.md",)
            qs += [
                Question(f"{tid}-rent-{_slug(unit)}", tid, f"What does {unit} pay each month?",
                         rent, src, "unit-specific"),
                Question(f"{tid}-end-{_slug(unit)}", tid,
                         f"When does {name}'s lease run out?", end, src, "unit-specific"),
            ]
        for i, (unit, *_rest) in enumerate(t["units"]):
            if i % 3 != 1:
                continue
            ref = lease_ref(tid, i)
            qs.append(
                Question(f"{tid}-ref-{ref.lower()}", tid, f"Who is the tenant on lease {ref}?",
                         unit, (f"{tid}/leases/{_slug(unit)}.md",), "identifier")
            )
    extra = {
        "harbor": ("My kitchen is flooding at 2am, what number do I call?", "203-555-0148"),
        "summit": ("What does it cost to park my car?", "$35 per month"),
        "oakline": ("When can I go swimming?", "9 a.m. to 9 p.m."),
    }
    for tid, (q, a) in extra.items():
        qs.append(Question(f"{tid}-extra", tid, q, a, (_EXTRAS[tid].source,), "tenant-only"))
    return qs


def seed(store, tenants: list[str] | None = None, reset: bool = True) -> dict[str, int]:
    """Ingest the corpus. Returns chunk counts per tenant.

    `reset` first deletes every document the demo tenant already has, so an older corpus can't
    linger as extra distractors. The DELETE has no WHERE clause: inside a tenant session the
    policy is the WHERE clause."""
    counts: dict[str, int] = {}
    for tid in tenants or list(TENANTS):
        with store.session(tid) as s:
            if reset:
                s.conn.execute("DELETE FROM rag_documents")
            for d in documents(tid):
                s.ingest(d.source, d.text, title=d.title, metadata=d.metadata)
            counts[tid] = s.count_chunks()
    return counts
