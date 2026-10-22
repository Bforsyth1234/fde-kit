"""Tenant content in Temporal's history and the tool journal.

The leak, measured first (`test_demonstration_*`): after a durable run, a plain Temporal client
with nothing but the server address fetches the history, base64-decodes its payloads, and reads
the prompt and every tool result. The journal file has the tool result in plaintext too.

The fix: with FDE_PAYLOAD_KEY set, every payload is AES-256-GCM ciphertext before it leaves the
worker, and the journal's content column is sealed the same way. The same outsider now gets
`binary/encrypted` blobs. A worker with the key still resumes a run, serves retries from the
journal, and returns the answer to a client with the key.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
from pathlib import Path

import pytest

temporalio = pytest.importorskip("temporalio")
from temporalio.client import Client  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from fde_kit import MockProvider, ToolRegistry  # noqa: E402
from fde_kit.durable.client import start_run  # noqa: E402
from fde_kit.durable.crypto import (  # noqa: E402
    ENCODING,
    PayloadCipher,
    PayloadKeyError,
    data_converter,
)
from fde_kit.durable.journal import FileJournal, PostgresJournal  # noqa: E402
from fde_kit.durable.models import ToolCallOutput  # noqa: E402
from fde_kit.durable.worker import run_worker  # noqa: E402

SECRET = "Summit's confidential: acquisition of Acme closes March 3 for $2.4M."
QUESTION = "What's the Acme deal?"
ANSWER = "It closes March 3."
SCRIPT = [{"tool_calls": [{"name": "search_documents", "input": {"query": "acme"}}]}, ANSWER]
KEY = PayloadCipher.generate().export_current()
PG_URL = os.environ.get("FDE_TEST_PG_URL")


def _registry(tenant: str | None = "summit") -> ToolRegistry:
    reg = ToolRegistry(tenant_id=tenant)

    @reg.tool
    def search_documents(query: str) -> str:
        """Search."""
        return SECRET

    return reg


@pytest.fixture(scope="module")
def env():
    async def _start():
        try:
            return await asyncio.wait_for(WorkflowEnvironment.start_local(), timeout=120)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"local Temporal dev server unavailable: {exc}")

    loop = asyncio.new_event_loop()
    e = loop.run_until_complete(_start())
    yield e, loop
    loop.run_until_complete(e.shutdown())
    loop.close()


async def _client(env: WorkflowEnvironment, cipher: PayloadCipher | None) -> Client:
    """A client to the test server with the given codec (None = an outsider)."""
    return await Client.connect(
        env.client.service_client.config.target_host, data_converter=data_converter(cipher)
    )


async def _run(env, queue: str, cipher: PayloadCipher | None, journal_dir: str, tenant="summit"):
    client = await _client(env, cipher)
    stop = asyncio.Event()
    worker = asyncio.create_task(
        run_worker(
            client=client,
            task_queue=queue,
            provider=MockProvider(SCRIPT),
            registry=_registry(tenant),
            journal_dir=journal_dir,
            stop=stop,
        )
    )
    await asyncio.sleep(0.4)
    try:
        handle = await start_run(QUESTION, client=client, task_queue=queue, tenant_id=tenant)
        out = await handle.result()
    finally:
        stop.set()
        await worker
    return handle.id, out


async def _outsider_reads(env, workflow_id: str) -> tuple[str, int]:
    """Everything an outsider can decode from the history without the key: the concatenated
    base64-decoded payload data, and how many payloads were marked encrypted."""
    outsider = await _client(env, None)
    raw = (await outsider.get_workflow_handle(workflow_id).fetch_history()).to_json()
    decoded = b"".join(
        base64.b64decode(m) for m in re.findall(r'"data":\s*"([A-Za-z0-9+/=]+)"', raw)
    ).decode("utf-8", "replace")
    return decoded, raw.count(base64.b64encode(ENCODING).decode())


# ------------------------------------------------------------------ the leak


def test_demonstration_history_and_journal_hold_tenant_content_in_plaintext(
    env, monkeypatch: pytest.MonkeyPatch
):
    """No key configured, an untenanted run (a tenant run is refused). The prompt and the tool
    result are readable by any client with the address, and in the journal file."""
    e, loop = env
    monkeypatch.delenv("FDE_PAYLOAD_KEY", raising=False)
    jdir = tempfile.mkdtemp()
    wf_id, _ = loop.run_until_complete(_run(e, "plain", None, jdir, tenant=None))
    decoded, encrypted = loop.run_until_complete(_outsider_reads(e, wf_id))
    assert SECRET in decoded and QUESTION in decoded and encrypted == 0
    files = "".join(p.read_text() for p in Path(jdir).glob("*.json"))
    assert SECRET in files


# ------------------------------------------------------------------ the fix


def test_outsider_sees_only_ciphertext(env, monkeypatch: pytest.MonkeyPatch):
    e, loop = env
    monkeypatch.setenv("FDE_PAYLOAD_KEY", KEY)
    cipher = PayloadCipher.from_settings(KEY, None)
    jdir = tempfile.mkdtemp()
    wf_id, out = loop.run_until_complete(_run(e, "enc", cipher, jdir))
    assert out.output == ANSWER  # a client holding the key gets the real result
    decoded, encrypted = loop.run_until_complete(_outsider_reads(e, wf_id))
    for text in (SECRET, QUESTION, ANSWER, "Acme", "acme"):
        assert text not in decoded, f"{text!r} readable without the key"
    assert encrypted >= 6  # run input, llm in/out x2, tool in/out, result
    files = "".join(p.read_text() for p in Path(jdir).glob("*.json"))
    assert SECRET not in files and '"content": "enc:' in files


def test_metadata_stays_visible_for_operators(env, monkeypatch: pytest.MonkeyPatch):
    """Encrypting payloads must not hide what an operator needs: activity names, attempts,
    the workflow type. Only payload data is sealed."""
    e, loop = env
    monkeypatch.setenv("FDE_PAYLOAD_KEY", KEY)
    cipher = PayloadCipher.from_settings(KEY, None)
    wf_id, _ = loop.run_until_complete(_run(e, "meta", cipher, tempfile.mkdtemp()))
    outsider = loop.run_until_complete(_client(e, None))
    raw = loop.run_until_complete(outsider.get_workflow_handle(wf_id).fetch_history()).to_json()
    assert "llm_call" in raw and "tool_execute" in raw and "AgentWorkflow" in raw


def test_second_worker_with_the_key_resumes_a_run(env, monkeypatch: pytest.MonkeyPatch):
    """Crash-resume still works through the codec: worker A dies after the first activity,
    worker B (same key, fresh process state) replays the encrypted history and finishes."""
    e, loop = env
    monkeypatch.setenv("FDE_PAYLOAD_KEY", KEY)
    cipher = PayloadCipher.from_settings(KEY, None)
    jdir = tempfile.mkdtemp()

    async def go():
        client = await _client(e, cipher)
        slow = [
            {"tool_calls": [{"name": "wait_then_search", "input": {"query": "acme"}}]},
            ANSWER,
        ]
        reg = ToolRegistry(tenant_id="summit")

        @reg.tool
        def wait_then_search(query: str) -> str:
            """Slow search."""
            import time

            time.sleep(2)
            return SECRET

        stop_a = asyncio.Event()
        a = asyncio.create_task(
            run_worker(
                client=client,
                task_queue="resume",
                provider=MockProvider(slow),
                registry=reg,
                journal_dir=jdir,
                stop=stop_a,
            )
        )
        await asyncio.sleep(0.4)
        handle = await start_run(QUESTION, client=client, task_queue="resume", tenant_id="summit")
        await asyncio.sleep(0.8)  # inside the slow tool
        stop_a.set()
        await a
        stop_b = asyncio.Event()
        b = asyncio.create_task(
            run_worker(
                client=client,
                task_queue="resume",
                provider=MockProvider(slow),
                registry=reg,
                journal_dir=jdir,
                stop=stop_b,
            )
        )
        try:
            out = await asyncio.wait_for(handle.result(), timeout=60)
        finally:
            stop_b.set()
            await b
        return out

    out = loop.run_until_complete(go())
    assert out.output == ANSWER and out.steps[0].tool_results[0]["content"] == SECRET


def test_wrong_key_fails_closed(env, monkeypatch: pytest.MonkeyPatch):
    e, loop = env
    monkeypatch.setenv("FDE_PAYLOAD_KEY", KEY)
    cipher = PayloadCipher.from_settings(KEY, None)
    wf_id, _ = loop.run_until_complete(_run(e, "wrong", cipher, tempfile.mkdtemp()))
    other = loop.run_until_complete(_client(e, PayloadCipher.generate()))
    with pytest.raises(PayloadKeyError):
        loop.run_until_complete(other.get_workflow_handle(wf_id).result())


def test_tenant_run_is_refused_without_a_key(env, monkeypatch: pytest.MonkeyPatch):
    e, loop = env
    monkeypatch.delenv("FDE_PAYLOAD_KEY", raising=False)
    monkeypatch.delenv("FDE_PAYLOAD_KEYS", raising=False)

    async def worker_side():
        stop = asyncio.Event()
        stop.set()
        await run_worker(
            client=await _client(e, None),
            task_queue="refuse",
            provider=MockProvider(SCRIPT),
            registry=_registry("summit"),
            journal_dir=tempfile.mkdtemp(),
            stop=stop,
        )

    with pytest.raises(RuntimeError, match="FDE_PAYLOAD_KEY"):
        loop.run_until_complete(worker_side())
    plain = loop.run_until_complete(_client(e, None))
    with pytest.raises(RuntimeError, match="without payload encryption"):
        loop.run_until_complete(
            start_run(QUESTION, client=plain, task_queue="refuse", tenant_id="summit")
        )


# ------------------------------------------------------------------ cipher and journal


def test_key_rotation_decrypts_old_and_encrypts_with_current():
    old = PayloadCipher.generate()
    old_blob = old.encrypt(b"before rotation")
    rotated = PayloadCipher(keys={"k2": os.urandom(32), "k1": old.keys["k1"]}, current="k2")
    assert rotated.decrypt(old_blob) == b"before rotation"
    assert rotated.encrypt(b"x")[:2] == b"k2"
    retired = PayloadCipher(keys={"k2": rotated.keys["k2"]}, current="k2")
    with pytest.raises(PayloadKeyError, match="no key 'k1'"):
        retired.decrypt(old_blob)


def test_tampered_ciphertext_is_rejected():
    c = PayloadCipher.generate()
    blob = bytearray(c.encrypt(b"payload"))
    blob[-1] ^= 1
    with pytest.raises(PayloadKeyError, match="authentication"):
        c.decrypt(bytes(blob))


@pytest.mark.parametrize("bad", ["not-base64!", base64.b64encode(b"short").decode()])
def test_malformed_keys_are_rejected(bad: str):
    with pytest.raises(PayloadKeyError):
        PayloadCipher.from_settings(bad, None)


def test_keys_setting_parses_rotation_list():
    k1, k2 = os.urandom(32), os.urandom(32)
    c = PayloadCipher.from_settings(
        None, f"new={base64.b64encode(k2).decode()},old={base64.b64encode(k1).decode()}"
    )
    assert c is not None and c.current == "new" and set(c.keys) == {"new", "old"}


def test_file_journal_seals_content_and_round_trips(tmp_path: Path):
    c = PayloadCipher.generate()
    j = FileJournal(tmp_path, cipher=c)
    j.put("k", ToolCallOutput(content=SECRET, is_error=False), tool_name="t", tenant_id="summit")
    on_disk = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert on_disk["content"].startswith("enc:") and SECRET not in on_disk["content"]
    got = j.get("k", tenant_id="summit", tool_name="t")
    assert got is not None and got.content == SECRET and got.from_journal
    with pytest.raises(PayloadKeyError):
        FileJournal(tmp_path, cipher=PayloadCipher.generate()).get("k", tenant_id="summit")


@pytest.mark.skipif(not PG_URL, reason="FDE_TEST_PG_URL not set")
def test_postgres_journal_seals_content_and_round_trips():
    import psycopg

    c = PayloadCipher.generate()
    j = PostgresJournal(PG_URL, cipher=c)  # type: ignore[arg-type]
    key = f"enc-test-{os.urandom(4).hex()}"
    j.put(key, ToolCallOutput(content=SECRET, is_error=False), tool_name="t", tenant_id="summit")
    with psycopg.connect(PG_URL) as conn:  # type: ignore[arg-type]
        rows = conn.execute(
            "SELECT content FROM tool_journal WHERE idempotency_key LIKE %s", (f"%{key}",)
        ).fetchall()
    assert rows and all(r[0].startswith("enc:") and SECRET not in r[0] for r in rows)
    got = j.get(key, tenant_id="summit", tool_name="t")
    assert got is not None and got.content == SECRET


def test_unencrypted_journal_row_is_refused_once_a_key_is_set(tmp_path: Path):
    """Turning encryption on after plaintext rows exist: those rows are not silently served."""
    FileJournal(tmp_path).put("k", ToolCallOutput(content=SECRET, is_error=False), tenant_id="t")
    with pytest.raises(PayloadKeyError, match="not encrypted"):
        FileJournal(tmp_path, cipher=PayloadCipher.generate()).get("k", tenant_id="t")
