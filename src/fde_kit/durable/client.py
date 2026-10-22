"""Client helpers: start a durable agent run, query it, wait for it.

from fde_kit.durable.client import run
result = asyncio.run(run("What is 17 * 23?"))
"""

from __future__ import annotations

import uuid
from typing import Any

from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowHandle

from .models import AgentRunInput, AgentRunOutput, PendingApproval, RunStatus
from .worker import DEFAULT_ADDRESS, DEFAULT_TASK_QUEUE
from .workflow import AgentWorkflow


async def connect(address: str = DEFAULT_ADDRESS) -> Client:
    return await Client.connect(address)


async def start_run(
    prompt: str,
    *,
    client: Client | None = None,
    task_queue: str = DEFAULT_TASK_QUEUE,
    workflow_id: str | None = None,
    max_turns: int = 10,
    system_prompt: str | None = None,
    approval_timeout_seconds: int | None = None,
    require_approval_for: list[str] | None = None,
    tenant_id: str | None = None,
) -> WorkflowHandle[Any, AgentRunOutput]:
    client = client or await connect()
    inp = AgentRunInput(
        prompt=prompt, max_turns=max_turns, system_prompt=system_prompt, tenant_id=tenant_id
    )
    if approval_timeout_seconds is not None:
        inp.approval_timeout_seconds = approval_timeout_seconds
    if require_approval_for:
        inp.require_approval_for = list(require_approval_for)
    return await client.start_workflow(
        AgentWorkflow.run,
        inp,
        id=workflow_id or f"agent-{uuid.uuid4().hex[:12]}",
        task_queue=task_queue,
    )


async def run(prompt: str, **kw: Any) -> AgentRunOutput:
    handle = await start_run(prompt, **kw)
    return await handle.result()


async def status(handle: WorkflowHandle[Any, AgentRunOutput]) -> RunStatus:
    return await handle.query(AgentWorkflow.status)


async def pending(handle: WorkflowHandle[Any, AgentRunOutput]) -> PendingApproval | None:
    return await handle.query(AgentWorkflow.pending_approval)


async def approve(
    handle: WorkflowHandle[Any, AgentRunOutput], tool_use_id: str, *, by: str = "", note: str = ""
) -> None:
    await handle.signal(AgentWorkflow.approve, args=[tool_use_id, by, note])


async def reject(
    handle: WorkflowHandle[Any, AgentRunOutput], tool_use_id: str, *, by: str = "", note: str = ""
) -> None:
    await handle.signal(AgentWorkflow.reject, args=[tool_use_id, by, note])


def handle_for(client: Client, workflow_id: str) -> WorkflowHandle[Any, AgentRunOutput]:
    """Reattach to a run by id from any process: a CLI, a web app, a Slack bot."""
    return client.get_workflow_handle_for(AgentWorkflow.run, workflow_id)


async def history_summary(handle: WorkflowHandle[Any, AgentRunOutput]) -> dict[str, Any]:
    """Count the interesting events in a run's history. Used by the crash demo and tests."""
    counts: dict[str, int] = {}
    activities: list[dict[str, Any]] = []
    scheduled: dict[int, dict[str, Any]] = {}
    async for ev in handle.fetch_history_events():
        name = EventType.Name(ev.event_type)
        counts[name] = counts.get(name, 0) + 1
        if name == "EVENT_TYPE_ACTIVITY_TASK_SCHEDULED":
            attrs = ev.activity_task_scheduled_event_attributes
            scheduled[ev.event_id] = {
                "activity": attrs.activity_type.name,
                "event_id": ev.event_id,
                "attempts": 0,
                "completed": False,
            }
            activities.append(scheduled[ev.event_id])
        elif name == "EVENT_TYPE_ACTIVITY_TASK_STARTED":
            attrs = ev.activity_task_started_event_attributes
            rec = scheduled.get(attrs.scheduled_event_id)
            if rec is not None:
                rec["attempts"] = max(rec["attempts"], attrs.attempt)
        elif name == "EVENT_TYPE_ACTIVITY_TASK_COMPLETED":
            attrs = ev.activity_task_completed_event_attributes
            rec = scheduled.get(attrs.scheduled_event_id)
            if rec is not None:
                rec["completed"] = True
    return {"counts": counts, "activities": activities}
