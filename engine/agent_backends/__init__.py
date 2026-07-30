#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agent_backends - provider-neutral execution seam under the do! endpoint.

`agent_do` prepares the session, the prompt, and the canonical tool
bindings; a backend only converts that request into one provider's wire
shape and runs it. Nothing in this package may append a ledger step, decide
a final result, or check mathematics: those stay with the tactic registry,
the numeric oracle, and the ledger.
"""
from agent_backends.base import (  # noqa: F401  (re-exported seam)
    BUDGET, CAPABILITY, USER,
    BUDGET_EXHAUSTED, CAPABILITY_VIOLATION, COMPLETED, FAILED, INTERRUPTED,
    AgentBudget, AgentOutcome, AgentRequest, CancellationToken,
    DEFAULT_GRACE_PERIOD, DoAgentError, ThreadedRunHandle, ToolBinding,
    ToolDispatcher, ToolArgumentError, UnknownToolError,
    status_for_reason, wait_interruptibly)
