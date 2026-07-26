"""Claim verdicts and the gate that makes a verifier fail loudly.

A verdict is exactly one of VERIFIED / FALSIFIED / BLOCKED.  Nothing here can
turn a toy, skipped or inconclusive result into a pass: a stage records the
*checks* it ran, and `Verdict.gate()` recomputes whether those checks actually
support the recorded status.  If they do not, the process exits non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VERIFIED = "VERIFIED"
FALSIFIED = "FALSIFIED"
BLOCKED = "BLOCKED"
_ALLOWED = {VERIFIED, FALSIFIED, BLOCKED}


@dataclass
class Check:
    """One machine-checkable assertion backing a verdict."""

    name: str
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)
    # A negative control is EXPECTED to fail its scientific condition; `ok` for a
    # control means "it failed for the intended reason", which is the pass.
    is_negative_control: bool = False

    def line(self) -> str:
        tag = "CONTROL" if self.is_negative_control else "CHECK  "
        mark = "ok  " if self.ok else "FAIL"
        return f"  [{tag}] {mark}  {self.name}" + (f"  --  {self.detail}" if self.detail else "")


@dataclass
class Verdict:
    claim_id: str
    title: str
    status: str
    statement: str = ""
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    runtime_s: float = 0.0

    def add(self, name: str, ok: bool, detail: str = "", **data: Any) -> Check:
        c = Check(name=name, ok=bool(ok), detail=detail, data=data)
        self.checks.append(c)
        return c

    def add_control(self, name: str, failed_as_intended: bool, detail: str = "", **data: Any) -> Check:
        c = Check(
            name=name,
            ok=bool(failed_as_intended),
            detail=detail,
            data=data,
            is_negative_control=True,
        )
        self.checks.append(c)
        return c

    # ---------------------------------------------------------------- gate -- #
    def gate(self) -> tuple[bool, str]:
        """Does the recorded evidence actually support the recorded status?"""
        if self.status not in _ALLOWED:
            return False, f"status {self.status!r} is not one of {sorted(_ALLOWED)}"
        if not self.checks:
            return False, "no checks recorded -- a vacuous verdict is never a pass"
        failed = [c.name for c in self.checks if not c.ok]
        if self.status in (VERIFIED, FALSIFIED) and failed:
            return False, f"status {self.status} but these checks failed: {failed}"
        if self.status in (VERIFIED, FALSIFIED) and not any(
            not c.is_negative_control for c in self.checks
        ):
            return False, f"status {self.status} backed only by negative controls"
        if self.status in (VERIFIED, FALSIFIED) and not any(
            c.is_negative_control for c in self.checks
        ):
            return False, f"status {self.status} with no negative control recorded"
        if self.status == BLOCKED and not self.limitations:
            return False, "BLOCKED without a recorded limitation"
        return True, "evidence supports the recorded status"

    def to_dict(self) -> dict:
        ok, reason = self.gate()
        return {
            "claim_id": self.claim_id,
            "title": self.title,
            "statement": self.statement,
            "status": self.status,
            "gate_ok": ok,
            "gate_reason": reason,
            "n_checks": len(self.checks),
            "n_failed": sum(1 for c in self.checks if not c.ok),
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "detail": c.detail,
                    "is_negative_control": c.is_negative_control,
                    "data": c.data,
                }
                for c in self.checks
            ],
            "numbers": self.numbers,
            "notes": self.notes,
            "limitations": self.limitations,
            "deviations": self.deviations,
            "artifacts": self.artifacts,
            "runtime_s": self.runtime_s,
        }

    def print_summary(self) -> None:
        print(f"\n  VERDICT {self.claim_id}: {self.status}", flush=True)
        for c in self.checks:
            print(c.line(), flush=True)
        ok, reason = self.gate()
        print(f"  GATE: {'PASS' if ok else 'FAIL'} -- {reason}", flush=True)
