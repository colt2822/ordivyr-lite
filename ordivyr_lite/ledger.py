"""Objective ledger: compact, deterministic task state that survives compaction and resume.

The ledger holds what the loop must be able to check without asking a model:
the original objective, hard constraints, an optional milestone plan, facts the
harness itself verified, approaches that already failed, and the declared write
scope.  It is rendered into the compaction checkpoint and persisted in the
session snapshot, so a restart or a full context never costs the assignment.

Nothing here calls a model, and nothing here changes the request prefix between
compactions (the ledger is only rendered when a checkpoint is written).
"""

from __future__ import annotations

import re
from . import requirements as REQ
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable

OBJECTIVE_CHARS = 1_500
ITEM_CHARS = 200
MAX_CONSTRAINTS = 12
MAX_MILESTONES = 20
MAX_VERIFIED = 10
MAX_FAILED = 8
RENDER_BUDGET = 2_400          # characters; about 600 tokens

_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+(.*\S)")
_RULE = re.compile(r"\b(must|must not|never|do not|don't|only|without|exactly|at least|at most)\b", re.I)
STATUSES = ("pending", "doing", "done")


def _clip(text: str, limit: int = ITEM_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def extract_constraints(prompt: str) -> list[str]:
    """Bullet/numbered lines and rule-bearing sentences from the user's prompt."""
    found: list[str] = []
    for line in str(prompt or "").splitlines():
        m = _BULLET.match(line)
        item = m.group(1) if m else (line.strip() if _RULE.search(line) and len(line) < 240 else "")
        item = _clip(item) if item else ""
        if item and item not in found:
            found.append(item)
        if len(found) >= MAX_CONSTRAINTS:
            break
    return found


def eviction_order(pressure: float) -> tuple[str, ...]:
    """Sections to shrink first when the rendered ledger exceeds its budget.

    ``pressure`` is rendered_size / budget (>= 1.0 here).  Return section names
    from FIRST-to-drop to LAST-to-drop out of: "verified", "failed",
    "milestones_done", "milestones_open", "constraints", "scope".  The
    objective is never dropped, and any section you omit is dropped last.
    """
    # TODO(human): choose the eviction order (optionally vary it with pressure).
    return ("verified", "milestones_done", "failed", "constraints", "milestones_open", "scope")


@dataclass
class Ledger:
    objective: str = ""
    constraints: list[str] = field(default_factory=list)
    milestones: list[dict] = field(default_factory=list)    # {"id", "text", "status"}
    verified: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)           # allowed path prefixes; [] = unrestricted
    drift: list[str] = field(default_factory=list)           # recorded violations, newest last
    requirements: list[dict] = field(default_factory=list)   # persistent acceptance checklist

    @classmethod
    def from_prompt(cls, prompt: str, scope: Iterable[str] = ()) -> "Ledger":
        return cls(objective=str(prompt or "").strip()[:OBJECTIVE_CHARS],
                   constraints=extract_constraints(prompt),
                   requirements=REQ.extract(prompt),
                   scope=[_norm(s) for s in scope if str(s).strip()])

    # ------------------------------------------------------------------ updates
    def note_verified(self, fact: str) -> None:
        _push(self.verified, _clip(fact), MAX_VERIFIED)

    def note_failed(self, approach: str) -> None:
        _push(self.failed, _clip(approach), MAX_FAILED)

    def note_drift(self, what: str) -> None:
        _push(self.drift, _clip(what), MAX_FAILED)

    def apply_plan(self, add: str = "", done: str = "", doing: str = "") -> str:
        """Model-facing plan update: ``add`` is one milestone per line; ``done``
        and ``doing`` are comma-separated milestone ids."""
        for line in str(add or "").splitlines():
            line = _BULLET.sub(r"\1", line).strip()
            if line and len(self.milestones) < MAX_MILESTONES:
                self.milestones.append({"id": len(self.milestones) + 1,
                                        "text": _clip(line), "status": "pending"})
        for ids, status in ((doing, "doing"), (done, "done")):
            for tok in re.findall(r"\d+", str(ids or "")):
                hit = next((m for m in self.milestones if m["id"] == int(tok)), None)
                if hit is not None:
                    hit["status"] = status
        return self.plan_line()

    def plan_line(self) -> str:
        done = sum(1 for m in self.milestones if m["status"] == "done")
        return f"plan: {done}/{len(self.milestones)} milestones done"

    def open_milestones(self) -> list[dict]:
        return [m for m in self.milestones if m["status"] != "done"]

    # ------------------------------------------------------------------ scope
    def allows(self, rel_path: str) -> bool:
        if not self.scope:
            return True
        p = _norm(rel_path)
        return any(p == s or p.startswith(s.rstrip("/") + "/") for s in self.scope)

    # ------------------------------------------------------------------ render
    def render(self, budget: int = RENDER_BUDGET) -> str:
        sections = self._sections()
        total = sum(len(t) for t in sections.values())
        if total > budget:
            for name in eviction_order(total / budget):
                if name in sections and sections[name]:
                    sections[name] = _shrink(sections[name])
                    total = sum(len(t) for t in sections.values())
                    if total <= budget:
                        break
        head = f"[ledger] objective: {self.objective}" if self.objective else "[ledger]"
        body = [t for k, t in sections.items() if t]
        out = "\n".join([head] + body)
        return out if len(out) <= budget + OBJECTIVE_CHARS else out[: budget + OBJECTIVE_CHARS]

    def _sections(self) -> dict[str, str]:
        open_m = [f"{m['id']}. [{m['status']}] {m['text']}" for m in self.open_milestones()]
        done_m = [f"{m['id']}. {m['text']}" for m in self.milestones if m["status"] == "done"]
        return {
            "requirements": _block("acceptance checklist (every item must hold before you finish)",
                                   REQ.render(self.requirements).splitlines()),
            "constraints": _block("constraints (still binding)", self.constraints),
            "scope": ("writes allowed only under: " + ", ".join(self.scope)) if self.scope else "",
            "milestones_open": _block("open milestones", open_m),
            "milestones_done": _block("done milestones", done_m),
            "verified": _block("verified by the harness", self.verified),
            "failed": _block("failed approaches (do not repeat)", self.failed + self.drift),
        }

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"objective": self.objective, "constraints": self.constraints,
                "milestones": self.milestones, "verified": self.verified,
                "failed": self.failed, "scope": self.scope, "drift": self.drift,
                "requirements": self.requirements}

    @classmethod
    def from_dict(cls, data: Any) -> "Ledger":
        d = data if isinstance(data, dict) else {}

        def lst(key: str) -> list:
            v = d.get(key)
            return list(v) if isinstance(v, list) else []
        requirements = []
        for original in lst('requirements'):
            if not isinstance(original, dict) or 'id' not in original:
                continue
            item = dict(original)
            checks = list(item.get('checks') or [])
            added = False
            # Older snapshots kept the full sentence but clipped its checks.
            # Restore omitted checks additively; retain existing custom checks
            # and leave the historical source record untouched.
            for parsed in REQ.extract(str(item.get('text') or '')):
                for check in parsed.get('checks', []):
                    if check not in checks:
                        checks.append(check)
                        added = True
            if added:
                item.update(checks=checks, status='open')
            requirements.append(item)
        return cls(objective=str(d.get("objective") or "")[:OBJECTIVE_CHARS],
                   constraints=[str(x) for x in lst("constraints")][:MAX_CONSTRAINTS],
                   milestones=[m for m in lst("milestones") if isinstance(m, dict) and "id" in m][:MAX_MILESTONES],
                   verified=[str(x) for x in lst("verified")][-MAX_VERIFIED:],
                   failed=[str(x) for x in lst("failed")][-MAX_FAILED:],
                   scope=[str(x) for x in lst("scope")],
                   drift=[str(x) for x in lst("drift")][-MAX_FAILED:],
                   requirements=requirements)


def _norm(path: str) -> str:
    text = str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return PurePosixPath(text or ".").as_posix()


def _push(items: list[str], item: str, cap: int) -> None:
    if item and item not in items:
        items.append(item)
        del items[:-cap]


def _block(title: str, items: list[str]) -> str:
    return f"{title}:\n" + "\n".join(f"- {i}" for i in items) if items else ""


def _shrink(text: str) -> str:
    """Keep the heading and the last two lines of a section."""
    lines = text.splitlines()
    return text if len(lines) <= 3 else "\n".join([lines[0], "- …"] + lines[-2:])
