"""Choose which already-permitted tools are advertised on the first request.

Permissions do not change here. A tool that is not returned can still run if
the profile allows it; the runner counts that as a selection miss. The choice
is a local regex so it cannot cost more than sending the schemas.

A clear read does not pay for write schemas. A prompt that might need a
change, a command, the network, or a tool the operator added keeps the full
permitted set. Operator allow-lists are not narrowed by the caller.
"""

from __future__ import annotations

import re

READ_TOOLS = frozenset({"list_files", "read_file", "grep", "find_files"})
WRITE_TOOLS = frozenset({"edit_file", "write_file"})
SHELL_TOOLS = frozenset({"run_command"})
WEB_TOOLS = frozenset({"fetch_url"})
ALWAYS = frozenset({"delegate", "update_plan"})
BUILTIN = READ_TOOLS | WRITE_TOOLS | SHELL_TOOLS | WEB_TOOLS | ALWAYS

# Checked before the read rule. Any of these keeps every permitted tool.
_BROAD = re.compile(
    r"\b(edit|write|fix|bug|implement|create|patch|refactor|change|update|rename|"
    r"add|remove|delete|replace|modify|insert|append|feature|typo|repair|correct|"
    r"broken|failing|fail|fails|run|command|pytest|lint|build|compile|shell|git|"
    r"tests?|https?|url|fetch|download)\b",
    re.I,
)
_READ = re.compile(
    r"\b(read|open|show|list|grep|find|search|explain|where|locate|inspect|trace)\b",
    re.I,
)


def select_tools(prompt: str, permitted: list[str]) -> tuple[list[str], str]:
    """Return (names in permitted order, reason).

    reason is "full", "read", or "empty". Never returns a name that is not
    permitted. Custom tools (MCP and others) and operator tools such as
    delegate stay on the request even for a read.
    """
    allowed = [name for name in permitted if name]
    if not allowed:
        return [], "empty"
    text = prompt or ""
    named = [name for name in allowed if re.search(rf"\b{re.escape(name)}\b", text)]
    extras = [name for name in allowed if name not in BUILTIN or name in ALWAYS]
    if len(text) > 400 or "```" in text or _BROAD.search(text):
        # A named tool is already inside `allowed`.
        return allowed, "full"
    if _READ.search(text):
        # Drop write schemas only. Shell and web stay when the profile
        # already allowed them; the operator turned those on deliberately.
        chosen = [name for name in allowed
                  if name not in WRITE_TOOLS or name in named or name in extras]
        if chosen and set(chosen) != set(allowed):
            return chosen, "read"
    return allowed, "full"
