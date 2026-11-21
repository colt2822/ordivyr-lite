"""Error types and process exit codes shared by the CLI and the job protocol."""

from __future__ import annotations


class LiteError(RuntimeError):
    """Base class for expected, user-facing Lite failures."""

    exit_code = 8


class ConfigError(LiteError):
    exit_code = 2


class JobError(LiteError):
    exit_code = 2


class PermissionDenied(LiteError):
    exit_code = 3


class SandboxError(LiteError):
    exit_code = 3


class ProviderError(LiteError):
    """A model endpoint failed.  Retryable failures can be routed to a fallback."""

    exit_code = 4

    def __init__(self, message: str, *, code: str = "provider_error", status: int | None = None,
                 retryable: bool = False, endpoint: str | None = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable
        self.endpoint = endpoint

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "status": self.status,
                "retryable": self.retryable, "endpoint": self.endpoint}


class BudgetError(LiteError):
    exit_code = 5


class ValidationError(LiteError):
    exit_code = 6


class Cancelled(LiteError):
    exit_code = 7


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PERMISSION = 3
EXIT_PROVIDER = 4
EXIT_BUDGET = 5
EXIT_INCOMPLETE = 6
EXIT_CANCELLED = 7
EXIT_INTERNAL = 8


def exit_code_for(stop_reason: str | None) -> int:
    """Map a loop stop reason onto a stable process exit code."""
    if stop_reason in (None, "done"):
        return EXIT_OK
    if stop_reason == "cancelled":
        return EXIT_CANCELLED
    if stop_reason in ("budget", "cost", "unpriced_model"):
        return EXIT_BUDGET
    if stop_reason in ("provider_error", "auth", "rate_limit", "quota", "network", "timeout"):
        return EXIT_PROVIDER
    if stop_reason in ("permission", "refused", "paid_not_authorized"):
        return EXIT_PERMISSION
    if stop_reason in ("malformed", "truncated", "repeat_failure", "unverified", "steps", "tool_calls",
                       "time", "notools", "invalid", "empty_reply"):
        return EXIT_INCOMPLETE
    return EXIT_INTERNAL
