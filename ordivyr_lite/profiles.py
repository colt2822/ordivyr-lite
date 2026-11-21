"""Provider adaptation: one engine, four wire dialects.

A Profile describes how a model family wants to be spoken to: whether it
streams, which payload extras improve its prefix cache, whether it can call
tools at all, and which recovery strategy applies when it answers in a shape
the protocol did not ask for.

The loop never branches on a provider name; it asks the profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlparse

FAMILIES = ("openai", "claude", "deepseek", "qwen", "generic")

# Prefix-cache granularity in tokens.  A stable prefix shorter than one block is
# not cached at all, which is why the request builder aligns to it.
CACHE_BLOCK = {"openai": 128, "claude": 1024, "deepseek": 64, "qwen": 32, "generic": 64}
REASONING_FIELD = "reasoning_content"
FENCE = chr(96) * 3
NL = chr(10)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_FENCE_RE = re.compile(FENCE + r"(?:json|tool_call)?\s*(\{.*?\})\s*" + FENCE, re.S)
_KV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Profile:
    family: str = "generic"
    stream: bool = True
    include_usage: bool = True
    parallel_tool_calls: bool | None = None
    cache_key: bool = False
    cache_prompt: bool = True
    tool_choice: str | None = "auto"
    temperature: float | None = None
    max_tokens_field: str = "max_tokens"
    omit_temperature: bool = False
    reasoning_field: str = REASONING_FIELD
    supports_tools: bool = True
    text_tool_calls: bool = True
    disable_thinking: bool = False
    cache_block: int = 64
    extras: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def cache_aware(self) -> bool:
        return self.cache_key or self.cache_prompt or self.family in ("claude", "deepseek")

    def cache_hint(self, system_text: str, task_text: str) -> str:
        """Stable, non-secret routing key: identical tasks share a cache shard."""
        h = hashlib.sha256((system_text or "").encode("utf-8", "replace"))
        h.update(bytes([0]))
        h.update((task_text or "")[:400].encode("utf-8", "replace"))
        return "ol-" + h.hexdigest()[:32]


DEEPSEEK = Profile(
    family="deepseek", stream=True, include_usage=True, cache_key=False,
    cache_prompt=False, parallel_tool_calls=None, cache_block=CACHE_BLOCK["deepseek"],
)
QWEN = Profile(
    family="qwen", stream=True, include_usage=True, cache_key=False, cache_prompt=True,
    parallel_tool_calls=None, disable_thinking=True, cache_block=CACHE_BLOCK["qwen"],
)
CLAUDE = Profile(
    family="claude", stream=True, include_usage=True, cache_key=False, cache_prompt=False,
    parallel_tool_calls=None, cache_block=CACHE_BLOCK["claude"],
)
OPENAI = Profile(
    family="openai", stream=True, include_usage=True, cache_key=True, cache_prompt=False,
    parallel_tool_calls=True, cache_block=CACHE_BLOCK["openai"],
)
GENERIC = Profile(family="generic", stream=True, include_usage=True, cache_key=False,
                  cache_prompt=True, cache_block=CACHE_BLOCK["generic"])

REASONER = Profile(
    family="deepseek", stream=True, include_usage=True, cache_key=False, cache_prompt=False,
    parallel_tool_calls=None, supports_tools=False, text_tool_calls=True,
    cache_block=CACHE_BLOCK["deepseek"],
)
O_SERIES = Profile(
    family="openai", stream=True, include_usage=True, cache_key=True, cache_prompt=False,
    parallel_tool_calls=False, max_tokens_field="max_completion_tokens",
    omit_temperature=True, cache_block=CACHE_BLOCK["openai"],
)


def family_for(model: str = "", provider: str = "", base_url: str = "",
               protocol: str = "openai") -> str:
    m = (model or "").lower()
    p = (provider or "").lower()
    host = (urlparse(base_url or "").hostname or "").lower()
    if protocol == "anthropic" or "anthropic" in host or m.startswith(("claude", "anthropic")):
        return "claude"
    if "deepseek" in host or "deepseek" in m or p == "deepseek":
        return "deepseek"
    if m.startswith("qwen") or "qwen" in m or "qwq" in m or "dashscope" in host:
        return "qwen"
    if m.startswith(("gpt-", "chatgpt", "o1", "o3", "o4")) or "openai.com" in host:
        return "openai"
    return "generic"


def detect(model: str = "", provider: str = "", base_url: str = "",
           protocol: str = "openai") -> Profile:
    fam = family_for(model, provider, base_url, protocol)
    m = (model or "").lower()
    if fam == "deepseek":
        if "reasoner" in m or m.endswith("-r1") or m.split("/")[-1] in ("r1", "deepseek-r1"):
            return REASONER
        return DEEPSEEK
    if fam == "qwen":
        # Qwen3 spends tokens in a thinking channel unless the template is told
        # not to. Qwen2.5 has no such channel, and the extra field makes some
        # llama-server builds reject the request.
        if any(token in m for token in ("qwen3", "qwq", "thinking")):
            return QWEN
        return replace(QWEN, disable_thinking=False)
    if fam == "claude":
        return CLAUDE
    if fam == "openai":
        if re.match(r"^o[134](-|$)", m) or "gpt-5" in m:
            return O_SERIES
        return OPENAI
    return GENERIC


def streaming_enabled(profile: Profile) -> bool:
    """Operator override wins: ORDIVYR_LITE_STREAM=0 disables SSE everywhere."""
    v = (os.environ.get("ORDIVYR_LITE_STREAM") or "").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    return profile.stream


# Ports this fleet actually serves with llama-server. 11434 is also Ollama's
# default; a strict server that rejects cache_prompt is stripped by the
# existing degrade path rather than failing the run.
_LLAMA_PORTS = frozenset({8080, 8181, 11434, 11436, 18435, 18436})


def is_llamacpp(base_url: str, provider: str = "") -> bool:
    if (provider or "").lower() in {"llamacpp", "llama.cpp", "llama-cpp"}:
        return True
    if "llama" in (base_url or "").lower():
        return True
    port = urlparse(base_url or "").port or 0
    return port in _LLAMA_PORTS


def openai_payload(profile: Profile, route, *, messages: list[dict], tools: list[dict] | None,
                   max_tokens: int, temperature: float, stream: bool,
                   cache_hint: str = "") -> dict:
    """The exact JSON body for an OpenAI-compatible endpoint, shaped per family."""
    payload: dict[str, Any] = {"model": route.model, "messages": messages}
    payload[profile.max_tokens_field] = int(max_tokens)
    if not profile.omit_temperature:
        payload["temperature"] = float(temperature)
    if stream:
        payload["stream"] = True
        if profile.include_usage:
            payload["stream_options"] = {"include_usage": True}
    if tools and profile.supports_tools:
        payload["tools"] = tools
        if profile.tool_choice:
            payload["tool_choice"] = profile.tool_choice
        if profile.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = bool(profile.parallel_tool_calls)
    if profile.cache_key and cache_hint:
        payload["prompt_cache_key"] = cache_hint
    if profile.cache_prompt and is_llamacpp(route.base_url, getattr(route, "provider", "")):
        # llama.cpp and sglang reuse the KV prefix only when caching is enabled.
        payload["cache_prompt"] = True
    if profile.disable_thinking and is_llamacpp(route.base_url, getattr(route, "provider", "")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    for key, value in profile.extras:
        payload[key] = value
    return payload


_CORE_FIELDS = ("model", "messages", "max_tokens", "max_completion_tokens", "temperature",
                "stream", "tools", "tool_choice", "system")


_FIELD_REJECT = re.compile(
    r"unknown|unsupported|unrecogni[sz]ed|unexpected|not support|extra (field|input|argument)|"
    r"invalid (field|parameter|argument)|additional propert", re.I)

# A 400 that names the usage-reporting stream field is a field rejection too, even
# when its wording is not in the generic list above.
_STREAM_FIELD = re.compile(r"stream_options|include_usage", re.I)


def has_extras(payload: dict) -> bool:
    """True when the body carries a field a strict endpoint may reject.

    ``stream_options`` counts: the degrade ladder drops it, so refusing it must
    reach that ladder instead of ending the run.
    """
    return any(k not in _CORE_FIELDS for k in payload)


def field_rejection(payload: dict, status: int, raw: bytes) -> bool:
    """A 400-class answer that is about an unknown field, not about the request's size."""
    if status not in (400, 404, 422) or not has_extras(payload):
        return False
    try:
        text = raw.decode("utf-8", "replace")[:600]
    except AttributeError:
        text = str(raw)[:600]
    if status == 404 and "unknown" not in text.lower():
        return False
    return bool(_FIELD_REJECT.search(text) or _STREAM_FIELD.search(text))


def strip_extras(payload: dict) -> dict:
    """The degrade ladder: core fields only, no family-specific extras."""
    return {k: v for k, v in payload.items()
            if k in _CORE_FIELDS and not k.startswith("_ol_")}


_FENCED_CALL_RE = re.compile(r"```(?:python|py|tool)?[ \t]*\n?(.*?)```", re.S)
_CALL_BODY_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\{.*\})\s*\)\s*$", re.S)


def fenced_tool_call(text: str, tool_names: set[str]) -> dict | None:
    """One fenced ``name({...})`` call, with nothing else in the fence.

    Qwen2.5 Coder writes ``write_file({...})`` inside a python fence after it
    was told that a code block does not edit the file. A fence that is a
    program, or a sentence that merely mentions a call, is not this shape.
    """
    fences = list(_FENCED_CALL_RE.finditer(text or ""))
    if len(fences) != 1:
        return None
    match = _CALL_BODY_RE.match(fences[0].group(1).strip())
    if match is None:
        return None
    name, raw_args = match.group(1), match.group(2)
    if name not in tool_names:
        return None
    try:
        args = json.loads(raw_args)
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def bare_tool_call(text: str) -> dict | None:
    """A message that is only one JSON tool call, with nothing around it.

    Qwen2.5 on llama.cpp often writes this instead of the tool_calls field.
    A call quoted inside a sentence is not this shape and is left alone.
    """
    raw = (text or "").strip()
    if len(raw) < 2 or raw[0] != "{" or raw[-1] != "}":
        return None
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool")
    args = obj.get("arguments", obj.get("parameters", obj.get("args")))
    if isinstance(name, dict):
        args = name.get("arguments", args)
        name = name.get("name")
    if not isinstance(name, str) or not _KV_RE.fullmatch(name):
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def text_tool_calls(text: str, limit: int = 8) -> list[dict]:
    """Recover tool calls a model wrote into the text stream instead of the field.

    Only two unambiguous shapes are accepted - an explicit tool_call block and a
    fenced JSON object that names a function - and both must parse as a JSON
    object with a name and an object arguments/parameters field.
    """
    if not text or "{" not in text:
        return []
    found: list[dict] = []
    seen: set[str] = set()
    blocks = [m.group(1) for m in _TOOL_CALL_RE.finditer(text)]
    blocks += [m.group(1) for m in _FENCE_RE.finditer(text)]
    for block in blocks:
        if len(found) >= limit or block in seen:
            continue
        seen.add(block)
        try:
            obj = json.loads(block)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments", obj.get("parameters", obj.get("args")))
        if isinstance(name, dict):
            args = name.get("arguments", args)
            name = name.get("name")
        if not isinstance(name, str) or not _KV_RE.fullmatch(name):
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                continue
        if not isinstance(args, dict):
            continue
        found.append({"name": name, "arguments": args})
    return found


def text_tool_protocol(tool_names: list[str]) -> str:
    """System-prompt appendix for families that cannot accept a tools array."""
    template = ('<tool_call>{"name": "<tool>", "arguments": {"<param>": <value>}}'
                '</tool_call>')
    return (NL + NL + "Tool protocol: this endpoint has no function-calling API. To call a "
            "tool, emit exactly one line of the form" + NL + template + NL +
            "and stop that turn. Available tools: " + ", ".join(tool_names) +
            ". Anything outside a <tool_call> block is a message to the user.")


def variant(profile: Profile, **changes: Any) -> Profile:
    """A copy of a base profile with fields overridden (config-driven tuning)."""
    return replace(profile, **changes)
