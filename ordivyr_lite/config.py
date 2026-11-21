"""Configuration, provider presets, BYOK resolution and model-aware limits.

Design rules:
  * no secret is ever stored in the repository or printed;
  * a non-loopback endpoint is PAID and fails closed: it needs an explicit
    authorisation plus positive spend caps before the first request;
  * local (loopback) inference is free and needs no key;
  * model output limits are derived from the model id, never a single global
    constant, and can be lowered by a caller but not silently raised.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .errors import ConfigError
from .util import ensure_private_dir, is_loopback, redact_key

DEFAULT_OUTPUT_TOKENS = 16_384
MIN_OUTPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 131_072
DEFAULT_COMPACT_AT = 48_000
DEFAULT_LIMITS = {
    "max_turns": 64,
    "max_tool_calls": 400,
    "max_seconds": 1800,
    "max_cost_usd": 0.0,
    "max_tokens": 0,
    "max_repairs": 2,
}
DEFAULT_PERMISSIONS = {"profile": "workspace", "sandbox": "auto", "network": False}

# label, protocol, base url, needs-key, common env vars
PRESETS: dict[str, dict[str, Any]] = {
    "openai": {"label": "OpenAI", "protocol": "openai", "base_url": "https://api.openai.com/v1",
               "key_env": ("OPENAI_API_KEY",)},
    "anthropic": {"label": "Anthropic", "protocol": "anthropic", "base_url": "https://api.anthropic.com",
                  "key_env": ("ANTHROPIC_API_KEY",)},
    "deepseek": {"label": "DeepSeek", "protocol": "openai", "base_url": "https://api.deepseek.com/v1",
                 "key_env": ("DEEPSEEK_API_KEY",)},
    "openrouter": {"label": "OpenRouter", "protocol": "openai", "base_url": "https://openrouter.ai/api/v1",
                   "key_env": ("OPENROUTER_API_KEY",)},
    "groq": {"label": "Groq", "protocol": "openai", "base_url": "https://api.groq.com/openai/v1",
             "key_env": ("GROQ_API_KEY",)},
    "mistral": {"label": "Mistral", "protocol": "openai", "base_url": "https://api.mistral.ai/v1",
                "key_env": ("MISTRAL_API_KEY",)},
    "xai": {"label": "xAI", "protocol": "openai", "base_url": "https://api.x.ai/v1",
            "key_env": ("XAI_API_KEY",)},
    "together": {"label": "Together AI", "protocol": "openai", "base_url": "https://api.together.xyz/v1",
                 "key_env": ("TOGETHER_API_KEY",)},
    "google": {"label": "Google Gemini (OpenAI-compatible)", "protocol": "openai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY")},
    "ollama": {"label": "Ollama (local)", "protocol": "openai", "base_url": "http://127.0.0.1:11434/v1",
               "key_env": ()},
    "lmstudio": {"label": "LM Studio (local)", "protocol": "openai", "base_url": "http://127.0.0.1:1234/v1",
                 "key_env": ()},
    "llamacpp": {"label": "llama.cpp server (local)", "protocol": "openai", "base_url": "http://127.0.0.1:8080/v1",
                 "key_env": ()},
    "custom": {"label": "Custom OpenAI-compatible endpoint", "protocol": "openai", "base_url": "",
               "key_env": ()},
}

# Known small output ceilings (substring of the lower-cased model id).  Anything
# unknown gets DEFAULT_OUTPUT_TOKENS; a provider that still refuses is handled
# by bounded halving in the loop.
_OUT_CAPS = (
    ("gpt-3.5", 4096), ("gpt-4-turbo", 4096), ("gpt-4-1106", 4096), ("gpt-4-0125", 4096),
    ("gpt-4-0613", 8192), ("gpt-4-32k", 8192), ("gpt-4o-mini", 16384), ("gpt-4o", 16384),
    ("gpt-4.1-mini", 32768), ("gpt-4.1", 32768), ("o1-mini", 65536), ("o3-mini", 100000),
    ("claude-3-5", 8192), ("claude-3-haiku", 4096), ("claude-3-opus", 4096), ("claude-3-sonnet", 4096),
    ("claude-3-7", 8192), ("claude-sonnet-4", 32000), ("claude-opus-4", 32000), ("claude-2", 4096),
    ("claude-instant", 4096), ("deepseek-chat", 8192), ("deepseek-reasoner", 8192),
    ("qwen2.5", 8192), ("qwen3", 8192), ("llama-3", 8192), ("mistral", 8192),
)

_KEY_ENV = {name: p["key_env"][0] for name, p in PRESETS.items() if p.get("key_env")}
_KEY_ENV["google"] = "GEMINI_API_KEY"


def config_home() -> Path:
    return ensure_private_dir(Path(os.environ.get("ORDIVYR_LITE_HOME") or (Path.home() / ".ordivyr-lite")))


def default_config() -> dict:
    return {
        "version": 1,
        "default_model": None,
        "providers": {},
        "prices": {},
        "limits": dict(DEFAULT_LIMITS),
        "permissions": dict(DEFAULT_PERMISSIONS),
        "validate": "auto",
        "telemetry": True,
    }


def load_config(path: str | os.PathLike | None = None) -> dict:
    home = config_home()
    cfg_path = Path(path).expanduser() if path else home / "config.json"
    cfg = default_config()
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read config {cfg_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"config {cfg_path} must be a JSON object")
        cfg.update(data)
    for key, value in default_config().items():
        if key not in cfg:
            cfg[key] = value
    cfg["_path"] = str(cfg_path)
    cfg["_home"] = str(home)
    return cfg


def save_config(cfg: dict, path: str | os.PathLike | None = None) -> Path:
    home = config_home()
    cfg_path = Path(path).expanduser() if path else home / "config.json"
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tmp = cfg_path.with_name(f".{cfg_path.name}.tmp")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(clean, indent=2) + "\n", "utf-8")
    os.replace(tmp, cfg_path)
    try:
        os.chmod(cfg_path, 0o600)
    except OSError:
        pass
    return cfg_path


def secrets_path() -> Path:
    return config_home() / "secrets.json"


def _load_secrets() -> dict:
    p = secrets_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    providers = data.get("providers")
    if isinstance(providers, dict):
        return {str(k): str(v) for k, v in providers.items()}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def store_provider_key(provider: str, key: str) -> Path:
    data = {"providers": _load_secrets()}
    data["providers"][provider] = key
    p = secrets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2) + "\n")
    return p


def read_key_file(path: str | os.PathLike) -> str:
    p = Path(path).expanduser()
    try:
        return p.read_text("utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError) as exc:
        raise ConfigError(f"cannot read API key file {p}: {exc}") from exc


def _pick_key(provider: str, cfg_provider: dict, explicit_env: str | None,
              explicit_file: str | None) -> tuple[str, str | None]:
    if explicit_env:
        key = os.environ.get(explicit_env, "")
        if not key:
            raise ConfigError(f"environment variable {explicit_env} is empty")
        return key, f"env:{explicit_env}"
    if explicit_file:
        return read_key_file(explicit_file), f"file:{explicit_file}"
    env_name = cfg_provider.get("key_env") or _KEY_ENV.get(provider)
    if env_name:
        if isinstance(env_name, str):
            env_name = (env_name,)
        for name in env_name:
            key = os.environ.get(str(name), "")
            if key:
                return key, f"env:{name}"
    key_file = cfg_provider.get("key_file")
    if key_file:
        return read_key_file(key_file), f"file:{key_file}"
    key = _load_secrets().get(provider, "")
    if key:
        return key, "secrets.json"
    return "", None


_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]{1,64}$")
# Headers a credential may arrive in when the provider does not declare one.
_AUTH_HEADERS = ("authorization", "x-api-key")


def normalize_headers(raw: Any) -> dict[str, str]:
    """Validate configured request headers; "@NAME" values come from the environment.

    A credential belongs in the environment or a key file, never in argv or a
    committed config, so a header value of the form @ENV_NAME is resolved at
    resolve time and a missing variable fails closed instead of sending blank.
    """
    if not raw:
        return {}
    items = raw.items() if isinstance(raw, dict) else raw
    out: dict[str, str] = {}
    for name, value in items:
        key = str(name).strip()
        if not _HEADER_NAME.match(key):
            raise ConfigError(f"invalid header name {key!r}")
        text = str(value)
        if text.startswith("@"):
            env_name = text[1:]
            text = os.environ.get(env_name, "")
            if not text:
                raise ConfigError(f"header {key!r} reads {env_name}, which is empty or unset")
        if "\n" in text or "\r" in text:
            raise ConfigError(f"header {key!r} value must be a single line")
        out[key] = text
    return out


def auth_header_name(headers: dict[str, str], declared: str | None) -> str | None:
    """Header that carries the credential, if an operator configured one."""
    lowered = {name.lower(): name for name in headers}
    if declared:
        found = lowered.get(declared.lower())
        if not found:
            raise ConfigError(f"auth_header {declared!r} is declared but not present in headers")
        return found
    for candidate in _AUTH_HEADERS:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_price(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            in_p = float(value.get("input", value.get("in", 0)))
            out_p = float(value.get("output", value.get("out", 0)))
            cr = float(value.get("cache_read", in_p))
            cw = float(value.get("cache_write", in_p))
            return (in_p, out_p, cr, cw)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            in_p, out_p = float(value[0]), float(value[1])
            cr = float(value[2]) if len(value) > 2 else in_p
            cw = float(value[3]) if len(value) > 3 else in_p
            return (in_p, out_p, cr, cw)
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class Route:
    provider: str
    protocol: str
    base_url: str
    model: str
    api_key: str = ""
    key_source: str | None = None
    paid: bool = False
    authorized: bool = True
    price: tuple[float, float, float, float] | None = None
    caps: dict[str, float] = field(default_factory=dict)
    context_window: int | None = None
    timeout: float = 120.0
    max_retries: int = 2
    headers: dict[str, str] = field(default_factory=dict)
    # Header that carries the credential when the provider does not use a key.
    header_auth: str | None = None

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key) or bool(self.header_auth)

    @property
    def is_local(self) -> bool:
        return not self.paid

    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    def redacted(self) -> dict:
        return {
            "provider": self.provider,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "model": self.model,
            "paid": self.paid,
            "authorized": self.authorized,
            "has_key": self.has_key,
            "has_credentials": self.has_credentials,
            "auth_header": self.header_auth,
            "key_source": self.key_source,
            "key": redact_key(self.api_key) if self.api_key else "",
            "price_usd_per_mtok": self.price,
            "caps": dict(self.caps),
            "context_window": self.context_window,
            "timeout_s": self.timeout,
        }

    def __repr__(self) -> str:  # never leak the key through logs/tracebacks
        return f"Route({self.provider}/{self.model} base={self.base_url} paid={self.paid} key={'set' if self.api_key else 'none'})"


def _merge_provider_cfg(cfg: dict, name: str) -> dict:
    preset = dict(PRESETS.get(name, {}))
    custom = cfg.get("providers", {}).get(name, {}) if isinstance(cfg.get("providers"), dict) else {}
    preset.update(custom or {})
    return preset


def resolve_route(cfg: dict, *, provider: str | None = None, model: str | None = None,
                  base_url: str | None = None, protocol: str | None = None,
                  api_key_env: str | None = None, api_key_file: str | None = None,
                  context_window: int | None = None, allow_paid: bool = False,
                  max_cost_usd: float = 0.0, timeout: float | None = None,
                  headers: Any = None) -> Route:
    provider = (provider or "").strip() or None
    model = (model or "").strip() or None

    if not provider and model and "/" in model:
        head, rest = model.split("/", 1)
        if head in PRESETS or head in (cfg.get("providers") or {}):
            provider, model = head, rest
    if not provider:
        default = cfg.get("default_model")
        if isinstance(default, str) and "/" in default:
            provider, dmodel = default.split("/", 1)
            model = model or dmodel
        elif isinstance(default, dict):
            provider = provider or default.get("provider")
            model = model or default.get("model")
    if not provider:
        provider = "custom" if base_url else None
    if not provider:
        raise ConfigError("no provider selected: use --provider or set default_model in config")
    if not model:
        raise ConfigError("no model selected: use --model")

    pcfg = _merge_provider_cfg(cfg, provider)
    proto = (protocol or pcfg.get("protocol") or "openai").lower()
    if proto not in ("openai", "anthropic"):
        raise ConfigError(f"unsupported protocol {proto!r}; Lite speaks openai and anthropic")
    base = (base_url or pcfg.get("base_url") or "").strip().rstrip("/")
    if not base:
        raise ConfigError(f"provider {provider!r} needs a base URL (--base-url)")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError("base URL must start with http:// or https://")
    if parsed.scheme == "http" and not is_loopback(parsed.hostname) and not pcfg.get("local"):
        raise ConfigError("plain http is only allowed for loopback/local endpoints; use https")
    if parsed.username or parsed.password:
        raise ConfigError("do not put credentials in the base URL")

    paid = bool(pcfg.get("paid")) if "paid" in pcfg else not is_loopback(parsed.hostname)
    # An operator can explicitly mark a non-loopback LAN endpoint free, but a
    # public host is paid unless the config says otherwise.
    key, source = _pick_key(provider, pcfg, api_key_env, api_key_file)
    hdrs = normalize_headers(pcfg.get("headers"))
    hdrs.update(normalize_headers(headers))
    header_auth = auth_header_name(hdrs, pcfg.get("auth_header"))
    if header_auth and not key:
        source = f"header:{header_auth}"
    if paid and not key and not header_auth:
        raise ConfigError(
            f"{provider} is a paid endpoint but no credential was found. Set "
            f"{(_KEY_ENV.get(provider) or 'ORDIVYR_LITE_API_KEY')}, use --api-key-env/--api-key-file, "
            "pass --header NAME:VALUE, or run `ordivyr-lite config set-key`.")

    authorized = True if not paid else bool(pcfg.get("authorized", False))
    caps = dict(pcfg.get("caps") or {})
    if paid and allow_paid and max_cost_usd and max_cost_usd > 0:
        authorized = True
        caps = {"per_request_usd": float(max_cost_usd), "daily_usd": float(max_cost_usd),
                "total_usd": float(max_cost_usd)}
    if paid and not authorized:
        raise ConfigError(
            f"{provider} is paid and not authorised. Re-run with --allow-paid --max-cost USD, "
            f"or run `ordivyr-lite config authorize {provider} --daily USD --total USD`.")
    if paid and authorized:
        for label in ("per_request_usd", "daily_usd", "total_usd"):
            try:
                caps[label] = float(caps.get(label) or 0)
            except (TypeError, ValueError):
                caps[label] = 0.0
        if not (caps["per_request_usd"] > 0 and caps["daily_usd"] > 0 and caps["total_usd"] > 0):
            raise ConfigError(f"{provider} paid caps must all be positive USD values")

    price = parse_price((cfg.get("prices") or {}).get(model)) or parse_price((pcfg.get("prices") or {}).get(model))
    if paid and price is None:
        raise ConfigError(
            f"no price configured for paid model {model!r}; spending cannot be capped without a price. "
            f"Run `ordivyr-lite config set-price {model} --input USD --output USD`.")
    if price and not paid:
        # local endpoints are free; keep the configured price out of cost math
        price = None

    try:
        cw = int(context_window or pcfg.get("context_window") or 0) or None
    except (TypeError, ValueError):
        cw = None
    try:
        to = float(timeout if timeout is not None else (pcfg.get("timeout_s") or 120))
    except (TypeError, ValueError):
        to = 120.0
    if to <= 0:
        to = 120.0
    retries = int(pcfg.get("max_retries", 2) or 0)
    return Route(provider=provider, protocol=proto, base_url=base, model=model, api_key=key,
                 key_source=source, paid=paid, authorized=authorized, price=price,
                 caps=caps, context_window=cw, timeout=to, max_retries=max(0, min(5, retries)),
                 headers=hdrs, header_auth=header_auth)


def output_limit(model: str | None, override: int | None = None,
                 context_window: int | None = None) -> int:
    """Output-token ceiling to request.

    Precedence: explicit override > env > known small model cap > default.
    A known context window lowers the ceiling to at most half the window so a
    local 16k model is not asked for a 16k completion.
    """
    def num(v: Any) -> int:
        try:
            return int(v) if v not in (None, "") else 0
        except (TypeError, ValueError):
            return 0

    cap = num(override) or num(os.environ.get("ORDIVYR_LITE_MAX_OUTPUT_TOKENS")) or DEFAULT_OUTPUT_TOKENS
    m = str(model or "").lower()
    for needle, value in _OUT_CAPS:
        if needle in m:
            cap = min(cap, value)
            break
    if context_window and context_window > 0:
        cap = min(cap, max(MIN_OUTPUT_TOKENS, context_window // 2))
    return max(MIN_OUTPUT_TOKENS, min(cap, MAX_OUTPUT_TOKENS))


def _safe_has_key(name: str, pcfg: dict) -> bool:
    try:
        return bool(_pick_key(name, pcfg, None, None)[0])
    except ConfigError:
        return False


def config_summary(cfg: dict) -> dict:
    """Redacted, machine-readable config view for `config show` and doctor."""
    providers = []
    for name in sorted(set(PRESETS) | set((cfg.get("providers") or {}))):
        pcfg = _merge_provider_cfg(cfg, name)
        if not pcfg.get("base_url") and name != "custom":
            continue
        providers.append({
            "name": name,
            "label": pcfg.get("label", name),
            "protocol": pcfg.get("protocol", "openai"),
            "base_url": pcfg.get("base_url", ""),
            "paid": bool(pcfg.get("paid", not is_loopback(urlparse(pcfg.get("base_url", "")).hostname))),
            "authorized": bool(pcfg.get("authorized", False)),
            "key_env": pcfg.get("key_env"),
            "has_key": bool(_safe_has_key(name, pcfg)),
            "caps": pcfg.get("caps") or {},
        })
    return {
        "path": cfg.get("_path"),
        "home": cfg.get("_home"),
        "default_model": cfg.get("default_model"),
        "providers": providers,
        "prices": cfg.get("prices") or {},
        "limits": cfg.get("limits") or DEFAULT_LIMITS,
        "permissions": cfg.get("permissions") or DEFAULT_PERMISSIONS,
        "validate": cfg.get("validate", "auto"),
    }


def doctor_report(cfg: dict | None = None) -> dict:
    """Cheap, side-effect-free environment check used by `ordivyr-lite doctor`."""
    import importlib.util
    import shutil
    import sys

    cfg = cfg or load_config()
    checks: dict[str, Any] = {}
    checks["python"] = {"ok": sys.version_info >= (3, 10), "version": sys.version.split()[0]}
    checks["bubblewrap"] = {"ok": bool(shutil.which("bwrap")), "path": shutil.which("bwrap")}
    checks["config_home"] = {"ok": Path(cfg["_home"]).is_dir(), "path": cfg["_home"]}
    checks["secrets_file"] = {"ok": True, "present": secrets_path().exists()}
    checks["pytest"] = {"ok": importlib.util.find_spec("pytest") is not None}
    local = []
    for name in ("ollama", "lmstudio", "llamacpp"):
        pcfg = _merge_provider_cfg(cfg, name)
        local.append({"name": name, "base_url": pcfg.get("base_url")})
    checks["local_presets"] = local
    checks["providers_configured"] = len(cfg.get("providers") or {})
    checks["ok"] = all(v["ok"] for k, v in checks.items() if isinstance(v, dict) and "ok" in v)
    return checks
