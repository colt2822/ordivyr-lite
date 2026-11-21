"""Terminal connection setup. Configuration stays local; inference is never sent here."""
import copy
import getpass
import math
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

from ordivyr_lite.config import PRESETS, load_config, save_config


def choose_model(provider, entry, pending_key=None):
    """An explicit metadata lookup; use existing transport and provider auth."""
    from .config import Route, _pick_key
    from .providers import ModelClient
    print("Use a model ID, or list models to check your connection without inference.")
    names = []
    if ask("Fetch the model list from this endpoint? yes/no", "no").lower() == "yes":
        key = pending_key or _pick_key(provider, entry, None, None)[0]
        route = Route(provider=provider, protocol=entry["protocol"],
                      base_url=entry["base_url"], model="", api_key=key)
        try:
            names = ModelClient(route).list_models(timeout=10)
            print("Connection accepted. Available models:")
            for i, name in enumerate(names[:100], 1):
                print(f"  {i}. {name}")
            if not names:
                print("The endpoint returned no models; enter the ID manually.")
        except Exception:
            # Do not echo untrusted upstream errors which may contain credentials.
            print("Could not fetch models. Check your URL, API key and provider access.")
            print("You can enter a model ID manually; connectivity is unverified.")
    model = ask("Model ID" + (" or listed number" if names else ""))
    if model.isdigit() and 1 <= int(model) <= min(len(names), 100):
        return names[int(model) - 1]
    return model


def ask(label, default=None):
    value = input(label + (f" [{default}]" if default else "") + ": ").strip()
    return value or default or ""


def amount(label, positive=False):
    while True:
        try:
            value = float(ask(label))
            if math.isfinite(value) and (value > 0 if positive else value >= 0):
                return value
        except ValueError:
            pass
        print("Enter a finite " + ("positive" if positive else "nonnegative") + " dollar amount.")


def configure():
    cfg = load_config()
    draft = copy.deepcopy(cfg)
    print("\nWelcome to ORDIVYR Lite — connection setup")
    print("No Elyraon account is needed. Credentials stay on this computer.")
    print("Connect a cloud API or a local model server. Neither is preselected.")
    print("Lite uses API credentials; native CLI logins stay with their own apps.")
    print("To use an installed CLI and its login: ordivyr-lite native claude (or codex / gemini).")
    print("API billing can be separate from your chat or coding subscription.")
    print("Press Ctrl+C at any point to cancel without saving.\n")
    print("1. Choose your connection")
    names = list(PRESETS)
    for index, name in enumerate(names, 1):
        label = "Claude / Anthropic" if name == "anthropic" else PRESETS[name]["label"]
        print(f"  {index:2}. {label} ({name})")
    while True:
        provider = ask("Provider name or number (no default)").lower()
        provider = {"claude": "anthropic", "gemini": "google", "chatgpt": "openai"}.get(provider, provider)
        if provider.isdigit() and 1 <= int(provider) <= len(names):
            provider = names[int(provider) - 1]
        if provider in PRESETS:
            break
        print("Choose one of the providers listed above.")
    entry = draft.setdefault("providers", {}).setdefault(provider, {})
    if provider == "custom":
        base = ask("API base URL", entry.get("base_url"))
    else:
        base = entry.get("base_url") or PRESETS[provider]["base_url"]
        print(f"Selected {PRESETS[provider]['label']}. Using API endpoint: {base}")
    parsed = urlparse(base)
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if not parsed.hostname or parsed.username or parsed.password or parsed.scheme not in ("http", "https"):
        raise ValueError("Use an HTTP(S) API URL without embedded credentials.")
    if not local and parsed.scheme != "https":
        raise ValueError("Cloud API connections require HTTPS.")
    entry.update(base_url=base, protocol=PRESETS[provider]["protocol"], paid=not local)
    if provider == "custom":
        protocol = ask("API protocol: openai or anthropic", "openai")
        if protocol not in ("openai", "anthropic"):
            raise ValueError("Choose openai or anthropic.")
        entry["protocol"] = protocol
    print("\n2. Connect credentials")
    if not local:
        print("Open your provider's developer console, enable API billing if needed, and create an API key.")
        print("A website password or CLI OAuth token is not an API key.")
    key = None
    if local:
        print("Start your model server separately. No model will be downloaded here.")
    print("Use an existing environment variable or key file, paste a key locally, or keep existing credentials.")
    method = ask("Credential method: env / file / paste / keep" + (" / none" if local else ""))
    if method == "env":
        env = ask("Environment variable name", next(iter(PRESETS[provider]["key_env"]), None))
        if not env or not os.environ.get(env):
            raise ValueError("That variable is not set in this terminal. Export it and rerun setup.")
        entry["key_env"] = env
        entry.pop("key_file", None)
    elif method == "file":
        path = Path(ask("Existing API key file path")).expanduser().resolve()
        if not path.is_file() or not path.read_text().strip():
            raise ValueError("The key file is missing or empty.")
        entry["key_file"] = str(path)
        entry.pop("key_env", None)
    elif method == "paste":
        print("The key will be saved unencrypted in an owner-only local file (0600).")
        key = getpass.getpass("API key (hidden): ").strip()
        if not key:
            raise ValueError("No key supplied.")
        entry.pop("key_env", None)
        entry["key_file"] = str(Path(cfg["_home"]) / "keys" / (provider + "-" + str(time.time_ns()) + ".key"))
    elif method != "keep" and not (local and method == "none"):
        raise ValueError("Choose a listed credential method.")
    print("\n3. Choose the model")
    model = choose_model(provider, entry, key)
    if not model:
        raise ValueError("A model ID is required.")
    draft["default_model"] = {"provider": provider, "model": model}
    if not local:
        print("\n4. Set API spending limits")
        print("Copy the current USD prices per million tokens from your provider.")
        print("These rates support Lite's estimates; provider billing remains authoritative.")
        draft.setdefault("prices", {})[model] = {
            "input": amount("Input price per million tokens"),
            "output": amount("Output price per million tokens"),
        }
        entry["caps"] = {
            "per_request_usd": amount("Maximum USD per request", True),
            "daily_usd": amount("Maximum USD per day", True),
            "total_usd": amount("Maximum USD total", True),
        }
        entry["authorized"] = True
    print(f"\nReady to save: {provider} / {model}\nEndpoint: {base}")
    print("Project files are sent to this provider when you submit a task.")
    if not local:
        caps = entry["caps"]
        print(f"Authorize API use: ${caps['per_request_usd']:g}/request, ${caps['daily_usd']:g}/day, ${caps['total_usd']:g} total.")
    if ask("Save" + (" and authorize these limits" if not local else "") + "? Type yes") != "yes":
        print("Cancelled. Nothing saved.")
        return False
    from ordivyr_lite.config import resolve_route
    key_path = Path(entry["key_file"]) if key else None
    if key_path:
        key_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(key + "\n")
    try:
        # Checks configuration only: no provider request or paid inference.
        resolve_route(draft)
        config_path = Path(cfg["_path"])
        if config_path.exists():
            backup = config_path.with_name("config.before-setup-" + str(time.time_ns()) + ".json")
            shutil.copy2(config_path, backup)
            backup.chmod(0o600)
        save_config(draft)
    except BaseException:
        if key_path:
            key_path.unlink(missing_ok=True)
        raise
    print("\nSetup saved locally. Configuration checks passed.")
    print("No inference request was made. Model-list lookup, if selected, only checks metadata access.")
    return True

