"""Explicit terminal handoff to an installed, unmodified native CLI."""
import json
import os
from pathlib import Path
import shutil
import sys

from .errors import ConfigError

CLIENTS={'claude':'Claude Code','codex':'Codex CLI','gemini':'Gemini CLI'}


def inspect(client,workdir):
    if client not in CLIENTS:
        raise ConfigError('Choose claude, codex or gemini.')
    root=Path(workdir).expanduser().resolve()
    if not root.is_dir():
        raise ConfigError('Choose an existing project directory.')
    found=shutil.which(client)
    return {'client':client,'application':CLIENTS[client],
            'installed':bool(found),'executable':str(Path(found).absolute()) if found else None,
            'workdir':str(root),'mode':'native CLI',
            'controls':'The native CLI owns authentication, permissions, billing and session storage. Lite limits do not apply.'}


def launch(client,workdir,check=False):
    state=inspect(client,workdir)
    if check:
        print(json.dumps(state,indent=2))
        return 0 if state['installed'] else 2
    if not state['installed']:
        raise ConfigError(f'{state["application"]} is not on PATH. Install its official CLI, then rerun this command; Lite will not install it or change your login.')
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ConfigError(f'Native CLI handoff needs an interactive terminal. Use `ordivyr-lite native {client} --check` to inspect it without starting a session.')
    print('Starting '+state['application']+' in '+json.dumps(state['workdir'])+'.\n'+state['controls'],file=sys.stderr,flush=True)
    previous=Path.cwd()
    try:
        os.chdir(state['workdir'])
        # No prompt, credential extraction, shell, policy overrides, extra tools,
        # fallback or Lite configuration is inserted into the native client.
        os.execv(state['executable'],[state['executable']])
    except OSError as exc:
        raise ConfigError('Could not start '+state['application']+': '+str(exc)) from exc
    finally:
        os.chdir(previous)
    return 0  # only reached by an exec test double
