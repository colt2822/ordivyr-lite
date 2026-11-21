"""Automatic validation and repair support.

Syntax validation is always cheap and local.  Test/command validation is
opt-in (``auto`` detects a Python test suite) and its exact output is fed back
to the model for a bounded number of repairs.  A validator never executes
model-authored commands directly; the command is chosen by the operator/config
and runs in the same sandbox as ``run_command``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import site
import sys

try:
    import tomllib
except ImportError:  # Python 3.10
    tomllib = None  # type: ignore[assignment]
from dataclasses import dataclass, field
from pathlib import Path

from . import sandbox as SB
from .filetools import FileToolError
from .verification import manifest, select_tests, failure_kind, change_triggers, condense_report, PARTIAL, SKIP
import hashlib


@dataclass
class ValidationResult:
    ok: bool | None
    report: str = ""
    kind: str = "none"
    tests_run: bool = False
    command: str | None = None
    evidence: dict = field(default_factory=dict)

    def concise(self, limit: int = 1600) -> str:
        return condense_report(self.report, limit)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind, "tests_run": self.tests_run,
                "command": self.command, "report": self.report[:2000],
                "evidence": self.evidence}


def _syntax_check(box, written: list[str]) -> tuple[bool | None, str]:
    checked = 0
    for rel in sorted(set(written)):
        try:
            p = box._path(rel)
        except Exception:
            continue
        if not p.is_file():
            continue
        try:
            text = p.read_text("utf-8", "replace")
            if rel.endswith(".py"):
                compile(text, rel, "exec")
                checked += 1
            elif rel.endswith(".json"):
                json.loads(text)
                checked += 1
            elif rel.endswith(".toml") and tomllib is not None:
                tomllib.loads(text)
                checked += 1
        except SyntaxError as exc:
            return False, f"{rel}: SyntaxError: {exc.msg} (line {exc.lineno})\n{exc.text or ''}"
        except ValueError as exc:
            return False, f"{rel}: invalid {Path(rel).suffix.lstrip('.')}: {exc}"
        except OSError as exc:
            return False, f"{rel}: cannot read: {exc}"
    if checked:
        return True, f"syntax ok ({checked} file(s))"
    return None, "no checkable files were written"


def detect_test_command(workdir: Path, trusted_inputs: dict | None = None) -> list[str] | None:
    """Conservative Python-test detection.  Returns argv or None."""
    if (workdir / ".ordivyr-lite-no-tests").exists():
        return None
    for rel, digest in (trusted_inputs or {}).items():
        if rel.endswith('conftest.py') or rel in ('pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg'):
            path = workdir / rel
            if not path.is_file():
                raise ValueError('test_definition_changed: original test configuration disappeared: ' + rel)
    # These names participate directly in pytest's import/plugin discovery.
    # Since the workspace is writable by the agent, treating them as validator
    # configuration would let it replace the validator or rewrite its outcome.
    shadows = [name for name in ("pytest.py", "pytest")
               if (workdir / name).exists()]
    for path in workdir.rglob("conftest.py"):
        if any(part in ('.venv', 'venv', 'node_modules', '.git') for part in path.relative_to(workdir).parts):
            continue
        rel = path.relative_to(workdir).as_posix()
        if (path.is_symlink() or not trusted_inputs or
                trusted_inputs.get(rel) != hashlib.sha256(path.read_bytes()).hexdigest()):
            shadows.append(rel)
    if shadows:
        raise ValueError("agent_pytest_shadow: refusing workspace validation with "
                         + ", ".join(shadows))
    markers = [workdir / "pytest.ini"]
    has_tests = any(p.exists() for p in markers)
    has_tests = has_tests or any(workdir.glob("test_*.py")) or any(workdir.glob("*_test.py"))
    # A generic tests/ directory is also conventional in JavaScript and Go.
    # Only actual Python tests identify it as a Python environment.
    tests_dir = workdir / 'tests'
    if tests_dir.is_dir():
        has_tests = has_tests or any(tests_dir.rglob('test_*.py')) or any(tests_dir.rglob('*_test.py'))
    pyproject = workdir / "pyproject.toml"
    if pyproject.exists():
        try:
            has_tests = has_tests or "pytest" in pyproject.read_text("utf-8", "replace")
        except OSError:
            pass
    if not has_tests:
        return None
    pytest_spec = importlib.util.find_spec("pytest")
    if pytest_spec is None:
        raise ValueError("validator_dependency_missing: pytest is unavailable in the trusted interpreter")
    # The process sandbox intentionally omits the user's ambient environment.
    # Add only Python installation roots so legitimate installed project/test
    # dependencies remain available without inheriting workspace-controlled
    # PYTHONPATH or PATH entries.
    trusted_python_roots = []
    try:
        for root in [*site.getsitepackages(), site.getusersitepackages()]:
            if root and Path(root).is_dir() and root not in trusted_python_roots:
                trusted_python_roots.append(root)
    except (AttributeError, OSError):
        pass
    if pytest_spec.origin:
        origin = Path(pytest_spec.origin).resolve()
        root = str(origin.parent.parent)
        if root not in trusted_python_roots and (origin.parent / "__init__.py").is_file():
            trusted_python_roots.append(root)
    prefix = (["env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
               "PYTHONPATH=" + os.pathsep.join(trusted_python_roots)]
              if trusted_python_roots else ["env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"])
    test_config = os.devnull
    for name in ('pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg'):
        path = workdir / name
        if path.is_file() and trusted_inputs and name in trusted_inputs:
            text = path.read_text('utf-8', 'replace')
            if name == 'pytest.ini' or '[tool.pytest.ini_options]' in text or '[pytest]' in text or '[tool:pytest]' in text:
                if hashlib.sha256(path.read_bytes()).hexdigest() != trusted_inputs[name]:
                    raise ValueError('test_definition_changed: ' + name)
                test_config = name
                break
    pytest_args = [sys.executable, "-m", "pytest", "-q", "-x", "-o", "addopts=",
                   "-p", "no:cacheprovider", "-c", test_config,
                   "--rootdir", ".", "--import-mode=importlib"]
    return prefix + pytest_args


_NPM_PLACEHOLDER = 'no test specified'


def detect_project_tests(workdir: Path) -> tuple[str, list[str], str] | None:
    """(kind, argv, definition digest) for JavaScript/TypeScript and Go projects.

    The definition digest binds the test command to what was present before the
    task: a run is refused if an agent-writable script definition changes
    (package.json scripts.test, go.mod/go.sum), the same class of risk as the
    pytest shadow-file refusal. Commands run in the sandbox without network.
    """
    pkg = workdir / 'package.json'
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text('utf-8', 'replace'))
        except (OSError, ValueError):
            data = {}
        script = ((data.get('scripts') or {}).get('test') if isinstance(data, dict) else None)
        npm = shutil.which('npm')
        if isinstance(script, str) and script.strip() and _NPM_PLACEHOLDER not in script and npm:
            return ('javascript', [npm, 'test', '--silent', '--ignore-scripts'],
                    hashlib.sha256(script.encode()).hexdigest())
        tsc = workdir / 'node_modules' / '.bin' / 'tsc'
        if (workdir / 'tsconfig.json').is_file() and tsc.is_file():
            return ('typescript', ['./node_modules/.bin/tsc', '--noEmit', '-p', '.'],
                    hashlib.sha256((workdir / 'tsconfig.json').read_bytes()).hexdigest())
    # Bare node:test/assert projects need no package.json or npm dependency.
    node = shutil.which('node')
    if node:
        tests = []
        for directory, dirs, files in os.walk(workdir, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(directory) / d).is_symlink())
            for name in sorted(files):
                if not re.search(r'\.(?:test|spec)\.[cm]?js$', name):
                    continue
                path = Path(directory) / name
                if path.is_file() and not path.is_symlink():
                    tests.append('./' + path.relative_to(workdir).as_posix())
        if tests:
            return ('javascript', [node, '--test', '--', *tests],
                    hashlib.sha256(b'node --test direct-discovery-v1').hexdigest())
    mod = workdir / 'go.mod'
    if mod.is_file() and any(workdir.rglob('*_test.go')):
        go = os.environ.get('ORDIVYR_LITE_GO_BINARY') or shutil.which('go')
        if go:
            digest = hashlib.sha256(mod.read_bytes() + (workdir / 'go.sum').read_bytes()
                                    if (workdir / 'go.sum').is_file() else mod.read_bytes()).hexdigest()
            return ('go', ['env', 'GOFLAGS=-mod=readonly', 'GOPROXY=off', 'GOTOOLCHAIN=local',
                           'CGO_ENABLED=0', 'GOCACHE=/tmp/go-cache', 'GOPATH=/tmp/go-path',
                           go, 'test', './...'], digest)
    return None


def _clean_pycache(root: Path) -> None:
    """A same-second edit can leave a stale .pyc that masks the model's change."""
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def _run_command(argv_or_shell: str | list[str], cwd: Path, *, timeout: int,
                 sandbox_mode: str, runner=None) -> tuple[bool, str, bool]:
    _clean_pycache(cwd)
    if isinstance(argv_or_shell, list):
        command = shlex.join(argv_or_shell)
    else:
        command = argv_or_shell
    try:
        if runner is not None:
            res = runner(command, timeout)
        else:
            res = SB.run(command, cwd, timeout=timeout, network=False, mode=sandbox_mode)
        report = res.text()
        return res.ok, report[-4000:], res.isolated
    except SB.SandboxError as exc:
        return False, f"validator could not run: {exc}", False


class Validator:
    def __init__(self, mode: str = "auto", command: str | None = None,
                 timeout: int = 120, sandbox_mode: str = "auto", baseline_tests: bool = False):
        mode = (mode or "auto").lower()
        if mode not in ("off", "syntax", "auto", "tests", "command"):
            raise ValueError(f"unknown validator mode {mode!r}")
        self.mode = mode
        self.command = command
        self.timeout = int(timeout or 120)
        self.sandbox_mode = sandbox_mode
        self.initial_files = None
        self.baseline_report = None
        self.baseline_tests = baseline_tests
        self.had_tests = False
        self.project_def = None
        self.python_config = None
        self._partial_fingerprint_generation = 0

    def fingerprint(self, box):
        if self.mode in ('auto', 'tests'):
            files = manifest(box.root)
            if PARTIAL in files:
                # A bounded manifest cannot prove content equivalence. Force a
                # fresh validation at every caller, including termination paths.
                self._partial_fingerprint_generation += 1
                return (('partial-validation', self._partial_fingerprint_generation),)
            return tuple(sorted(files.items()))
        return None

    def snapshot(self):
        return {'initial_files': self.initial_files, 'had_tests': self.had_tests,
                'project_def': self.project_def, 'python_config': self.python_config,
                'policy': {'mode': self.mode, 'command': self.command, 'timeout': self.timeout}}

    def restore(self, state):
        # Failure observations are not restored as historical passing evidence.
        self.initial_files = state.get('initial_files')
        self.had_tests = bool(state.get('had_tests'))
        pd = state.get('project_def')
        self.project_def = tuple(pd) if isinstance(pd, (list, tuple)) and len(pd) == 2 else None
        pc = state.get('python_config')
        self.python_config = tuple(pc) if isinstance(pc, (list, tuple)) and len(pc) == 2 else None

    def prepare(self, box):
        """Capture current user state before edits; never reset a user's tree.

        A resumed run deliberately has no claimed pre-task baseline. Command
        validators are not pre-run: arbitrary operator commands may mutate state.
        """
        if self.mode not in ('auto', 'tests'):
            return
        self.initial_files = manifest(box.root)
        try:
            argv = detect_test_command(box.root, self.initial_files)
        except ValueError:
            return
        self.had_tests = bool(argv)
        if argv and '-c' in argv:
            name = argv[argv.index('-c') + 1]
            if name != os.devnull:
                self.python_config = (name, self.initial_files[name])
        if not argv:
            proj = detect_project_tests(box.root)
            if proj:
                self.had_tests = True
                self.project_def = (proj[0], proj[2])
        if not self.baseline_tests:
            return
        if argv:
            ok, report, _ = _run_command(argv, box.root, timeout=self.timeout,
                                       sandbox_mode=self.sandbox_mode,
                                       runner=getattr(box, 'sandbox_run', None))
            if not ok:
                self.baseline_report = report

    def _tests(self, box, written, argv, syn_report, *, focused_only=False):
        current = manifest(box.root)
        changed = set(written)
        if self.initial_files is not None:
            changed.update(p for p in current.keys() | self.initial_files.keys()
                           if current.get(p) != self.initial_files.get(p))
        selection = select_tests(box.root, sorted(changed), current)
        evidence = {'changed': sorted(changed), 'selection': selection.as_dict(),
                    'stages': [], 'baseline_inventory_available': self.initial_files is not None,
                    'baseline_failure_observed': self.baseline_report is not None,
                    'partial_manifest': PARTIAL in current}
        all_tests = {p for p in current if p.endswith('.py') and
                     (Path(p).name.startswith('test_') or Path(p).name.endswith('_test.py'))}
        stages = ([('focused', argv + ['--', *selection.tests])]
                  if focused_only and not selection.broad and set(selection.tests) != all_tests
                  else [('full', argv)])
        evidence['full_suite'] = any(stage == 'full' for stage, _ in stages)
        reports = [syn_report, 'Changed inputs: ' + ', '.join(sorted(changed))[:1000],
                   'Test selection: ' + selection.reason + ' -> ' + ', '.join(selection.tests)[:1000]]
        for stage, command in stages:
            ok, report, isolated = _run_command(command, box.root, timeout=self.timeout,
                                               sandbox_mode=self.sandbox_mode,
                                               runner=getattr(box, 'sandbox_run', None))
            evidence['stages'].append({'stage': stage, 'ok': ok,
                                       'command': shlex.join(command), 'isolated': isolated})
            reports.append(f'[{stage}]\n' + report)
            if not ok:
                evidence['failure_kind'] = failure_kind(report)
                # A matching node outside the selected impact set is only a
                # pre-existing candidate. Never turn a red suite green.
                import re
                nodes = set(re.findall(r'^(?:FAILED|ERROR) (\S+)', report, re.M))
                old = set(re.findall(r'^(?:FAILED|ERROR) (\S+)', self.baseline_report or '', re.M))
                evidence['preexisting_failure_nodes'] = sorted(nodes & old)
                evidence['outside_selected_failure_nodes'] = sorted(
                    n for n in nodes if n.split('::')[0] not in selection.tests)
                return ValidationResult(False, '\n'.join(reports), kind='tests',
                                        tests_run=True, command=shlex.join(command), evidence=evidence)
        if manifest(box.root) != current:
            return ValidationResult(False, 'verification inputs changed while tests ran; inspect and rerun',
                                    kind='tests', tests_run=True, evidence=evidence)
        return ValidationResult(True, '\n'.join(reports), kind='tests', tests_run=True,
                                command=shlex.join(argv), evidence=evidence)

    def _project_tests(self, box, written, proj, syn_report):
        """JavaScript/TypeScript/Go: always the whole suite, never agent-redefined."""
        kind, argv, digest = proj
        current = manifest(box.root)
        changed = sorted(set(written) | ({p for p in current.keys() | self.initial_files.keys()
                                          if current.get(p) != self.initial_files.get(p)}
                                         if self.initial_files is not None else set()))
        evidence = {'project': kind, 'changed': changed, 'stages': [],
                    'triggers': change_triggers(changed), 'partial_manifest': PARTIAL in current,
                    'agent_defined_test_command': self.project_def is None,
                    'baseline_inventory_available': self.initial_files is not None}
        if self.project_def is not None and self.project_def != (kind, digest):
            return ValidationResult(
                False, 'test_definition_changed: the ' + kind + ' test command definition differs from the one '
                'present before the task; refusing to run an agent-redefined check. Restore it or ask the operator.',
                kind='tests', tests_run=False, evidence=evidence)
        ok, report, isolated = _run_command(argv, box.root, timeout=self.timeout, sandbox_mode=self.sandbox_mode,
                                            runner=getattr(box, 'sandbox_run', None))
        evidence['stages'].append({'stage': 'full', 'ok': ok, 'command': shlex.join(argv), 'isolated': isolated})
        if not ok:
            evidence['failure_kind'] = failure_kind(report)
        elif manifest(box.root) != current:
            return ValidationResult(False, 'verification inputs changed while tests ran; inspect and rerun',
                                    kind='tests', tests_run=True, evidence=evidence)
        head = 'Changed inputs: ' + ', '.join(changed)[:600]
        return ValidationResult(ok, '\n'.join([syn_report, head, f'[{kind} full]', report]),
                                kind='syntax' if kind == 'typescript' else 'tests',
                                tests_run=kind != 'typescript', command=shlex.join(argv), evidence=evidence)

    def checkpoint(self, box, written: list[str]) -> ValidationResult:
        """Observe tests before the remaining model budget is exhausted.

        Unchanged code can still contain the task's original defect. This is
        ordinary verification, never recovery/replay of an interrupted command.
        """
        if self.mode == 'auto' and not any(box.has(name) for name in ('write_file', 'edit_file', 'run_command')):
            return ValidationResult(None, 'automatic checkpoint skipped under read-only effective tools', kind='none')
        return self(box, written, force_tests=True)

    def observe(self, box, written: list[str]) -> ValidationResult:
        """Edit feedback may use the impact set; final verification stays full.

        Conservative selectors still run the whole suite. This result must not
        satisfy the cache entry for a final or budget checkpoint validation.
        """
        return self(box, written, focused_only=True)

    def __call__(self, box, written: list[str], *, force_tests: bool = False, focused_only: bool = False) -> ValidationResult:
        if self.mode == "off":
            return ValidationResult(None, kind="none")
        if self.mode in ('auto', 'tests') and self.python_config is not None:
            name, digest = self.python_config
            path = box.root / name
            if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return ValidationResult(False, 'test_definition_changed: captured Python test configuration changed: ' + name,
                                        kind='tests', tests_run=False)
        runner = getattr(box, "sandbox_run", None)
        if self.initial_files is not None:
            current = manifest(box.root)
            written = sorted(set(written) | {p for p in current.keys() | self.initial_files.keys()
                                             if current.get(p) != self.initial_files.get(p)})
        syn_ok, syn_report = _syntax_check(box, written)
        if syn_ok is False:
            return ValidationResult(False, syn_report, kind="syntax")
        if self.mode == "syntax":
            return ValidationResult(syn_ok, syn_report, kind="syntax")
        if self.mode == "command":
            if not self.command:
                return ValidationResult(None, "no validator command configured", kind="none")
            ok, report, isolated = _run_command(self.command, box.root,
                                                timeout=self.timeout,
                                                sandbox_mode=self.sandbox_mode,
                                                runner=runner)
            label = "" if isolated else "\n[sandbox: process (non-isolated) — bwrap unavailable]"
            return ValidationResult(ok, report + label, kind="command", tests_run=True,
                                    command=self.command)
        if self.mode == "tests":
            try:
                argv = detect_test_command(box.root, self.initial_files)
            except ValueError as exc:
                return ValidationResult(False, str(exc), kind="tests", tests_run=False)
            if not argv:
                proj = detect_project_tests(box.root)
                if proj:
                    return self._project_tests(box, written, proj, syn_report)
                if self.had_tests:
                    return ValidationResult(False, 'previously detected test suite disappeared or was disabled', kind='tests')
                return ValidationResult(None, "no test suite detected", kind="none")
            return self._tests(box, written, argv, syn_report, focused_only=focused_only)
        # auto: syntax first, then a detected test suite
        if syn_ok is None and not written and self.initial_files is not None and not force_tests:
            return ValidationResult(None, "no files were written", kind="none")
        try:
            argv = detect_test_command(box.root, self.initial_files)
        except ValueError as exc:
            return ValidationResult(False, syn_report + "\n" + str(exc), kind="tests",
                                    tests_run=False)
        if not argv:
            proj = detect_project_tests(box.root)
            if proj:
                return self._project_tests(box, written, proj, syn_report)
            if self.had_tests:
                return ValidationResult(False, 'previously detected test suite disappeared or was disabled', kind='tests')
            return ValidationResult(syn_ok, syn_report, kind="syntax")
        return self._tests(box, written, argv, syn_report, focused_only=focused_only)
