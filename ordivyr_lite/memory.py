"""Deterministic, stdlib-only memory: a bounded codebase index, relevance-ranked
compression of old tool results, and a persistent per-root context store.

Nothing here calls a model, a network endpoint or a subprocess.  The index never
leaves its root: symlinks are not followed, build output, binary and oversized
files are skipped, and every collection is sorted before it is used, so two runs
over the same tree rank identically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from .util import atomic_write_text, ensure_private_dir, messages_chars, utc_now

INDEX_VERSION = 2
CONTEXT_VERSION = 1

SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next", "target", ".idea",
    ".gradle", ".ordivyr-lite", ".ordivyr-spill", "site-packages", ".cache",
})

BM25_K1 = 1.2
BM25_B = 0.75
HASH_DIM = 512
BM25_WEIGHT = 0.7
COSINE_WEIGHT = 0.3
PATH_BOOST = 0.15
MAX_TOKENS_PER_FILE = 200
BINARY_SNIFF = 8192
SNIPPET_CHARS = 200
OUTLINE_CAP = 1200
RECALL_CAP = 900
MIN_TOOL_KEEP = 320
TOOL_STUB = "[earlier tool result dropped to save context; call the tool again if you need it]"

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_FAMILIES = {"generic": (0.60, 0.30), "shell": (0.35, 0.55), "test": (0.25, 0.65),
             "json": (0.50, 0.40), "diff": (0.55, 0.35)}
_ELIDED_MARK = "chars of an earlier tool result elided"


def _home() -> Path:
    return Path(os.environ.get("ORDIVYR_LITE_HOME") or (Path.home() / ".ordivyr-lite")).expanduser()


def _root_key(root: Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    tail = "\n[... truncated]"
    return text[: max(0, limit - len(tail))] + tail


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _tokenize(text: str) -> list[str]:
    """Lowercased tokens; identifiers also contribute their sub-words."""
    out: list[str] = []
    for match in _WORD_RE.finditer(text):
        raw = match.group(0)
        lowered = raw.lower()
        if len(lowered) >= 2:
            out.append(lowered)
        if "_" in raw or any(c.isupper() for c in raw[1:]):
            for part in re.split(r"_+", raw):
                for piece in _CAMEL_RE.findall(part) or [part]:
                    token = piece.lower()
                    if len(token) >= 2 and token != lowered:
                        out.append(token)
    return out


def _counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _bounded(counts: dict[str, int]) -> dict[str, int]:
    if len(counts) <= MAX_TOKENS_PER_FILE:
        return dict(sorted(counts.items()))
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_TOKENS_PER_FILE]
    return dict(sorted(top))


def _file_counts(rel: str, text: str) -> dict[str, int]:
    counts = _counts(_tokenize(text))
    # Keep path and declaration names ahead of frequent body words. A rare
    # exported symbol otherwise disappears from the 200-token file budget.
    names = re.findall(r"(?:def|class|function|interface|type|struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    priority = _counts(_tokenize(rel + " " + " ".join(names)))
    result = dict(sorted(priority.items())[:MAX_TOKENS_PER_FILE])
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token in result:
            result[token] += count
        elif len(result) < MAX_TOKENS_PER_FILE:
            result[token] = count
    return dict(sorted(result.items()))


def _hash_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HASH_DIM


def _hashed_vector(counts: dict[str, int]) -> list[float]:
    vec = [0.0] * HASH_DIM
    for token in sorted(counts):
        vec[_hash_index(token)] += 1.0 + math.log(counts[token])
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _unit(values: list[float]) -> list[float]:
    top = max(values) if values else 0.0
    if top <= 0.0:
        return [0.0] * len(values)
    return [v / top for v in values]


def _hybrid_scores(query_tokens: list[str], docs: list[tuple[str, dict[str, int], int]]) -> list[float]:
    """BM25 + hashed bag-of-words cosine, each max-normalised, then blended.

    docs is [(key, token_counts, length)] in a deterministic order; the result is
    one score per doc in the same order.
    """
    if not docs or not query_tokens:
        return [0.0] * len(docs)
    terms = sorted(set(query_tokens))
    total = len(docs)
    df = {t: sum(1 for _, counts, _ in docs if t in counts) for t in terms}
    avgdl = (sum(length for _, _, length in docs) / total) or 1.0
    idf = {t: math.log(1.0 + (total - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    bm25: list[float] = []
    for _, counts, length in docs:
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if tf:
                score += idf[term] * (tf * (BM25_K1 + 1.0)) / (
                    tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avgdl))
        bm25.append(score)
    query_vec = _hashed_vector(_counts(query_tokens))
    cosine: list[float] = []
    for i, (_, counts, _) in enumerate(docs):
        cosine.append(_cosine(query_vec, _hashed_vector(counts)) if bm25[i] > 0.0 else 0.0)
    a, b = _unit(bm25), _unit(cosine)
    return [BM25_WEIGHT * a[i] + COSINE_WEIGHT * b[i] for i in range(total)]


def rank_texts(query: str, items: list[tuple[str, str]], k: int) -> list[tuple[str, str, float]]:
    """Rank (key, text) pairs against a query; positive scores only, best first."""
    query_tokens = _tokenize(query or "")
    k = max(0, int(k or 0))
    if not query_tokens or not items or not k:
        return []
    docs = []
    for key, text in items:
        counts = _counts(_tokenize(text or ""))
        docs.append((str(key), counts, sum(counts.values())))
    scores = _hybrid_scores(query_tokens, docs)
    order = sorted(range(len(items)), key=lambda i: (-scores[i], str(items[i][0])))
    out: list[tuple[str, str, float]] = []
    for i in order:
        if len(out) >= k or scores[i] <= 0.0:
            break
        out.append((items[i][0], items[i][1], round(scores[i], 6)))
    return out


def _read_text(path: Path, max_bytes: int) -> str | None:
    """Bounded, symlink-free read of a regular file; None when it cannot be indexed."""
    try:
        if path.is_symlink():
            return None
        size = path.stat().st_size
        if size > max_bytes:
            return None
        if size == 0:
            return ""
        with open(path, "rb") as fh:
            raw = fh.read(max_bytes + 1)
    except OSError:
        return None
    if len(raw) > max_bytes or b"\0" in raw[:BINARY_SNIFF]:
        return None
    return raw.decode("utf-8", "replace")


class CodeIndex:
    """A bounded, incrementally refreshed index of the files under one root."""

    def __init__(self, root, cache_dir=None, max_files: int = 4000, max_bytes: int = 1_000_000):
        self.root = Path(root).expanduser().resolve()
        self.max_files = max(1, int(max_files))
        self.max_bytes = max(1, int(max_bytes))
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else _home() / "index"
        self.cache_path = self.cache_dir / f"{_root_key(self.root)}.json"
        self._files: dict[str, dict] = {}
        self._loaded = False
        self._refreshed = False
        self._cache_loaded = False
        self._skipped_large = 0
        self._skipped_binary = 0
        self._last = {"files": 0, "reindexed": 0, "skipped": 0}

    def refresh(self) -> dict:
        """Reindex only files whose (mtime_ns, size) changed since the cached index."""
        self._ensure_loaded()
        previous = self._files
        current: dict[str, dict] = {}
        reindexed = skipped = 0
        self._skipped_large = self._skipped_binary = 0
        for rel, path, stat_ns, size in self._walk():
            if len(current) >= self.max_files:
                break
            if size > self.max_bytes:
                self._skipped_large += 1
                continue
            old = previous.get(rel)
            if old is not None and old["mtime"] == stat_ns and old["size"] == size and old.get("identity") == self._identity(path):
                current[rel] = old
                skipped += 1
                continue
            text = _read_text(path, self.max_bytes)
            if text is None:
                self._skipped_binary += 1
                continue
            current[rel] = {"mtime": stat_ns, "size": size,
                            "identity": self._identity(path),
                            "counts": _file_counts(rel, text)}
            reindexed += 1
        self._files = current
        self._refreshed = True
        self._last = {"files": len(current), "reindexed": reindexed, "skipped": skipped}
        if current != previous or not self.cache_path.exists():
            self._save()
        return dict(self._last)

    def search(self, query: str, k: int = 8) -> list[dict]:
        """Rank indexed files for a query; [] for an empty or useless query."""
        self._ensure_loaded()
        self._ensure_refreshed()
        query_tokens = _tokenize(query or "")
        k = max(0, int(k or 0))
        if not query_tokens or not k or not self._files:
            return []
        rels = sorted(self._files)
        docs = [(rel, self._files[rel]["counts"], sum(self._files[rel]["counts"].values()))
                for rel in rels]
        scores = _hybrid_scores(query_tokens, docs)
        wanted = set(query_tokens)
        ranked: list[tuple[float, str]] = []
        for i, rel in enumerate(rels):
            shared = wanted & set(_tokenize(rel))
            total = scores[i] + PATH_BOOST * (len(shared) / len(wanted))
            if total > 0.0:
                ranked.append((total, rel))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        out: list[dict] = []
        for score, rel in ranked[:k]:
            line, snippet = self._locate(rel, wanted)
            out.append({"path": rel, "score": round(score, 6), "line": line, "snippet": snippet})
        return out

    def outline(self, limit: int = 40) -> str:
        """A compact, deterministic project map (at most OUTLINE_CAP characters)."""
        self._ensure_loaded()
        self._ensure_refreshed()
        limit = max(1, int(limit or 1))
        paths = sorted(self._files)
        if not paths:
            return "(no files indexed)"
        groups: dict[str, list[str]] = {}
        for rel in paths:
            parts = rel.split("/")
            if len(parts) > 1:
                groups.setdefault(parts[0] + "/", []).append("/".join(parts[1:]))
            else:
                groups.setdefault(".", []).append(rel)
        order = sorted(groups, key=lambda name: (-len(groups[name]), name))
        lines = [f"{self.root.name or '.'}: {len(paths)} files indexed"]
        used = 0
        done = 0
        for name in order:
            if used >= limit:
                break
            members = groups[name]
            shown = members[: max(0, min(3, limit - used))]
            if not shown:
                break
            line = f"{name} ({len(members)}): {', '.join(shown)}"
            if len(members) > len(shown):
                line += f" +{len(members) - len(shown)}"
            if sum(len(x) + 1 for x in lines) + len(line) > OUTLINE_CAP - 40:
                break
            lines.append(line)
            used += len(shown)
            done += 1
        if done < len(order):
            lines.append(f"... +{len(order) - done} more dirs, {len(paths) - used} more files")
        return _cap("\n".join(lines), OUTLINE_CAP)

    def stats(self) -> dict:
        self._ensure_loaded()
        return {
            "root": str(self.root),
            "files": len(self._files),
            "bytes": sum(int(entry["size"]) for entry in self._files.values()),
            "tokens": sum(len(entry["counts"]) for entry in self._files.values()),
            "cache": str(self.cache_path),
            "cache_loaded": self._cache_loaded,
            "skipped_large": self._skipped_large,
            "skipped_binary": self._skipped_binary,
            "last_refresh": dict(self._last),
        }

    def _walk(self) -> list[tuple[str, Path, int, int]]:
        """Deterministic (rel, path, mtime_ns, size) list; symlinks and build output pruned."""
        entries: list[tuple[str, Path, int, int]] = []
        budget = self.max_files * 4
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in SKIP_DIRS
                                 and not os.path.islink(os.path.join(dirpath, d)))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    info = path.stat()
                except OSError:
                    continue
                try:
                    rel = path.relative_to(self.root).as_posix()
                except ValueError:
                    continue
                entries.append((rel, path, info.st_mtime_ns, info.st_size))
                if len(entries) >= budget:
                    break
            else:
                continue
            break
        entries.sort(key=lambda item: item[0])
        return entries

    def _locate(self, rel: str, wanted: set[str]) -> tuple[int, str]:
        path = self.root / rel
        try:
            real = path.resolve()
        except OSError:
            return 0, ""
        if real != self.root and self.root not in real.parents:
            return 0, ""
        text = _read_text(path, self.max_bytes)
        if not text:
            return 0, ""
        best_line, best_hits, best_text = 0, 0, ""
        for number, line in enumerate(text.split("\n"), 1):
            hits = len(wanted & set(_tokenize(line)))
            if hits > best_hits:
                best_line, best_hits, best_text = number, hits, line
        if not best_hits:
            return 0, ""
        return best_line, _clip(best_text, SNIPPET_CHARS)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_cache()

    @staticmethod
    def _identity(path: Path) -> list[int] | None:
        try:
            info = path.stat()
            return [info.st_dev, info.st_ino, info.st_ctime_ns]
        except OSError:
            return None

    def _ensure_refreshed(self) -> None:
        # Every retrieval observes the current tree, including external edits and
        # branch switches. refresh stats files but reads only changed contents.
        self.refresh()

    def _load_cache(self) -> None:
        try:
            raw = json.loads(self.cache_path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict) or raw.get("version") != INDEX_VERSION:
            return
        if raw.get("root") != str(self.root) or not isinstance(raw.get("files"), dict):
            return
        files: dict[str, dict] = {}
        for rel, entry in raw["files"].items():
            if not isinstance(rel, str) or not rel or rel.startswith("/"):
                continue
            if ".." in Path(rel).parts or not isinstance(entry, dict):
                continue
            counts = entry.get("counts")
            if not isinstance(counts, dict):
                continue
            clean: dict[str, int] = {}
            for token, value in counts.items():
                if isinstance(token, str) and isinstance(value, (int, float)) and value > 0:
                    clean[token] = int(value)
            try:
                files[rel] = {"mtime": int(entry["mtime"]), "size": int(entry["size"]),
                              "counts": clean, "identity": entry.get("identity")}
            except (KeyError, TypeError, ValueError):
                continue
        self._files = files
        self._last = {"files": len(files), "reindexed": 0, "skipped": len(files)}
        self._refreshed = True
        self._cache_loaded = True

    def _save(self) -> None:
        payload = {"version": INDEX_VERSION, "root": str(self.root), "files": self._files}
        try:
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        try:
            ensure_private_dir(self.cache_dir)
            atomic_write_text(self.cache_path, text, 0o600, fsync=False)
        except OSError:
            pass


class ContextStore:
    """Per-root, cross-run memory of what earlier runs did, ranked by task text."""

    MAX_ENTRIES = 200

    def __init__(self, root, home=None):
        self.root = Path(root).expanduser().resolve()
        base = Path(home).expanduser() if home else _home()
        self.home = base
        self.path = base / "context" / f"{_root_key(self.root)}.json"

    def remember(self, *, run_id, task, files, errors, outcome) -> None:
        if not _names(files) and not _names(errors) and str(outcome) == "done":
            return
        entries = self._load()
        entries.append({
            "run_id": _clip(run_id, 120),
            "task": _clip(task, 400),
            "files": _names(files),
            "errors": _names(errors),
            "outcome": _clip(outcome, 200),
            "at": utc_now(),
        })
        entries = entries[-self.MAX_ENTRIES:]
        payload = {"version": CONTEXT_VERSION, "root": str(self.root), "entries": entries}
        try:
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            ensure_private_dir(self.path.parent)
            atomic_write_text(self.path, text, 0o600, fsync=False)
        except OSError:
            pass

    def recall(self, task: str, limit: int = 3) -> str:
        """A compact digest of the most relevant earlier runs, or "" when nothing matches."""
        query = str(task or "").strip()
        limit = max(0, int(limit or 0))
        if not query or not limit:
            return ""
        entries = self._load()
        if not entries:
            return ""
        items = [(str(i), _entry_text(entry)) for i, entry in enumerate(entries)]
        # One shared word is not a memory. Two keeps "invoice vat" and drops "pong".
        items = [(key, text) for key, text in items if _shared_tokens(query, text) >= 2]
        ranked = rank_texts(query, items, limit)
        if not ranked:
            return ""
        lines = [f"earlier runs on this root ({len(ranked)} of {len(entries)}):"]
        used = len(lines[0])
        for key, _text, _score in ranked:
            entry = entries[int(key)]
            parts = [f"- {entry['run_id'] or 'run'}"]
            if entry["outcome"]:
                parts.append(entry["outcome"])
            if entry["task"]:
                parts.append("task: " + _clip(entry["task"], 160))
            if entry["files"]:
                parts.append("files: " + ", ".join(entry["files"][:4]))
            if entry["errors"]:
                parts.append("errors: " + _clip(entry["errors"][0], 100))
            line = " | ".join(parts)
            if used + len(line) + 1 > RECALL_CAP:
                break
            lines.append(line)
            used += len(line) + 1
        if len(lines) == 1:
            return ""
        return _cap("\n".join(lines), RECALL_CAP)

    def as_dict(self) -> dict:
        entries = self._load()
        return {"root": str(self.root), "store": str(self.path),
                "count": len(entries), "entries": entries}

    def _load(self) -> list[dict]:
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(raw, dict) or raw.get("version") != CONTEXT_VERSION:
            return []
        if raw.get("root") != str(self.root) or not isinstance(raw.get("entries"), list):
            return []
        entries: list[dict] = []
        for item in raw["entries"]:
            if not isinstance(item, dict):
                continue
            entries.append({
                "run_id": str(item.get("run_id") or ""),
                "task": str(item.get("task") or ""),
                "files": _names(item.get("files")),
                "errors": _names(item.get("errors")),
                "outcome": str(item.get("outcome") or ""),
                "at": str(item.get("at") or ""),
            })
        return entries


def _names(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [_clip(item, 160) for item in value if isinstance(item, str) and item.strip()][:16]


def _shared_tokens(query: str, text: str) -> int:
    q = set(_tokenize(query))
    if len(q) < 2:
        return 0
    return len(q & set(_tokenize(text)))


def _entry_text(entry: dict) -> str:
    return " ".join([entry["task"], entry["outcome"], " ".join(entry["files"]),
                     " ".join(entry["errors"])])


def _shrink_tool(text: str, need: int, head_ratio: float, tail_ratio: float) -> str | None:
    if _ELIDED_MARK in text:
        return None
    keep = max(MIN_TOOL_KEEP, len(text) - need - 160)
    if keep >= len(text) - 80:
        return None
    head = int(keep * head_ratio)
    tail = int(keep * tail_ratio)
    dropped = len(text) - head - tail
    marker = f"\n[... {dropped} {_ELIDED_MARK}]\n"
    new = text[:head] + marker + (text[-tail:] if tail else "")
    return new if len(new) < len(text) else None


def compress(messages: list[dict], *, budget_tokens: int, protect: int = 3,
             family: str = "generic") -> int:
    """Shrink OLD tool-result contents in place until the history fits the budget.

    Only ``tool`` messages are ever rewritten, and never the first message nor the
    last ``protect`` messages.  Returns the number of characters removed.
    """
    if not isinstance(messages, list) or not messages:
        return 0
    try:
        budget = int(budget_tokens)
    except (TypeError, ValueError):
        return 0
    if budget <= 0:
        return 0
    protect = max(0, int(protect or 0))
    head_ratio, tail_ratio = _FAMILIES.get(str(family or "generic"), _FAMILIES["generic"])
    target = budget * 4
    total = messages_chars(messages)
    if total <= target:
        return 0
    stop = len(messages) - protect
    candidates = [i for i in range(1, stop)
                  if isinstance(messages[i], dict)
                  and messages[i].get("role") == "tool"
                  and isinstance(messages[i].get("content"), str)]
    removed = 0
    for i in candidates:
        if total <= target:
            break
        old = messages[i]["content"]
        new = _shrink_tool(old, total - target, head_ratio, tail_ratio)
        if new is None:
            continue
        messages[i]["content"] = new
        delta = len(old) - len(new)
        removed += delta
        total -= delta
    for i in candidates:
        if total <= target:
            break
        old = messages[i]["content"]
        if old == TOOL_STUB:
            continue
        delta = len(old) - len(TOOL_STUB)
        if delta <= 0:
            continue
        messages[i]["content"] = TOOL_STUB
        removed += delta
        total -= delta
    return removed
