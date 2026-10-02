"""Disk-backed transcript prefix cache for MLX prompt caches."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import (
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    load_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_MANIFEST_FILE = "manifest.sqlite3"
_STALE_AFTER_SECONDS = 60 * 60


@dataclass(slots=True)
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    entry_hash: str | None = None


@dataclass(slots=True)
class CacheEntry:
    key_hash: str
    namespace: str
    parent_hash: str | None
    tokens: tuple[int, ...]
    token_count: int
    file_path: Path
    nbytes: int
    checkpoint_reason: str
    created_at: float
    last_access: float


@dataclass(slots=True)
class CacheLookup:
    prompt_cache: list[Any] | None
    uncached_tail: list[int]
    cached_tokens: int
    loaded_bytes: int
    key_hash: str | None = None


class TranscriptPrefixCacheManager:
    """Persist and reload transcript KV caches from local disk."""

    chunk_size = 256
    stale_after_seconds = _STALE_AFTER_SECONDS

    def __init__(self, cache_dir: Path, max_bytes: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_bytes = max(0, int(max_bytes))
        self.enabled = self.max_bytes > 0
        self._manifest_path = self.cache_dir / _MANIFEST_FILE
        self._lock = threading.RLock()
        self._tries: dict[str, _TrieNode] = {}
        self._entries: dict[str, CacheEntry] = {}
        self._total_bytes = 0
        self.last_lookup: dict[str, Any] | None = None
        self.last_store: dict[str, Any] | None = None
        self.last_eviction: dict[str, Any] | None = None
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_manifest()
        self._load_manifest()

    def close(self) -> None:
        return None

    def wait_for_idle(self, timeout: float = 10.0) -> None:
        del timeout
        return None

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def lookup(self, namespace: str, tokens: list[int]) -> CacheLookup:
        if not self.enabled or not tokens:
            self.last_lookup = {
                "namespace": namespace,
                "hit_tokens": 0,
                "loaded_bytes": 0,
                "checkpoint_reason": "disabled",
            }
            return CacheLookup(prompt_cache=None, uncached_tail=list(tokens), cached_tokens=0, loaded_bytes=0)

        self._evict_expired_entries()

        with self._lock:
            root = self._tries.get(namespace)
            if root is None:
                self.last_lookup = {
                    "namespace": namespace,
                    "hit_tokens": 0,
                    "loaded_bytes": 0,
                    "checkpoint_reason": "miss",
                }
                return CacheLookup(
                    prompt_cache=None,
                    uncached_tail=list(tokens),
                    cached_tokens=0,
                    loaded_bytes=0,
                )
            exact_hash, shorter_hash, longer_hash, common_prefix = self._search(root, tokens)

        if exact_hash is not None and len(tokens) > 1:
            entry = self._load_entry(exact_hash)
            if entry is not None:
                trimmed = trim_prompt_cache(entry, 1)
                if trimmed == 1:
                    result = CacheLookup(
                        prompt_cache=entry,
                        uncached_tail=[tokens[-1]],
                        cached_tokens=len(tokens) - 1,
                        loaded_bytes=self._entries[exact_hash].nbytes,
                        key_hash=exact_hash,
                    )
                    self.last_lookup = {
                        "namespace": namespace,
                        "hit_tokens": result.cached_tokens,
                        "loaded_bytes": result.loaded_bytes,
                        "checkpoint_reason": "exact",
                    }
                    return result

        shorter_len = self._entries[shorter_hash].token_count if shorter_hash is not None else 0
        if longer_hash is not None and common_prefix > shorter_len and len(tokens) > 1:
            entry = self._load_entry(longer_hash)
            if entry is not None:
                prefix_len = min(len(tokens) - 1, common_prefix)
                to_trim = self._entries[longer_hash].token_count - prefix_len
                trimmed = trim_prompt_cache(entry, to_trim)
                if trimmed == to_trim:
                    result = CacheLookup(
                        prompt_cache=entry,
                        uncached_tail=list(tokens[prefix_len:]),
                        cached_tokens=prefix_len,
                        loaded_bytes=self._entries[longer_hash].nbytes,
                        key_hash=longer_hash,
                    )
                    self.last_lookup = {
                        "namespace": namespace,
                        "hit_tokens": result.cached_tokens,
                        "loaded_bytes": result.loaded_bytes,
                        "checkpoint_reason": "longer",
                    }
                    return result

        if shorter_hash is not None:
            entry = self._load_entry(shorter_hash)
            if entry is not None:
                result = CacheLookup(
                    prompt_cache=entry,
                    uncached_tail=list(tokens[shorter_len:]),
                    cached_tokens=shorter_len,
                    loaded_bytes=self._entries[shorter_hash].nbytes,
                    key_hash=shorter_hash,
                )
                self.last_lookup = {
                    "namespace": namespace,
                    "hit_tokens": result.cached_tokens,
                    "loaded_bytes": result.loaded_bytes,
                    "checkpoint_reason": "shorter",
                }
                return result

        self.last_lookup = {
            "namespace": namespace,
            "hit_tokens": 0,
            "loaded_bytes": 0,
            "checkpoint_reason": "miss",
        }
        return CacheLookup(prompt_cache=None, uncached_tail=list(tokens), cached_tokens=0, loaded_bytes=0)

    def enqueue_persist(
        self,
        namespace: str,
        transcript_tokens: list[int],
        prompt_cache: list[Any],
        checkpoint_reason: str,
        parent_tokens: list[int] | None = None,
    ) -> None:
        if not self.enabled or not transcript_tokens:
            return
        self._evict_expired_entries()
        if not self._is_supported_prompt_cache(prompt_cache):
            logger.info("Transcript cache disabled for unsupported prompt cache type")
            return

        key_hash = self._hash_tokens(namespace, transcript_tokens)
        with self._lock:
            if key_hash in self._entries:
                return

        parent_hash = (
            self._hash_tokens(namespace, parent_tokens)
            if parent_tokens
            else None
        )
        self._persist_snapshot(
            key_hash=key_hash,
            namespace=namespace,
            transcript_tokens=tuple(int(t) for t in transcript_tokens),
            prompt_cache=prompt_cache,
            checkpoint_reason=checkpoint_reason,
            parent_hash=parent_hash,
        )

    def invalidate_namespace(self, namespace: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            hashes = [entry.key_hash for entry in self._entries.values() if entry.namespace == namespace]
        for key_hash in hashes:
            self._delete_entry(key_hash)

    def invalidate_namespace_prefix(self, prefix: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            hashes = [
                entry.key_hash for entry in self._entries.values() if entry.namespace.startswith(prefix)
            ]
        for key_hash in hashes:
            self._delete_entry(key_hash)

    def _persist_snapshot(
        self,
        key_hash: str,
        namespace: str,
        transcript_tokens: tuple[int, ...],
        prompt_cache: list[Any],
        checkpoint_reason: str,
        parent_hash: str | None,
    ) -> None:
        file_path = self.cache_dir / f"{key_hash}.safetensors"
        started = time.time()
        try:
            mx.eval([cache.state for cache in prompt_cache])
            save_prompt_cache(
                str(file_path),
                prompt_cache,
                {
                    "schema_version": str(_SCHEMA_VERSION),
                    "namespace": namespace,
                    "checkpoint_reason": checkpoint_reason,
                    "parent_hash": parent_hash or "",
                    "token_count": str(len(transcript_tokens)),
                },
            )
            nbytes = file_path.stat().st_size
            entry = CacheEntry(
                key_hash=key_hash,
                namespace=namespace,
                parent_hash=parent_hash,
                tokens=transcript_tokens,
                token_count=len(transcript_tokens),
                file_path=file_path,
                nbytes=nbytes,
                checkpoint_reason=checkpoint_reason,
                created_at=started,
                last_access=started,
            )
            with self._lock:
                self._entries[key_hash] = entry
                self._add_to_trie(entry)
                self._total_bytes += nbytes
                self.last_store = {
                    "namespace": namespace,
                    "saved_bytes": nbytes,
                    "token_count": len(transcript_tokens),
                    "checkpoint_reason": checkpoint_reason,
                }
            self._write_manifest_entry(entry)
            self._evict_expired_entries(now=started)
            self._evict_to_limit()
        except Exception:
            logger.warning("Failed to persist transcript cache entry", exc_info=True)

    def _load_entry(self, key_hash: str) -> list[Any] | None:
        expired_entry = False
        access_time = time.time()
        with self._lock:
            entry = self._entries.get(key_hash)
            if entry is None:
                return None
            if self._is_expired(entry, access_time):
                expired_entry = True
            else:
                entry.last_access = access_time
        if expired_entry:
            self._delete_entry(key_hash, eviction_reason="ttl")
            return None
        self._touch_manifest_entry(key_hash, access_time)
        try:
            prompt_cache = load_prompt_cache(str(entry.file_path))
        except Exception:
            logger.warning("Failed to load transcript cache entry %s", key_hash, exc_info=True)
            return None
        if not self._is_supported_prompt_cache(prompt_cache):
            return None
        return prompt_cache

    def _evict_to_limit(self) -> None:
        if not self.enabled:
            return
        while True:
            with self._lock:
                if self._total_bytes <= self.max_bytes:
                    return
                leaf = self._oldest_leaf()
            if leaf is None:
                return
            self._delete_entry(leaf.key_hash, eviction_reason="lru")

    def _oldest_leaf(self) -> CacheEntry | None:
        leaves = [entry for entry in self._entries.values() if self._is_leaf(entry)]
        if not leaves:
            return None
        return min(leaves, key=lambda entry: entry.last_access)

    def _is_leaf(self, entry: CacheEntry) -> bool:
        root = self._tries.get(entry.namespace)
        if root is None:
            return True
        current = root
        for token in entry.tokens:
            current = current.children[int(token)]
        return len(current.children) == 0

    def _delete_entry(self, key_hash: str, *, eviction_reason: str = "manual") -> None:
        with self._lock:
            entry = self._entries.pop(key_hash, None)
            if entry is None:
                return
            self._remove_from_trie(entry)
            self._total_bytes = max(0, self._total_bytes - entry.nbytes)
            self.last_eviction = {
                "namespace": entry.namespace,
                "evicted_bytes": entry.nbytes,
                "token_count": entry.token_count,
                "eviction_reason": eviction_reason,
            }
        self._delete_manifest_entry(key_hash)
        try:
            entry.file_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to delete transcript cache file %s", entry.file_path, exc_info=True)

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        return (now - entry.last_access) > self.stale_after_seconds

    def _evict_expired_entries(self, now: float | None = None) -> None:
        if not self.enabled:
            return
        deadline = time.time() if now is None else now
        with self._lock:
            expired_hashes = [
                entry.key_hash for entry in self._entries.values() if self._is_expired(entry, deadline)
            ]
        for key_hash in expired_hashes:
            self._delete_entry(key_hash, eviction_reason="ttl")

    def _is_supported_prompt_cache(self, prompt_cache: list[Any]) -> bool:
        return all(self._is_supported_cache(cache) for cache in prompt_cache)

    def _is_supported_cache(self, cache: Any) -> bool:
        if isinstance(cache, CacheList):
            return all(self._is_supported_cache(sub_cache) for sub_cache in cache.caches)
        if isinstance(cache, RotatingKVCache):
            return False
        return isinstance(cache, (KVCache, QuantizedKVCache))

    def _hash_tokens(self, namespace: str, tokens: list[int] | tuple[int, ...]) -> str:
        payload = json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "namespace": namespace,
                "tokens": list(tokens),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _add_to_trie(self, entry: CacheEntry) -> None:
        root = self._tries.setdefault(entry.namespace, _TrieNode())
        current = root
        for token in entry.tokens:
            current = current.children.setdefault(int(token), _TrieNode())
        current.entry_hash = entry.key_hash

    def _remove_from_trie(self, entry: CacheEntry) -> None:
        root = self._tries.get(entry.namespace)
        if root is None:
            return
        path: list[tuple[int | None, _TrieNode]] = [(None, root)]
        current = root
        for token in entry.tokens:
            child = current.children.get(int(token))
            if child is None:
                return
            path.append((int(token), child))
            current = child
        current.entry_hash = None
        for idx in range(len(path) - 1, 0, -1):
            token, node = path[idx]
            parent = path[idx - 1][1]
            if node.entry_hash is None and not node.children:
                del parent.children[token]
            else:
                break
        if not root.children and root.entry_hash is None:
            self._tries.pop(entry.namespace, None)

    def _search(
        self,
        root: _TrieNode,
        tokens: list[int],
    ) -> tuple[str | None, str | None, str | None, int]:
        current = root
        last_entry_hash: str | None = None
        prev_entry_hash: str | None = None
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current.children:
            current = current.children[tokens[index]]
            if current.entry_hash is not None:
                prev_entry_hash = last_entry_hash
                last_entry_hash = current.entry_hash
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return last_entry_hash, prev_entry_hash, None, len(tokens)

        shorter_hash = last_entry_hash if last_cache_index >= 0 else None
        longer_hash: str | None = None
        common_prefix = index
        if index > 0:
            stack: list[tuple[_TrieNode, int]] = [(current, 0)]
            best: tuple[int, str] | None = None
            while stack:
                node, extra_len = stack.pop()
                if node.entry_hash is not None:
                    if best is None or extra_len < best[0]:
                        best = (extra_len, node.entry_hash)
                for child in node.children.values():
                    stack.append((child, extra_len + 1))
            if best is not None:
                longer_hash = best[1]
        return None, shorter_hash, longer_hash, common_prefix

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._manifest_path)

    def _init_manifest(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key_hash TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    parent_hash TEXT,
                    token_count INTEGER NOT NULL,
                    tokens_json TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    nbytes INTEGER NOT NULL,
                    checkpoint_reason TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_entries_namespace ON cache_entries(namespace)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_entries_last_access ON cache_entries(last_access)"
            )
            conn.commit()

    def _load_manifest(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key_hash, namespace, parent_hash, token_count, tokens_json, file_path,
                       nbytes, checkpoint_reason, created_at, last_access
                FROM cache_entries
                """
            ).fetchall()
        with self._lock:
            self._tries.clear()
            self._entries.clear()
            self._total_bytes = 0
            for row in rows:
                file_path = Path(row[5])
                if not file_path.exists():
                    continue
                entry = CacheEntry(
                    key_hash=row[0],
                    namespace=row[1],
                    parent_hash=row[2],
                    token_count=int(row[3]),
                    tokens=tuple(int(token) for token in json.loads(row[4])),
                    file_path=file_path,
                    nbytes=int(row[6]),
                    checkpoint_reason=row[7],
                    created_at=float(row[8]),
                    last_access=float(row[9]),
                )
                self._entries[entry.key_hash] = entry
                self._add_to_trie(entry)
                self._total_bytes += entry.nbytes
        self._evict_expired_entries()
        self._evict_to_limit()

    def _write_manifest_entry(self, entry: CacheEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (
                    key_hash, namespace, parent_hash, token_count, tokens_json,
                    file_path, nbytes, checkpoint_reason, created_at, last_access
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.key_hash,
                    entry.namespace,
                    entry.parent_hash,
                    entry.token_count,
                    json.dumps(list(entry.tokens), separators=(",", ":")),
                    str(entry.file_path),
                    entry.nbytes,
                    entry.checkpoint_reason,
                    entry.created_at,
                    entry.last_access,
                ),
            )
            conn.commit()

    def _touch_manifest_entry(self, key_hash: str, last_access: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE cache_entries SET last_access = ? WHERE key_hash = ?",
                (last_access, key_hash),
            )
            conn.commit()

    def _delete_manifest_entry(self, key_hash: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache_entries WHERE key_hash = ?", (key_hash,))
            conn.commit()
