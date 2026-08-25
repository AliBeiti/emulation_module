"""
dataset_cache.py

Dict-like (duck-typed) cache sitting in front of
dataset_generator.get_or_generate(), bounded by total bytes on disk rather
than entry count, with LRU eviction.

Implements __contains__/__getitem__/__setitem__ so it's a drop-in
replacement for the plain `cache: Dict[str, Dict]` get_or_generate()
already accepts -- no changes to dataset_generator.py are required.

Usage:
    cache = SizeBoundLRUCache(out_dir=DATASETS_DIR, max_bytes=500 * 1024 * 1024)

    # normal runtime use (via get_or_generate, unchanged):
    entry = get_or_generate(h, s, a, e, cache, ...)

    # pre-warm: suspend eviction, pin everything added, one eviction pass after
    with cache.prewarm():
        for exp in tier_A:
            h, s, a, e = parse_composition(exp)
            get_or_generate(h, s, a, e, cache, ...)
"""

import os
import time
import logging
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class _CacheRecord:
    entry:      Dict
    size_bytes: int
    last_used:  float
    pinned:     bool = False


class SizeBoundLRUCache:
    """
    LRU cache of dataset_generator entry dicts, bounded by total bytes of
    the underlying .parquet files on disk (not entry count).

    Duck-types dict's __contains__/__getitem__/__setitem__ so it can be
    passed directly as the `cache` argument to get_or_generate() with no
    changes required there. Bookkeeping (size, last-used, pinned) lives on
    an internal _CacheRecord wrapper -- __getitem__ returns the original
    entry dict unmodified, so callers see exactly the same shape as before.
    """

    def __init__(self, out_dir: str, max_bytes: int):
        self._out_dir      = out_dir
        self._max_bytes    = max_bytes
        self._entries: "OrderedDict[str, _CacheRecord]" = OrderedDict()
        self._total_bytes  = 0
        self._prewarming   = False

    # ── dict-like interface used by get_or_generate() ──────────────────────────

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> Dict:
        rec = self._entries[key]
        rec.last_used = time.time()
        self._entries.move_to_end(key)
        return rec.entry

    def __setitem__(self, key: str, entry: Dict) -> None:
        path = os.path.join(self._out_dir, os.path.basename(entry["file"]))
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
            logger.warning(f"Cache: could not stat {path} for {key} — tracking as 0 bytes")

        pinned = False
        if key in self._entries:
            old = self._entries.pop(key)
            self._total_bytes -= old.size_bytes
            pinned = old.pinned   # re-insertion keeps an existing pin

        self._entries[key] = _CacheRecord(
            entry=entry, size_bytes=size, last_used=time.time(), pinned=pinned,
        )
        self._total_bytes += size
        self._entries.move_to_end(key)

        if not self._prewarming:
            self._evict_until_within_budget()

    def __len__(self) -> int:
        return len(self._entries)

    # ── non-mutating introspection (does NOT affect LRU order) ─────────────────

    def peek(self, key: str) -> Optional[Dict]:
        """Return the entry for key without touching LRU order, or None if absent."""
        rec = self._entries.get(key)
        return rec.entry if rec else None

    def snapshot(self) -> Dict[str, Dict]:
        """Return {key: entry} for every cached entry without touching LRU order."""
        return {key: rec.entry for key, rec in self._entries.items()}

    # ── pinning ──────────────────────────────────────────────────────────────

    def pin(self, key: str) -> None:
        if key in self._entries:
            self._entries[key].pinned = True

    def unpin(self, key: str) -> None:
        if key in self._entries:
            self._entries[key].pinned = False

    @contextmanager
    def prewarm(self):
        """
        Suspend eviction for everything inserted inside this block, then
        pin every entry currently in the cache (including any that existed
        before the block started) and run exactly one eviction pass. If
        max_bytes is smaller than the now-fully-pinned footprint, that pass
        can't evict anything and logs a warning instead of silently
        exceeding the budget.
        """
        self._prewarming = True
        try:
            yield
        finally:
            self._prewarming = False
            for rec in self._entries.values():
                rec.pinned = True
            self._evict_until_within_budget()

    # ── eviction ─────────────────────────────────────────────────────────────

    def _evict_until_within_budget(self) -> None:
        while self._total_bytes > self._max_bytes:
            victim_key = self._find_lru_unpinned()
            if victim_key is None:
                logger.warning(
                    f"Cache over budget ({self._total_bytes:,} > {self._max_bytes:,} "
                    f"bytes) but all {len(self._entries)} entries are pinned — "
                    f"cannot evict further"
                )
                break
            self._evict(victim_key)

    def _find_lru_unpinned(self) -> Optional[str]:
        for key, rec in self._entries.items():   # OrderedDict: oldest first
            if not rec.pinned:
                return key
        return None

    def _evict(self, key: str) -> None:
        rec = self._entries.pop(key)
        self._total_bytes -= rec.size_bytes
        path = os.path.join(self._out_dir, os.path.basename(rec.entry["file"]))
        try:
            os.remove(path)
        except OSError as e:
            logger.warning(f"Cache: could not remove evicted file {path}: {e}")
        logger.info(
            f"Evicted {key} ({rec.size_bytes:,} bytes) — "
            f"now {self._total_bytes:,}/{self._max_bytes:,} bytes"
        )

    # ── introspection ────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        pinned = [r for r in self._entries.values() if r.pinned]
        return {
            "entries":        len(self._entries),
            "total_bytes":    self._total_bytes,
            "max_bytes":      self._max_bytes,
            "pinned_entries": len(pinned),
            "pinned_bytes":   sum(r.size_bytes for r in pinned),
        }
