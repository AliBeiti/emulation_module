"""
dataset_selector.py

Responsible for selecting the correct emulation dataset given a workload
composition. Generates datasets on demand via dataset_generator.py and
caches them in a size-bound LRU cache (dataset_cache.py) for the life of
the process (no more static dataset_index.json / precomputed grid).

At startup: sweeps DATASETS_DIR of any stray non-Tier-A .parquet files left
over from a previous run (they're cheap to regenerate on demand, so nothing
is persisted across restarts), then pre-warms the cache with all 62 Tier A
compositions, pinned so they're never evicted.

Usage:
    selector = DatasetSelector()
    path, meta = selector.select({"hotel": 2, "sn": 1, "sa": 3, "es": 1})
    # path = ".../datasets/h2s1a3e1_pod.parquet"
"""

import os
import logging
from typing import Dict, Optional, Tuple

import config
from dataset_generator import (
    load_templates, load_experiment_meta, get_or_generate,
    composition_to_key, parse_composition,
)
from dataset_cache import SizeBoundLRUCache

logger = logging.getLogger(__name__)


class DatasetSelector:
    """
    Selects (generating on demand, if needed) the dataset for a given
    workload composition.

    Lookup order (delegated to dataset_generator.get_or_generate):
    1. In-memory cache hit        → return immediately
    2. Exact Tier A match on disk → copy + recompute directly
    3. No exact match             → build from closest Tier A base + extra replicas
    4. No viable base             → return (None, None), caller must handle gracefully
    """

    def __init__(self):
        self._templates = load_templates(config.CORRECTED_DIR)
        self._meta       = load_experiment_meta(config.CORRECTED_DIR)
        self._available  = {
            f.replace('_pod_corrected.csv', '')
            for f in os.listdir(config.CORRECTED_DIR)
            if f.endswith('_pod_corrected.csv')
        }

        self._sweep_non_tier_a()

        self._cache = SizeBoundLRUCache(
            out_dir=config.DATASETS_DIR,
            max_bytes=config.DATASET_CACHE_MAX_BYTES,
        )
        self._prewarm_tier_a()

        logger.info(
            f"DatasetSelector ready — "
            f"{len(self._meta.get('tier_A', []))} Tier A experiments, "
            f"{len(self._available)} corrected files available, "
            f"cache: {self._cache.stats()}"
        )

    def _sweep_non_tier_a(self) -> None:
        """
        Delete any .parquet file already in DATASETS_DIR whose composition
        key is not a Tier A composition. Bounds disk usage across restarts
        without a persisted cache index — non-Tier-A files are cheap to
        regenerate on demand.
        """
        tier_A = self._meta.get("tier_A", [])
        tier_A_keys = set()
        for exp in tier_A:
            h, s, a, e = parse_composition(exp)
            tier_A_keys.add(composition_to_key(h, s, a, e))

        out_dir = config.DATASETS_DIR
        if not os.path.isdir(out_dir):
            return

        removed, removed_bytes = 0, 0
        for fname in os.listdir(out_dir):
            if not fname.endswith("_pod.parquet"):
                continue
            key = fname[: -len("_pod.parquet")]
            if key in tier_A_keys:
                continue
            path = os.path.join(out_dir, fname)
            try:
                removed_bytes += os.path.getsize(path)
                os.remove(path)
                removed += 1
            except OSError as e:
                logger.warning(f"Sweep: could not remove {path}: {e}")

        if removed:
            logger.info(
                f"Startup sweep: removed {removed} non-Tier-A dataset "
                f"file(s) ({removed_bytes:,} bytes)"
            )

    def _prewarm_tier_a(self) -> None:
        """
        Generate (or copy) all Tier A compositions into the cache before
        the first real request, pinned so they're never evicted.
        """
        tier_A = self._meta.get("tier_A", [])
        with self._cache.prewarm():
            for exp in tier_A:
                h, s, a, e = parse_composition(exp)
                entry = get_or_generate(
                    h, s, a, e,
                    self._cache, self._templates, tier_A,
                    self._available, config.CORRECTED_DIR, config.BASELINE_PATH,
                    out_dir=config.DATASETS_DIR,
                )
                if entry is None:
                    logger.warning(f"Pre-warm: failed to generate Tier A experiment '{exp}'")

        stats = self._cache.stats()
        logger.info(
            f"Pre-warm complete: {stats['entries']} entries, "
            f"{stats['total_bytes']:,} bytes"
        )

    def select(
        self,
        composition: Dict[str, int]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Select (generating on demand if needed) the best dataset for the
        given composition.

        Returns:
            (file_path, metadata_dict) or (None, None) if nothing found
        """
        h = composition.get("hotel", 0)
        s = composition.get("sn",    0)
        a = composition.get("sa",    0)
        e = composition.get("es",    0)
        key = composition_to_key(h, s, a, e)

        entry = get_or_generate(
            h, s, a, e,
            self._cache, self._templates, self._meta.get("tier_A", []),
            self._available, config.CORRECTED_DIR, config.BASELINE_PATH,
            out_dir=config.DATASETS_DIR,
        )

        if entry is None:
            logger.error(f"No dataset could be generated for {key}")
            return None, None

        # entry["file"] carries a hardcoded "datasets/" prefix from
        # dataset_generator.py (a vestige of the old batch-script convention);
        # join against the real, possibly differently-named, DATASETS_DIR
        # directly rather than trusting that prefix to match its basename.
        path = os.path.join(config.DATASETS_DIR, os.path.basename(entry["file"]))
        logger.info(f"Dataset ready: {key} → {path} (source={entry.get('source')})")
        return path, entry

    def key_for(self, composition: Dict[str, int]) -> str:
        """
        Convert a {"hotel":.., "sn":.., "sa":.., "es":..} composition dict
        into its dataset_generator composition_to_key() string
        (e.g. "h2s1a3e1"). Thin public wrapper so callers (main.py) don't
        need to import dataset_generator directly just to log/compare keys.
        """
        return composition_to_key(
            composition.get("hotel", 0), composition.get("sn", 0),
            composition.get("sa",    0), composition.get("es", 0),
        )

    def get_entry(self, composition: Dict[str, int]) -> Optional[Dict]:
        """
        Return the cached index entry for a composition if it has already
        been generated this run. Does NOT trigger generation — call
        select() for that.
        """
        return self._cache.peek(self.key_for(composition))

    def exists(self, composition: Dict[str, int]) -> bool:
        """
        True if this composition has already been generated and cached
        this run. Does NOT check whether it *could* be generated — that
        only happens on select().
        """
        return self.key_for(composition) in self._cache

    def list_available(self) -> Dict:
        """
        Return datasets generated so far this run (the 62 pre-warmed Tier A
        entries plus anything select() has produced since). Unlike the old
        static index, this is NOT the full theoretical grid — generation is
        lazy/on-demand.
        """
        return self._cache.snapshot()
