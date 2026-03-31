"""
dataset_selector.py

Responsible for selecting the correct emulation dataset given a workload
composition. Loads dataset_index.json once at startup and keeps it in memory.

Usage:
    selector = DatasetSelector()
    path, meta = selector.select({"hotel": 2, "sn": 1, "sa": 3})
    # path = "datasets/h2s1a3_pod.csv"
"""

import json
import os
import logging
from typing import Dict, Optional, Tuple

from config import DATASET_INDEX, DATASETS_DIR

logger = logging.getLogger(__name__)


class DatasetSelector:
    """
    Selects the best matching dataset for a given workload composition.

    Lookup order:
    1. Exact match  → use it directly
    2. No match     → find closest by minimizing replica gap
    3. No close     → return None (caller must handle gracefully)
    """

    def __init__(self):
        self._index: Dict = {}
        self._load_index()

    def _load_index(self):
        """Load dataset_index.json into memory. Called once at startup."""
        if not os.path.exists(DATASET_INDEX):
            raise FileNotFoundError(
                f"Dataset index not found: {DATASET_INDEX}\n"
                f"Run prepare_datasets.py first."
            )
        with open(DATASET_INDEX, "r") as f:
            self._index = json.load(f)
        logger.info(f"Loaded dataset index: {len(self._index)} entries")

    @staticmethod
    def _composition_to_key(composition: Dict[str, int]) -> str:
        """Convert {hotel:2, sn:1, sa:3} → 'h2s1a3'"""
        h = composition.get("hotel", 0)
        s = composition.get("sn", 0)
        a = composition.get("sa", 0)
        return f"h{h}s{s}a{a}"

    @staticmethod
    def _key_to_composition(key: str) -> Dict[str, int]:
        """Convert 'h2s1a3' → {hotel:2, sn:1, sa:3}"""
        import re
        m = re.match(r"h(\d+)s(\d+)a(\d+)", key)
        if not m:
            return {"hotel": 0, "sn": 0, "sa": 0}
        return {
            "hotel": int(m.group(1)),
            "sn":    int(m.group(2)),
            "sa":    int(m.group(3)),
        }

    def select(
        self,
        composition: Dict[str, int]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Select the best dataset for the given composition.

        Returns:
            (file_path, metadata_dict) or (None, None) if nothing found
        """
        key = self._composition_to_key(composition)

        # ── Exact match ───────────────────────────────────────────────────────
        if key in self._index:
            entry = self._index[key]
            path  = os.path.join(os.path.dirname(DATASETS_DIR), entry["file"])
            logger.info(f"Exact match: {key} → {path}")
            return path, entry

        # ── Closest match ─────────────────────────────────────────────────────
        logger.warning(f"No exact match for {key}, finding closest...")
        closest_key, closest_score = self._find_closest(composition)

        if closest_key is None:
            logger.error(f"No suitable dataset found for {key}")
            return None, None

        entry = self._index[closest_key]
        path  = os.path.join(os.path.dirname(DATASETS_DIR), entry["file"])
        logger.warning(f"Using closest match: {closest_key} (score={closest_score}) for {key}")
        return path, entry

    def _find_closest(
        self,
        composition: Dict[str, int]
    ) -> Tuple[Optional[str], float]:
        """
        Find the closest dataset key by minimizing total replica gap.
        Only considers datasets that do not exceed requested counts.
        """
        h = composition.get("hotel", 0)
        s = composition.get("sn", 0)
        a = composition.get("sa", 0)

        best_key   = None
        best_score = float("inf")

        for key, entry in self._index.items():
            bh = entry.get("hotel", 0)
            bs = entry.get("sn",    0)
            ba = entry.get("sa",    0)

            # do not use a dataset that has MORE of any app than requested
            if bh > h or bs > s or ba > a:
                continue

            # gap = total missing replicas
            gap = (h - bh) + (s - bs) + (a - ba)

            # penalize if required apps are completely absent
            missing_apps = 0
            if h > 0 and bh == 0: missing_apps += 1
            if s > 0 and bs == 0: missing_apps += 1
            if a > 0 and ba == 0: missing_apps += 1

            score = gap + missing_apps * 100

            if score < best_score:
                best_score = score
                best_key   = key

        return best_key, best_score

    def list_available(self) -> Dict:
        """Return full index for inspection."""
        return self._index

    def get_entry(self, composition: Dict[str, int]) -> Optional[Dict]:
        """Return index entry for a composition without loading the file."""
        key = self._composition_to_key(composition)
        return self._index.get(key)

    def exists(self, composition: Dict[str, int]) -> bool:
        """Check if an exact match exists for this composition."""
        return self._composition_to_key(composition) in self._index