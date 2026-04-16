"""
replay_engine.py

Loads an emulation dataset into memory and replays it window by window.
Manages namespace mapping between dataset pod names and buyer-named KWOK namespaces.

Key responsibilities:
  - Load dataset CSV once → keep in memory as dict[window_index → list of dicts]
  - Advance window index every tick, loop at MAX_WINDOWS
  - Apply namespace mapping: dataset ns → buyer-named ns (e.g. hotel/ → hotel_serf2/)
  - Preserve window index across dataset switches (no restart from 0)
  - Return current window's pod rows for annotation

Namespace mapping (new):
  Dataset has: hotel/, hotel2/, hotel3/  (for 3 hotel jobs)
  Active buyers: serf1, serf2, serf3     (ordered by job start time)
  Map:
    hotel/  → hotel_serf1/
    hotel2/ → hotel_serf2/
    hotel3/ → hotel_serf3/

  When serf2's job expires:
    Dataset switches to h2s0a0: hotel/, hotel2/
    Remaining buyers: serf1, serf3
    Map:
      hotel/  → hotel_serf1/
      hotel2/ → hotel_serf3/
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import MAX_WINDOWS, POD_CSV_COLS

logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Manages dataset loading and window-by-window replay with
    buyer-named namespace mapping.
    """

    def __init__(self):
        self._windows: Dict[int, List[Dict]] = {}
        self._total_windows   = 0
        self._current_window  = 0
        self._current_dataset = None
        self._namespace_map: Dict[str, str] = {}  # dataset_ns → kwok_ns
        self._loaded          = False

    # ── Dataset loading ───────────────────────────────────────────────────────

    def load(self, csv_path: str, dataset_key: str, preserve_window: bool = False):
        """
        Load a new dataset.

        Args:
            csv_path:         path to pod CSV
            dataset_key:      key string e.g. 'h2s0a0'
            preserve_window:  if True, keep current window index (clamped to
                              new dataset size) instead of resetting to 0.
                              Use True when switching datasets due to a job
                              joining/leaving — replay continues at same point.
        """
        logger.info(f"Loading dataset: {dataset_key} from {csv_path}")

        df = pd.read_csv(csv_path)

        # validate required columns
        missing = [c for c in POD_CSV_COLS if c not in df.columns]
        if missing:
            logger.warning(f"Dataset missing columns: {missing} — filling with 0")
            for c in missing:
                df[c] = 0.0

        # build window dict
        self._windows = {}
        for win_idx, group in df.groupby("window_index"):
            self._windows[int(win_idx)] = group.to_dict(orient="records")

        self._total_windows  = len(self._windows)
        self._current_dataset = dataset_key
        self._loaded = True

        if preserve_window and self._total_windows > 0:
            # clamp to new dataset size so we never go out of bounds
            self._current_window = self._current_window % self._total_windows
            logger.info(
                f"Dataset loaded: {dataset_key} | "
                f"{self._total_windows} windows | "
                f"window preserved at {self._current_window}"
            )
        else:
            self._current_window = 0
            logger.info(
                f"Dataset loaded: {dataset_key} | "
                f"{self._total_windows} windows | "
                f"window reset to 0"
            )

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Namespace mapping ─────────────────────────────────────────────────────

    def build_namespace_map(
        self,
        alive_namespaces: List[str],
    ) -> Dict[str, str]:
        """
        Build positional mapping from dataset namespaces → alive KWOK namespaces.
        Used in no-kwok mode or legacy mode.
        """
        if not self._loaded:
            return {}

        dataset_namespaces = self._get_dataset_namespaces_set()
        app_types = ("hotel", "sn", "sa")
        mapping   = {}

        for app in app_types:
            ds_ns = sorted(
                [ns for ns in dataset_namespaces if ns.startswith(app)],
                key=lambda x: (len(x), x)
            )
            alive_ns = sorted(
                [ns for ns in alive_namespaces if ns.startswith(app)],
                key=lambda x: (len(x), x)
            )
            for i, ds in enumerate(ds_ns):
                if i < len(alive_ns):
                    mapping[ds] = alive_ns[i]
                    logger.debug(f"Namespace map: {ds} → {alive_ns[i]}")

        self._namespace_map = mapping
        logger.info(f"Namespace map built: {mapping}")
        return mapping

    def build_buyer_namespace_map(
        self,
        active_jobs_by_app: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """
        Build namespace mapping using buyer names.

        Args:
            active_jobs_by_app: dict of {app_type: [buyer_name, ...]}
                                ordered by job start time (oldest first)
                                e.g. {"hotel": ["serf1", "serf3"], "sn": ["serf2"]}

        Returns:
            mapping dict stored in self._namespace_map

        Example:
            Dataset h2s0a0 has: hotel/, hotel2/
            active_jobs_by_app = {"hotel": ["serf1", "serf3"]}
            Result: {hotel/: hotel_serf1/, hotel2/: hotel_serf3/}
        """
        if not self._loaded:
            return {}

        dataset_namespaces = self._get_dataset_namespaces_set()
        app_types = ("hotel", "sn", "sa")
        mapping   = {}

        for app in app_types:
            ds_ns = sorted(
                [ns for ns in dataset_namespaces if ns.startswith(app)],
                key=lambda x: (len(x), x)
            )
            buyers = active_jobs_by_app.get(app, [])

            for i, ds in enumerate(ds_ns):
                if i < len(buyers):
                    buyer   = buyers[i]
                    kwok_ns = f"{app}_{buyer}"
                    mapping[ds] = kwok_ns
                    logger.debug(f"Buyer namespace map: {ds} → {kwok_ns}")
                else:
                    logger.debug(f"Dataset ns {ds} has no buyer — skipping")

        self._namespace_map = mapping
        logger.info(f"Buyer namespace map built: {mapping}")
        return mapping

    def apply_namespace_map(self, pod_name: str) -> Optional[str]:
        """
        Apply namespace mapping to a pod name.
        Returns remapped pod name or None if namespace not in map.
        """
        parts = str(pod_name).split("/", 1)
        if len(parts) != 2:
            return None
        ns, service = parts
        mapped_ns = self._namespace_map.get(ns)
        if mapped_ns is None:
            return None
        return f"{mapped_ns}/{service}"

    # ── Window replay ─────────────────────────────────────────────────────────

    def get_current_window_pods(self) -> List[Dict]:
        """Return pod rows for current window with namespace mapping applied."""
        if not self._loaded or self._current_window not in self._windows:
            return []

        raw_rows    = self._windows[self._current_window]
        mapped_rows = []
        for row in raw_rows:
            remapped = self.apply_namespace_map(row["pod_name"])
            if remapped is None:
                continue
            row_copy             = dict(row)
            row_copy["pod_name"] = remapped
            mapped_rows.append(row_copy)

        return mapped_rows

    def advance_window(self):
        """Move to next window, looping at MAX_WINDOWS or dataset end."""
        if not self._loaded:
            return
        limit = min(self._total_windows, MAX_WINDOWS)
        self._current_window = (self._current_window + 1) % limit
        logger.debug(f"Window advanced to {self._current_window}/{limit}")

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_window_index(self) -> int:
        return self._current_window

    def get_total_windows(self) -> int:
        return self._total_windows

    def get_dataset_key(self) -> Optional[str]:
        return self._current_dataset

    def get_namespace_map(self) -> Dict[str, str]:
        return dict(self._namespace_map)

    def get_dataset_namespaces(self) -> List[str]:
        """Return sorted list of unique namespaces in the loaded dataset."""
        return sorted(self._get_dataset_namespaces_set())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_dataset_namespaces_set(self) -> set:
        if not self._loaded or 0 not in self._windows:
            return set()
        ns_set = set()
        for row in self._windows[0]:
            ns = str(row["pod_name"]).split("/")[0]
            ns_set.add(ns)
        return ns_set