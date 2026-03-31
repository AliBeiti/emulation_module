"""
replay_engine.py

Loads an emulation dataset into memory and replays it window by window.
Manages namespace mapping between dataset pod names and real KWOK pod names.

Key responsibilities:
  - Load dataset CSV once → keep in memory as dict[window_index → DataFrame]
  - Advance window index every tick, loop at MAX_WINDOWS
  - Apply namespace mapping: dataset ns → kwok ns
  - Return current window's pod rows for annotation

Namespace mapping example:
  Dataset has: hotel/, hotel2/, sn1/, sn2/, sa1/, sa2/, sa3/
  KWOK alive:  hotel/, hotel2/, sn1/, sn2/, sa1/, sa3/, sa4/  (sa2 removed)
  Map:
    hotel/  → hotel/
    hotel2/ → hotel2/
    sn1/    → sn1/
    sn2/    → sn2/
    sa1/    → sa1/
    sa2/    → sa3/    ← positional mapping
    sa3/    → sa4/
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import MAX_WINDOWS, POD_CSV_COLS

logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Manages dataset loading and window-by-window replay with namespace mapping.

    Thread safety: the caller (main loop) is single-threaded for replay.
    The api.py may read current state concurrently — reads are safe since
    Python dict/int reads are atomic for simple types.
    """

    def __init__(self):
        # dataset storage: window_index → list of pod row dicts
        self._windows: Dict[int, List[Dict]] = {}
        self._total_windows    = 0
        self._current_window   = 0
        self._current_dataset  = None   # key of currently loaded dataset
        self._namespace_map: Dict[str, str] = {}  # dataset_ns → kwok_ns
        self._loaded           = False

    # ── Dataset loading ───────────────────────────────────────────────────────

    def load(self, csv_path: str, dataset_key: str):
        """
        Load a new dataset. Resets window index to 0.
        Called when composition changes and a new dataset is selected.
        """
        logger.info(f"Loading dataset: {dataset_key} from {csv_path}")

        df = pd.read_csv(csv_path)

        # validate required columns
        missing = [c for c in POD_CSV_COLS if c not in df.columns]
        if missing:
            logger.warning(f"Dataset missing columns: {missing} — filling with 0")
            for c in missing:
                df[c] = 0.0

        # build window dict: {window_index: [{pod_row}, ...]}
        self._windows = {}
        for win_idx, group in df.groupby("window_index"):
            self._windows[int(win_idx)] = group.to_dict(orient="records")

        self._total_windows  = len(self._windows)
        self._current_window = 0
        self._current_dataset = dataset_key
        self._loaded = True

        logger.info(
            f"Dataset loaded: {dataset_key} | "
            f"{self._total_windows} windows | "
            f"{len(df)} total pod rows"
        )

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Namespace mapping ─────────────────────────────────────────────────────

    def build_namespace_map(
        self,
        alive_namespaces: List[str]
    ) -> Dict[str, str]:
        """
        Build a positional mapping from dataset namespaces → alive KWOK namespaces.

        Logic:
          - For each app type (hotel, sn, sa), sort both dataset namespaces
            and alive namespaces for that app type
          - Map them positionally: 1st dataset ns → 1st alive ns, etc.
          - Store result in self._namespace_map

        Args:
            alive_namespaces: list of currently alive KWOK namespace names
                              e.g. ["hotel", "hotel2", "sn1", "sa1", "sa3", "sa4"]

        Returns:
            The built mapping dict (also stored in self._namespace_map)
        """
        if not self._loaded:
            return {}

        # get all dataset namespaces from first window
        dataset_namespaces = set()
        if 0 in self._windows:
            for row in self._windows[0]:
                ns = str(row["pod_name"]).split("/")[0]
                dataset_namespaces.add(ns)

        # group by app type
        app_types = ("hotel", "sn", "sa")
        mapping   = {}

        for app in app_types:
            # dataset namespaces for this app
            ds_ns = sorted([
                ns for ns in dataset_namespaces
                if ns.startswith(app)
            ], key=lambda x: (len(x), x))

            # alive namespaces for this app
            alive_ns = sorted([
                ns for ns in alive_namespaces
                if ns.startswith(app)
            ], key=lambda x: (len(x), x))

            # positional mapping
            for i, ds in enumerate(ds_ns):
                if i < len(alive_ns):
                    mapping[ds] = alive_ns[i]
                    logger.debug(f"Namespace map: {ds} → {alive_ns[i]}")
                else:
                    # more dataset namespaces than alive — skip extras
                    logger.debug(f"Namespace {ds} has no alive counterpart, skipping")

        self._namespace_map = mapping
        logger.info(f"Namespace map built: {mapping}")
        return mapping

    def apply_namespace_map(self, pod_name: str) -> Optional[str]:
        """
        Apply namespace mapping to a pod name.
        Returns remapped pod name or None if namespace not in map.

        e.g. "sa2/customer-feedback" → "sa3/customer-feedback"
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
        """
        Return pod rows for the current window with namespace mapping applied.
        Rows with unmapped namespaces are skipped.
        """
        if not self._loaded or self._current_window not in self._windows:
            return []

        raw_rows = self._windows[self._current_window]
        mapped_rows = []

        for row in raw_rows:
            remapped_name = self.apply_namespace_map(row["pod_name"])
            if remapped_name is None:
                continue
            row_copy = dict(row)
            row_copy["pod_name"] = remapped_name
            mapped_rows.append(row_copy)

        return mapped_rows

    def advance_window(self):
        """
        Move to the next window. Loops back to 0 after MAX_WINDOWS
        or after the last window in the dataset.
        """
        if not self._loaded:
            return

        limit = min(self._total_windows, MAX_WINDOWS)
        self._current_window = (self._current_window + 1) % limit
        logger.debug(f"Window advanced to {self._current_window}/{limit}")

    def get_window_index(self) -> int:
        return self._current_window

    def get_total_windows(self) -> int:
        return self._total_windows

    def get_dataset_key(self) -> Optional[str]:
        return self._current_dataset

    def get_namespace_map(self) -> Dict[str, str]:
        return dict(self._namespace_map)

    def get_dataset_namespaces(self) -> List[str]:
        """Return list of unique namespaces in the loaded dataset."""
        if not self._loaded or 0 not in self._windows:
            return []
        ns_set = set()
        for row in self._windows[0]:
            ns = str(row["pod_name"]).split("/")[0]
            ns_set.add(ns)
        return sorted(ns_set)