"""
test_emulation_module.py

Unit tests for all Emulation Module components.
No Kubernetes required — runs completely locally.
Mocks dataset files where needed.

Run with:
  python test_emulation_module.py
  python test_emulation_module.py -v    # verbose
"""

import os
import sys
import json
import time
import tempfile
import unittest
import threading
import pandas as pd
import numpy as np

# ── Add emulation_module to path ──────────────────────────────────────────────
MODULE_DIR = os.path.join(os.path.dirname(__file__), "emulation_module")
sys.path.insert(0, MODULE_DIR)

# ── Imports ───────────────────────────────────────────────────────────────────
from config import (
    NODE_CPU_MCORES, NODE_RAM_MI, NODE_PSI_MAX_US,
    NODE_EBPF_MAX_MS, TICK_INTERVAL_S, MAX_WINDOWS,
    ACTIVE_CPU_THRESH,
)
from timeline        import Timeline, Job
from replay_engine   import ReplayEngine
from aggregator      import Aggregator
from dataset_selector import DatasetSelector


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_pod_csv(tmp_dir, filename, n_windows=10, namespaces=None):
    """Create a minimal emulation-ready pod CSV for testing."""
    if namespaces is None:
        namespaces = ["hotel", "sn1"]

    rows = []
    for win in range(n_windows):
        for ns in namespaces:
            rows.append({
                "window_index":     win,
                "pod_name":         f"{ns}/service-a",
                "cpu_usage_mcores": 5000 + win * 100,
                "ram_usage_mi":     1024,
                "disk_space_mb":    500,
                "disk_usage_mb":    10,
                "disk_ios":         5,
                "cpu_psi_some_us":  50000 + win * 1000,
                "sched_total_ms":   1000 + win * 10,
                "dstate_total_ms":  50 + win,
                "softirq_total_ms": 20 + win,
                "pod_cpu_watts":    12.0 + win * 0.1,
            })

    path = os.path.join(tmp_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_dataset_index(tmp_dir, entries):
    """Create a dataset_index.json for testing."""
    index_path = os.path.join(tmp_dir, "dataset_index.json")
    with open(index_path, "w") as f:
        json.dump(entries, f)
    return index_path


# ═══════════════════════════════════════════════════════════════════════════════
# Timeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeline(unittest.TestCase):

    def setUp(self):
        self.tl = Timeline()

    def test_add_job_returns_job(self):
        """add_job should return a Job with correct fields."""
        job = self.tl.add_job("hotel", 300)
        self.assertEqual(job.app_type, "hotel")
        self.assertEqual(job.lifetime_seconds, 300)
        self.assertEqual(job.status, "queued")
        self.assertIsNotNone(job.job_id)

    def test_job_becomes_active_at_next_tick(self):
        """Job should be queued immediately, active after first tick."""
        job = self.tl.add_job("hotel", 300)
        # before tick: still queued
        self.assertEqual(job.status, "queued")
        # after tick: running
        self.tl.tick()
        self.assertEqual(job.status, "running")

    def test_composition_reflects_active_jobs(self):
        """Composition should count active jobs per app type."""
        self.tl.add_job("hotel", 300)
        self.tl.add_job("hotel", 300)
        self.tl.add_job("sn",    300)
        self.tl.tick()
        comp = self.tl.get_composition()
        self.assertEqual(comp.get("hotel", 0), 2)
        self.assertEqual(comp.get("sn",    0), 1)
        self.assertEqual(comp.get("sa",    0), 0)

    def test_job_expires_after_lifetime(self):
        """Job should be removed after its end_tick."""
        # lifetime=5s, tick=5s → end_tick = start_tick + 1
        job = self.tl.add_job("hotel", 5)
        self.tl.tick()   # tick 1: job starts (start_tick=1, end_tick=2)
        comp1 = self.tl.get_composition()
        self.assertEqual(comp1.get("hotel", 0), 1)
        self.tl.tick()   # tick 2: job expires
        comp2 = self.tl.get_composition()
        self.assertEqual(comp2.get("hotel", 0), 0)

    def test_composition_changed_detected(self):
        """composition_changed() should be True when jobs are added/expired."""
        # no jobs → tick → no change
        self.tl.tick()
        self.assertFalse(self.tl.composition_changed())
        # add job
        self.tl.add_job("sa", 300)
        self.tl.tick()
        self.assertTrue(self.tl.composition_changed())

    def test_multiple_app_types(self):
        """Multiple app types should be tracked independently."""
        self.tl.add_job("hotel", 300)
        self.tl.add_job("sn",    300)
        self.tl.add_job("sa",    300)
        self.tl.add_job("sa",    300)
        self.tl.tick()
        comp = self.tl.get_composition()
        self.assertEqual(comp["hotel"], 1)
        self.assertEqual(comp["sn"],    1)
        self.assertEqual(comp["sa"],    2)

    def test_empty_after_all_jobs_expire(self):
        """is_empty() should be True after all jobs expire."""
        self.tl.add_job("hotel", 5)
        self.tl.tick()
        self.tl.tick()
        self.assertTrue(self.tl.is_empty())

    def test_concurrent_job_submission(self):
        """Concurrent add_job calls should not cause race conditions."""
        errors = []
        def submit():
            try:
                for _ in range(10):
                    self.tl.add_job("hotel", 300)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(len(errors), 0)
        self.tl.tick()
        comp = self.tl.get_composition()
        self.assertEqual(comp.get("hotel", 0), 50)

    def test_get_active_jobs_format(self):
        """get_active_jobs() should return list of dicts with required fields."""
        self.tl.add_job("sn", 300)
        self.tl.tick()
        jobs = self.tl.get_active_jobs()
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        for field in ["job_id", "app_type", "lifetime_seconds", "status"]:
            self.assertIn(field, job)


# ═══════════════════════════════════════════════════════════════════════════════
# ReplayEngine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayEngine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.engine = ReplayEngine()

    def test_not_loaded_initially(self):
        self.assertFalse(self.engine.is_loaded())

    def test_load_csv(self):
        """load() should correctly parse a pod CSV into windows."""
        path = make_pod_csv(self.tmp, "test.csv", n_windows=10,
                            namespaces=["hotel", "sn1"])
        self.engine.load(path, "h1s1a0")
        self.assertTrue(self.engine.is_loaded())
        self.assertEqual(self.engine.get_total_windows(), 10)
        self.assertEqual(self.engine.get_window_index(), 0)
        self.assertEqual(self.engine.get_dataset_key(), "h1s1a0")

    def test_window_advance_loops(self):
        """advance_window() should loop back to 0 after MAX_WINDOWS."""
        path = make_pod_csv(self.tmp, "test2.csv", n_windows=5)
        self.engine.load(path, "test")
        # advance through all windows and check loop
        for i in range(1, 5):
            self.engine.advance_window()
            self.assertEqual(self.engine.get_window_index(), i)
        # should loop at window count (5 in this case)
        self.engine.advance_window()
        self.assertEqual(self.engine.get_window_index(), 0)

    def test_get_current_window_pods_count(self):
        """get_current_window_pods() should return pods for current window."""
        path = make_pod_csv(self.tmp, "test3.csv", n_windows=3,
                            namespaces=["hotel", "sn1", "sa1"])
        self.engine.load(path, "h1s1a1")
        # build identity namespace map
        self.engine._namespace_map = {
            "hotel": "hotel", "sn1": "sn1", "sa1": "sa1"
        }
        pods = self.engine.get_current_window_pods()
        self.assertEqual(len(pods), 3)  # one pod per namespace

    def test_namespace_map_identity(self):
        """build_namespace_map() with matching namespaces → identity map."""
        path = make_pod_csv(self.tmp, "test4.csv", n_windows=2,
                            namespaces=["hotel", "sn1"])
        self.engine.load(path, "h1s1a0")
        mapping = self.engine.build_namespace_map(["hotel", "sn1"])
        self.assertEqual(mapping.get("hotel"), "hotel")
        self.assertEqual(mapping.get("sn1"),   "sn1")

    def test_namespace_map_positional(self):
        """build_namespace_map() maps dataset sa2→sa3 when sa2 removed."""
        path = make_pod_csv(self.tmp, "test5.csv", n_windows=2,
                            namespaces=["sa1", "sa2", "sa3"])
        self.engine.load(path, "h0s0a3")
        # sa2 is removed from alive namespaces
        mapping = self.engine.build_namespace_map(["sa1", "sa3", "sa4"])
        self.assertEqual(mapping.get("sa1"), "sa1")
        self.assertEqual(mapping.get("sa2"), "sa3")
        self.assertEqual(mapping.get("sa3"), "sa4")

    def test_unmapped_pods_excluded(self):
        """Pods with unmapped namespaces should be excluded from output."""
        path = make_pod_csv(self.tmp, "test6.csv", n_windows=2,
                            namespaces=["hotel", "sn1", "sa1"])
        self.engine.load(path, "h1s1a1")
        # only map hotel and sn1 — sa1 excluded
        self.engine._namespace_map = {"hotel": "hotel", "sn1": "sn1"}
        pods = self.engine.get_current_window_pods()
        namespaces = [p["pod_name"].split("/")[0] for p in pods]
        self.assertNotIn("sa1", namespaces)
        self.assertIn("hotel", namespaces)

    def test_apply_namespace_map(self):
        """apply_namespace_map() should remap pod names correctly."""
        self.engine._namespace_map = {"sa2": "sa3"}
        result = self.engine.apply_namespace_map("sa2/customer-feedback")
        self.assertEqual(result, "sa3/customer-feedback")

    def test_apply_namespace_map_unknown(self):
        """apply_namespace_map() returns None for unmapped namespaces."""
        self.engine._namespace_map = {"hotel": "hotel"}
        result = self.engine.apply_namespace_map("sn1/unknown-service")
        self.assertIsNone(result)

    def test_pod_values_preserved(self):
        """Pod metric values should be preserved through load and get."""
        path = make_pod_csv(self.tmp, "test7.csv", n_windows=2,
                            namespaces=["hotel"])
        self.engine.load(path, "h1s0a0")
        self.engine._namespace_map = {"hotel": "hotel"}
        pods = self.engine.get_current_window_pods()
        self.assertEqual(len(pods), 1)
        pod = pods[0]
        self.assertIn("cpu_usage_mcores", pod)
        self.assertIn("cpu_psi_some_us",  pod)
        self.assertIn("pod_cpu_watts",    pod)
        self.assertGreater(pod["cpu_usage_mcores"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggregator(unittest.TestCase):

    def setUp(self):
        self.agg = Aggregator()

    def _make_pods(self, n=5, cpu=10000, ram=1024, psi=100000):
        return [
            {
                "cpu_usage_mcores": cpu,
                "ram_usage_mi":     ram,
                "disk_space_mb":    500,
                "disk_usage_mb":    10,
                "cpu_psi_some_us":  psi,
                "sched_total_ms":   1000,
                "dstate_total_ms":  50,
                "softirq_total_ms": 20,
                "pod_cpu_watts":    12.0,
            }
            for _ in range(n)
        ]

    def test_empty_pods_returns_zeros(self):
        """compute([]) should return zeroed snapshot."""
        result = self.agg.compute([])
        self.assertEqual(result["cpu_usage_pct"],    0.0)
        self.assertEqual(result["ram_usage_pct"],    0.0)
        self.assertEqual(result["node_cpu_watts"],   0.0)

    def test_cpu_pct_calculation(self):
        """CPU percentage should be (sum_mcores / NODE_CPU_MCORES) * 100."""
        pods   = self._make_pods(n=2, cpu=25600)  # 2 × 25600 = 51200 mcores
        result = self.agg.compute(pods)
        expected_pct = (51200 / NODE_CPU_MCORES) * 100
        self.assertAlmostEqual(result["cpu_usage_pct"], expected_pct, places=1)

    def test_ram_pct_calculation(self):
        """RAM percentage should be (sum_mi / NODE_RAM_MI) * 100."""
        pods   = self._make_pods(n=4, ram=NODE_RAM_MI // 4)
        result = self.agg.compute(pods)
        self.assertAlmostEqual(result["ram_usage_pct"], 100.0, places=1)

    def test_power_is_sum(self):
        """Node power should be sum of pod watts."""
        pods   = self._make_pods(n=3, cpu=5000)
        total  = sum(p["pod_cpu_watts"] for p in pods)
        result = self.agg.compute(pods)
        self.assertAlmostEqual(result["node_cpu_watts"], total, places=2)

    def test_sched_is_sum(self):
        """sched_total_ms should be sum across pods."""
        pods   = self._make_pods(n=4)
        total  = sum(p["sched_total_ms"] for p in pods)
        result = self.agg.compute(pods)
        self.assertAlmostEqual(result["sched_total_ms"], total, places=2)

    def test_psi_within_limits(self):
        """PSI result should never exceed NODE_PSI_MAX_US."""
        # use extremely high pod PSI values
        pods   = self._make_pods(n=100, psi=NODE_PSI_MAX_US)
        result = self.agg.compute(pods)
        self.assertLessEqual(result["cpu_psi_some_pct"], 100.0)

    def test_psi_positive(self):
        """PSI should always be >= 0."""
        pods   = self._make_pods(n=2, psi=0, cpu=0)
        result = self.agg.compute(pods)
        self.assertGreaterEqual(result["cpu_psi_some_pct"], 0.0)

    def test_cpu_pct_capped_at_100(self):
        """CPU percentage should never exceed 100%."""
        pods   = self._make_pods(n=1000, cpu=NODE_CPU_MCORES)
        result = self.agg.compute(pods)
        self.assertLessEqual(result["cpu_usage_pct"], 100.0)

    def test_timestamp_format(self):
        """Timestamp should be in YYYY-MM-DD HH:MM:SS format."""
        import re
        pods   = self._make_pods(n=1)
        result = self.agg.compute(pods)
        pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        self.assertRegex(result["timestamp"], pattern)

    def test_psi_correlates_with_cpu(self):
        """Higher CPU should generally produce higher PSI estimate."""
        pods_low  = self._make_pods(n=3, cpu=1000,  psi=10000)
        pods_high = self._make_pods(n=3, cpu=50000, psi=500000)
        res_low  = self.agg.compute(pods_low)
        res_high = self.agg.compute(pods_high)
        self.assertGreater(res_high["cpu_psi_some_pct"], res_low["cpu_psi_some_pct"])

    def test_all_required_fields_present(self):
        """Result should contain all fields expected by /usage/latest."""
        required = [
            "timestamp", "cpu_usage_pct", "cpu_psi_some_pct",
            "ram_usage_pct", "ram_usage_mi", "disk_used_pct",
            "sched_total_ms", "dstate_total_ms", "softirq_total_ms",
            "node_cpu_watts"
        ]
        pods   = self._make_pods(n=2)
        result = self.agg.compute(pods)
        for field in required:
            self.assertIn(field, result, f"Missing field: {field}")


# ═══════════════════════════════════════════════════════════════════════════════
# DatasetSelector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetSelector(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # create minimal CSV files
        for key in ["h1s0a0", "h0s1a0", "h0s0a1", "h1s1a0", "h1s1a3"]:
            make_pod_csv(self.tmp, f"{key}_pod.csv", n_windows=5)

        # create index
        self.index = {
            "h1s0a0": {"file": f"datasets/h1s0a0_pod.csv",
                       "source": "real", "hotel": 1, "sn": 0, "sa": 0,
                       "windows": 5, "pods": 2},
            "h0s1a0": {"file": f"datasets/h0s1a0_pod.csv",
                       "source": "real", "hotel": 0, "sn": 1, "sa": 0,
                       "windows": 5, "pods": 2},
            "h0s0a1": {"file": f"datasets/h0s0a1_pod.csv",
                       "source": "real", "hotel": 0, "sn": 0, "sa": 1,
                       "windows": 5, "pods": 2},
            "h1s1a0": {"file": f"datasets/h1s1a0_pod.csv",
                       "source": "real", "hotel": 1, "sn": 1, "sa": 0,
                       "windows": 5, "pods": 2},
            "h1s1a3": {"file": f"datasets/h1s1a3_pod.csv",
                       "source": "generated", "hotel": 1, "sn": 1, "sa": 3,
                       "windows": 5, "pods": 2},
        }
        index_path = make_dataset_index(self.tmp, self.index)

        # patch config to use tmp index
        import config as cfg
        self._orig_index    = cfg.DATASET_INDEX
        self._orig_datasets = cfg.DATASETS_DIR
        cfg.DATASET_INDEX = index_path
        cfg.DATASETS_DIR  = self.tmp

        self.selector = DatasetSelector()

    def tearDown(self):
        import config as cfg
        cfg.DATASET_INDEX = self._orig_index
        cfg.DATASETS_DIR  = self._orig_datasets

    def test_exact_match_hotel(self):
        """Exact match for hotel:1 should return h1s0a0."""
        path, meta = self.selector.select({"hotel": 1})
        self.assertIsNotNone(path)
        self.assertEqual(meta["hotel"], 1)
        self.assertEqual(meta["sn"],    0)

    def test_exact_match_combined(self):
        """Exact match for hotel:1+sn:1+sa:3 should return h1s1a3."""
        path, meta = self.selector.select({"hotel": 1, "sn": 1, "sa": 3})
        self.assertIsNotNone(path)
        self.assertEqual(meta["sa"], 3)

    def test_closest_match_fallback(self):
        """For an unlisted combination should fall back to closest available."""
        # h1s1a3 is in index but h1s1a4 is not — closest should be h1s1a3
        path, meta = self.selector.select({"hotel": 1, "sn": 1, "sa": 4})
        self.assertIsNotNone(path)
        # closest must not exceed requested counts
        self.assertLessEqual(meta["hotel"], 1)
        self.assertLessEqual(meta["sn"],    1)
        self.assertLessEqual(meta["sa"],    4)

    def test_composition_to_key(self):
        """_composition_to_key should produce correct key format."""
        key = self.selector._composition_to_key({"hotel": 2, "sn": 3, "sa": 1})
        self.assertEqual(key, "h2s3a1")

    def test_key_to_composition(self):
        """_key_to_composition should parse key correctly."""
        comp = self.selector._key_to_composition("h3s2a5")
        self.assertEqual(comp["hotel"], 3)
        self.assertEqual(comp["sn"],    2)
        self.assertEqual(comp["sa"],    5)

    def test_exists_true(self):
        self.assertTrue(self.selector.exists({"hotel": 1, "sn": 0, "sa": 0}))

    def test_exists_false(self):
        # use a combination guaranteed not in our mock index
        self.assertFalse(self.selector.exists({"hotel": 9, "sn": 9, "sa": 9}))

    def test_closest_does_not_exceed_requested(self):
        """Closest match should never use MORE replicas than requested."""
        # request hotel:1, sn:0, sa:0 — should not fall back to h1s1a0
        path, meta = self.selector.select({"hotel": 1, "sn": 0, "sa": 0})
        self.assertEqual(meta["sn"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Timeline + ReplayEngine + Aggregator
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.tmp    = tempfile.mkdtemp()
        self.tl     = Timeline()
        self.engine = ReplayEngine()
        self.agg    = Aggregator()

    def test_full_tick_cycle(self):
        """
        Simulate a full tick cycle:
        1. Add a job
        2. Tick → composition changes
        3. Load dataset
        4. Build namespace map
        5. Get pods → aggregate → check metrics
        """
        # create dataset
        path = make_pod_csv(self.tmp, "h1s1a0_pod.csv",
                            n_windows=5,
                            namespaces=["hotel", "sn1"])

        # add job and tick
        self.tl.add_job("hotel", 300)
        changed = self.tl.tick()
        self.assertTrue(changed)

        # load dataset
        self.engine.load(path, "h1s1a0")
        self.engine.build_namespace_map(["hotel", "sn1"])

        # get pods and aggregate
        pods   = self.engine.get_current_window_pods()
        self.assertEqual(len(pods), 2)

        metrics = self.agg.compute(pods)
        self.assertGreater(metrics["cpu_usage_pct"],  0)
        self.assertGreater(metrics["node_cpu_watts"], 0)
        self.assertIn("timestamp", metrics)

    def test_window_advances_each_tick(self):
        """Window index should advance each tick."""
        path = make_pod_csv(self.tmp, "test.csv", n_windows=5)
        self.engine.load(path, "test")
        self.engine._namespace_map = {"hotel": "hotel", "sn1": "sn1"}

        for expected_win in range(3):
            self.assertEqual(self.engine.get_window_index(), expected_win)
            self.engine.advance_window()

    def test_composition_drives_metrics(self):
        """More active pods should produce higher metrics."""
        path_small = make_pod_csv(self.tmp, "small.csv",
                                  n_windows=3, namespaces=["hotel"])
        path_large = make_pod_csv(self.tmp, "large.csv",
                                  n_windows=3,
                                  namespaces=["hotel", "sn1", "sa1", "sa2"])

        self.engine.load(path_small, "small")
        self.engine._namespace_map = {"hotel": "hotel"}
        pods_small  = self.engine.get_current_window_pods()
        res_small   = self.agg.compute(pods_small)

        self.engine.load(path_large, "large")
        self.engine._namespace_map = {
            "hotel": "hotel", "sn1": "sn1", "sa1": "sa1", "sa2": "sa2"
        }
        pods_large  = self.engine.get_current_window_pods()
        res_large   = self.agg.compute(pods_large)

        self.assertGreater(res_large["cpu_usage_pct"],  res_small["cpu_usage_pct"])
        self.assertGreater(res_large["node_cpu_watts"], res_small["node_cpu_watts"])


# ═══════════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests(verbose=False):
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestTimeline,
        TestReplayEngine,
        TestAggregator,
        TestDatasetSelector,
        TestIntegration,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    verbosity = 2 if verbose else 1
    runner    = unittest.TextTestRunner(verbosity=verbosity)
    result    = runner.run(suite)

    print("\n" + "="*60)
    print(f"RESULTS: {result.testsRun} tests | "
          f"{len(result.failures)} failures | "
          f"{len(result.errors)} errors")
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
        for fail in result.failures + result.errors:
            print(f"\n  {fail[0]}")
            print(f"  {fail[1][:200]}")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    success = run_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)