"""
kwok_manager.py

The only component that communicates with Kubernetes.
Manages KWOK node, namespaces, and pod lifecycle.
Patches pod annotations with metric values every tick.

Optimization principles:
  - Create/delete only the diff (what changed), never teardown everything
  - Cache pod existence to avoid redundant API calls
  - Batch annotation patches per namespace
  - Reuse kubernetes client connection pool
"""

import json
import logging
import subprocess
from typing import Dict, List, Optional, Set

from config import (
    KWOK_NODE_NAME, KWOK_NODE_CPU, KWOK_NODE_MEMORY,
    ANNOT_CPU, ANNOT_RAM, ANNOT_PSI,
    ANNOT_SCHED, ANNOT_DSTATE, ANNOT_SOFTIRQ,
    ANNOT_POWER, ANNOT_DISK_SPACE, ANNOT_DISK_USAGE,
    ANNOT_WINDOW, ANNOT_TIMESTAMP,
)

logger = logging.getLogger(__name__)

try:
    from kubernetes import client, config as k8s_config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    logger.warning("kubernetes package not available — KWOK operations disabled")


class KWOKManager:
    """
    Manages KWOK node, namespaces, and pod lifecycle in Kubernetes.

    Responsibilities:
      1. Create fake KWOK node at startup
      2. Sync namespaces and pods when composition changes (diff only)
      3. Patch pod annotations every tick with current metrics
      4. Clean up on shutdown

    Pod naming convention:
      Each pod_name in dataset has format "namespace/service-name"
      e.g. "hotel2/frontend" → namespace="hotel2", pod_name="frontend"
    """

    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run
        self._k8s: Optional[client.CoreV1Api] = None
        self._alive_namespaces: Set[str]  = set()
        self._alive_pods: Dict[str, Set[str]] = {}  # {namespace: {pod_name}}

        if not dry_run and K8S_AVAILABLE:
            self._init_k8s_client()

    # ── Kubernetes client ─────────────────────────────────────────────────────

    def _init_k8s_client(self):
        """Initialize Kubernetes client with connection pooling."""
        try:
            try:
                k8s_config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes config")
            except k8s_config.ConfigException:
                import os
                k3s_cfg = "/etc/rancher/k3s/k3s.yaml"
                if os.path.exists(k3s_cfg) and "KUBECONFIG" not in os.environ:
                    os.environ["KUBECONFIG"] = k3s_cfg
                k8s_config.load_kube_config()
                logger.info("Using kubeconfig file")

            cfg = client.Configuration.get_default_copy()
            cfg.connection_pool_maxsize = 20
            api_client = client.ApiClient(configuration=cfg)
            self._k8s = client.CoreV1Api(api_client=api_client)
            self._k8s.list_namespace(limit=1)   # test connection
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            self._k8s = None

    # ── KWOK Node ─────────────────────────────────────────────────────────────

    def ensure_node(self):
        """Create the KWOK emulation node if it does not exist."""
        if self._dry_run or not self._k8s:
            logger.info(f"[DRY-RUN] Would create KWOK node: {KWOK_NODE_NAME}")
            return

        try:
            self._k8s.read_node(KWOK_NODE_NAME)
            logger.info(f"KWOK node already exists: {KWOK_NODE_NAME}")
            return
        except ApiException as e:
            if e.status != 404:
                raise

        manifest = {
            "apiVersion": "v1",
            "kind": "Node",
            "metadata": {
                "name": KWOK_NODE_NAME,
                "annotations": {
                    "node.alpha.kubernetes.io/ttl": "0",
                    "kwok.x-k8s.io/node": "fake"
                },
                "labels": {
                    "type": "kwok",
                    "kubernetes.io/hostname": KWOK_NODE_NAME,
                    "emulation.k8s.io/node": "true"
                }
            },
            "spec": {
                "taints": [{
                    "effect": "NoSchedule",
                    "key": "kwok.x-k8s.io/node",
                    "value": "fake"
                }]
            },
            "status": {
                "allocatable": {
                    "cpu": KWOK_NODE_CPU,
                    "memory": KWOK_NODE_MEMORY,
                    "pods": "500"
                },
                "capacity": {
                    "cpu": KWOK_NODE_CPU,
                    "memory": KWOK_NODE_MEMORY,
                    "pods": "500"
                },
                "phase": "Running",
                "conditions": [
                    {"type": "Ready", "status": "True",
                     "reason": "KubeletReady",
                     "message": "kubelet is posting ready status"},
                ]
            }
        }
        self._kubectl_apply(manifest)
        logger.info(f"Created KWOK node: {KWOK_NODE_NAME}")

    # ── Namespace & Pod Sync ──────────────────────────────────────────────────

    def sync(self, new_pod_rows: List[Dict]):
        """
        Sync namespaces and pods to match the new dataset's pod list.
        Only creates/deletes what changed — never tears down everything.

        Args:
            new_pod_rows: pod rows from the new dataset (single window sample)
                          used only to determine required namespaces and pod names
        """
        # determine required namespaces and pods from dataset
        required: Dict[str, Set[str]] = {}
        for row in new_pod_rows:
            parts = str(row["pod_name"]).split("/", 1)
            if len(parts) != 2:
                continue
            ns, pod = parts
            if ns not in required:
                required[ns] = set()
            required[ns].add(pod)

        required_ns  = set(required.keys())
        current_ns   = set(self._alive_namespaces)

        ns_to_create = required_ns - current_ns
        ns_to_delete = current_ns  - required_ns

        # create new namespaces and their pods
        for ns in ns_to_create:
            self._create_namespace(ns)
            for pod_name in required.get(ns, set()):
                self._create_pod(ns, pod_name)

        # delete removed namespaces (deletes all pods inside too)
        for ns in ns_to_delete:
            self._delete_namespace(ns)

        # for unchanged namespaces, sync pod diff
        for ns in required_ns & current_ns:
            current_pods  = self._alive_pods.get(ns, set())
            required_pods = required.get(ns, set())
            for pod in required_pods - current_pods:
                self._create_pod(ns, pod)
            for pod in current_pods - required_pods:
                self._delete_pod(ns, pod)

        # update internal state
        self._alive_namespaces = required_ns
        self._alive_pods = {ns: set(pods) for ns, pods in required.items()}

        logger.info(
            f"Sync complete: {len(required_ns)} namespaces, "
            f"{sum(len(p) for p in required.values())} pods"
        )

    def get_alive_namespaces(self) -> List[str]:
        """Return list of currently alive namespace names."""
        return list(self._alive_namespaces)

    # ── Annotation Patching ───────────────────────────────────────────────────

    def patch_annotations(self, pod_rows: List[Dict], window_index: int):
        """
        Patch all pods with their current metric values as annotations.
        Called every tick. Grouped by namespace for efficiency.

        Args:
            pod_rows: current window's pod rows with mapped pod_names
            window_index: current replay window index
        """
        if self._dry_run or not self._k8s:
            return

        from datetime import datetime
        ts = datetime.now().isoformat()

        # group rows by namespace
        by_ns: Dict[str, List[Dict]] = {}
        for row in pod_rows:
            parts = str(row["pod_name"]).split("/", 1)
            if len(parts) != 2:
                continue
            ns, pod = parts
            if ns not in by_ns:
                by_ns[ns] = []
            by_ns[ns].append((pod, row))

        for ns, pod_list in by_ns.items():
            for pod_name, row in pod_list:
                annotations = {
                    ANNOT_CPU:        str(round(row.get("cpu_usage_mcores", 0), 2)),
                    ANNOT_RAM:        str(round(row.get("ram_usage_mi",     0), 2)),
                    ANNOT_PSI:        str(round(row.get("cpu_psi_some_us",  0), 2)),
                    ANNOT_SCHED:      str(round(row.get("sched_total_ms",   0), 2)),
                    ANNOT_DSTATE:     str(round(row.get("dstate_total_ms",  0), 2)),
                    ANNOT_SOFTIRQ:    str(round(row.get("softirq_total_ms", 0), 2)),
                    ANNOT_POWER:      str(round(row.get("pod_cpu_watts",    0), 4)),
                    ANNOT_DISK_SPACE: str(round(row.get("disk_space_mb",    0), 2)),
                    ANNOT_DISK_USAGE: str(round(row.get("disk_usage_mb",    0), 2)),
                    ANNOT_WINDOW:     str(window_index),
                    ANNOT_TIMESTAMP:  ts,
                }
                patch = {"metadata": {"annotations": annotations}}
                try:
                    self._k8s.patch_namespaced_pod(
                        name=pod_name,
                        namespace=ns,
                        body=patch
                    )
                except ApiException as e:
                    if e.status != 404:
                        logger.warning(f"Patch failed {ns}/{pod_name}: {e.reason}")
                except Exception as e:
                    logger.warning(f"Patch error {ns}/{pod_name}: {e}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _create_namespace(self, ns: str):
        if self._dry_run or not self._k8s:
            logger.info(f"[DRY-RUN] create namespace: {ns}")
            return
        try:
            self._k8s.create_namespace(client.V1Namespace(
                metadata=client.V1ObjectMeta(name=ns)
            ))
            logger.info(f"Created namespace: {ns}")
        except ApiException as e:
            if e.status != 409:   # 409 = already exists
                logger.error(f"Failed to create namespace {ns}: {e}")

    def _delete_namespace(self, ns: str):
        if self._dry_run or not self._k8s:
            logger.info(f"[DRY-RUN] delete namespace: {ns}")
            return
        try:
            self._k8s.delete_namespace(ns)
            self._alive_pods.pop(ns, None)
            logger.info(f"Deleted namespace: {ns}")
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete namespace {ns}: {e}")

    def _create_pod(self, ns: str, pod_name: str):
        if self._dry_run or not self._k8s:
            logger.debug(f"[DRY-RUN] create pod: {ns}/{pod_name}")
            return
        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": ns,
                "labels": {"app": pod_name, "emulation.k8s.io/pod": "true"},
                "annotations": {"emulation.k8s.io/source": "emulation-module"}
            },
            "spec": {
                "nodeName": KWOK_NODE_NAME,
                "tolerations": [{
                    "key": "kwok.x-k8s.io/node",
                    "operator": "Exists",
                    "effect": "NoSchedule"
                }],
                "containers": [{
                    "name": pod_name,
                    "image": "fake-image:latest",
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits":   {"cpu": "1000m","memory": "512Mi"}
                    }
                }]
            }
        }
        try:
            self._kubectl_apply(manifest)
            if ns not in self._alive_pods:
                self._alive_pods[ns] = set()
            self._alive_pods[ns].add(pod_name)
            logger.debug(f"Created pod: {ns}/{pod_name}")
        except Exception as e:
            logger.error(f"Failed to create pod {ns}/{pod_name}: {e}")

    def _delete_pod(self, ns: str, pod_name: str):
        if self._dry_run or not self._k8s:
            logger.debug(f"[DRY-RUN] delete pod: {ns}/{pod_name}")
            return
        try:
            self._k8s.delete_namespaced_pod(
                name=pod_name, namespace=ns,
                body=client.V1DeleteOptions(grace_period_seconds=0)
            )
            if ns in self._alive_pods:
                self._alive_pods[ns].discard(pod_name)
            logger.debug(f"Deleted pod: {ns}/{pod_name}")
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete pod {ns}/{pod_name}: {e}")

    def _kubectl_apply(self, manifest: dict):
        """Apply a manifest via kubectl stdin."""
        process = subprocess.Popen(
            ["kubectl", "apply", "-f", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=json.dumps(manifest))
        if process.returncode != 0:
            raise RuntimeError(f"kubectl apply failed: {stderr}")