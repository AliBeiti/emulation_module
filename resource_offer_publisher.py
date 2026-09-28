"""
resource_offer_publisher.py

Background thread that publishes this node's sellable resources to the
global Redis every RESOURCE_OFFER_INTERVAL_S seconds:

  1. GET this node's Admission Control resource-offer endpoint
     (KSENSE_RESOURCE_OFFER_API_SERVICE_HOST/PORT, injected by Kubernetes).
  2. Wrap the response as {"node", "collected_at", "sellable"}.
  3. On a separate Redis connection to REDIS_GLOBAL_HOST:REDIS_GLOBAL_PORT
     (distinct from transaction_poller.py's own "localhost" connection to
     the per-node "emulate" stream):
       - HSET REDIS_GLOBAL_HASH_KEY <node_name> <message>
       - PUBLISH REDIS_GLOBAL_CHANNEL <message>

Never crashes the process and never blocks the other threads: any AC or
Redis failure is logged and the loop simply retries next cycle.
"""

import json
import logging
import socket
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

import redis

from config import (
    NODE_NAME,
    KSENSE_RESOURCE_OFFER_API_SERVICE_HOST,
    KSENSE_RESOURCE_OFFER_API_SERVICE_PORT,
    RESOURCE_OFFER_INTERVAL_S,
    RESOURCE_OFFER_HTTP_TIMEOUT_S,
    REDIS_GLOBAL_HOST,
    REDIS_GLOBAL_PORT,
    REDIS_GLOBAL_HASH_KEY,
    REDIS_GLOBAL_CHANNEL,
)

logger = logging.getLogger(__name__)

CLAB_NODE_PREFIX = "clab-nebula-extended-"


class ResourceOfferPublisher:
    """
    Polls this node's Admission Control /resource_offer endpoint and
    publishes the result to the global Redis (hash + pub/sub) every
    RESOURCE_OFFER_INTERVAL_S seconds.
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._rd: Optional[redis.Redis] = None

        # Resolve node name: env var first, then ContainerLab-style fallback
        self._node_name = NODE_NAME.strip() or f"{CLAB_NODE_PREFIX}{socket.gethostname()}"
        logger.info(f"Resource offer publisher node name resolved: {self._node_name}")

        self._ac_url = (
            f"http://{KSENSE_RESOURCE_OFFER_API_SERVICE_HOST}:"
            f"{KSENSE_RESOURCE_OFFER_API_SERVICE_PORT}/resource_offer"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target = self._loop,
            daemon = True,
            name   = "resource-offer-publisher"
        )
        self._thread.start()
        logger.info(
            f"Resource offer publisher started | node={self._node_name} | "
            f"interval={RESOURCE_OFFER_INTERVAL_S}s | ac_url={self._ac_url} | "
            f"redis={REDIS_GLOBAL_HOST}:{REDIS_GLOBAL_PORT} | "
            f"hash={REDIS_GLOBAL_HASH_KEY} | channel={REDIS_GLOBAL_CHANNEL}"
        )

    def stop(self):
        self._running = False

    # ── Redis connection (separate from transaction_poller.py's) ───────────────

    def _connect(self) -> bool:
        try:
            self._rd = redis.Redis(
                host=REDIS_GLOBAL_HOST,
                port=REDIS_GLOBAL_PORT,
                decode_responses=True,
            )
            self._rd.ping()
            return True
        except redis.exceptions.RedisError as e:
            logger.warning(f"Global Redis unavailable: {e}")
            self._rd = None
            return False

    # ── Loop ─────────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            loop_start = time.time()
            try:
                self._publish_once()
            except Exception as e:
                logger.error(f"Resource offer publisher error: {e}", exc_info=True)

            elapsed   = time.time() - loop_start
            remaining = RESOURCE_OFFER_INTERVAL_S - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _publish_once(self):
        if not KSENSE_RESOURCE_OFFER_API_SERVICE_HOST or not KSENSE_RESOURCE_OFFER_API_SERVICE_PORT:
            logger.warning(
                "KSENSE_RESOURCE_OFFER_API_SERVICE_HOST/PORT not set — "
                "skipping this cycle"
            )
            return

        sellable = self._fetch_offer()
        if sellable is None:
            return

        message = {
            "node":         self._node_name,
            "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sellable":     sellable,
        }
        payload = json.dumps(message)

        if self._rd is None:
            if not self._connect():
                return

        try:
            self._rd.hset(REDIS_GLOBAL_HASH_KEY, self._node_name, payload)
            self._rd.publish(REDIS_GLOBAL_CHANNEL, payload)
            logger.debug(f"Published resource offer for {self._node_name}")
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to publish resource offer to Redis: {e}")
            self._rd = None  # force reconnect next cycle

    # ── Admission Control HTTP fetch ────────────────────────────────────────────

    def _fetch_offer(self) -> Optional[dict]:
        try:
            req = urllib.request.Request(
                self._ac_url,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=RESOURCE_OFFER_HTTP_TIMEOUT_S) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.URLError as e:
            logger.warning(f"Resource offer API unreachable: {e}")
            return None
        except Exception as e:
            logger.error(f"Resource offer fetch/decode error: {e}")
            return None
