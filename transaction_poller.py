"""
transaction_poller.py

Background thread that reads the Redis stream "emulate" every POLL_INTERVAL_S
seconds and adds workloads to the timeline for new OnGoing transactions
belonging to this seller node.

Replaces the old CometBFT HTTP polling approach with Redis stream consumer
groups. Each message is acknowledged (XACK) immediately after processing so
it is never delivered again, even across pod restarts.

Seller IP detection (automatic, no hardcoding):
  1. SELLER_NODE_IP env var if set (manual override)
  2. UDP socket trick to find the 10.0.x interface IP automatically
  3. If neither works, matches ALL sellers (logs a warning)

Requires hostNetwork: true in pod spec so localhost:6379 reaches Redis.
"""

import json
import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import redis

from config import SELLER_NODE_IP, POLL_INTERVAL_S
from event_logger import EventLogger

logger = logging.getLogger(__name__)

REDIS_HOST         = "localhost"
REDIS_PORT         = 6379
STREAM_KEY         = "emulate"
STREAM_FIELD       = "ongoingtx"
CONSUMER_GROUP     = "emulation-module"

CLAB_SUBNET_PREFIX = "10.0."
CLAB_PROBE_TARGET  = "10.0.1.1"

_event_logger = EventLogger()


def detect_clab_ip() -> str:
    """
    Automatically detect the ContainerLab experiment network IP.

    Strategy: send a UDP packet (no data) to a target in the 10.0.x subnet.
    The OS selects the correct outgoing interface, revealing the local IP.
    Works reliably with hostNetwork: true since the pod shares the node's
    network namespace.

    Returns empty string if detection fails.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect((CLAB_PROBE_TARGET, 80))
        ip = s.getsockname()[0]
        s.close()
        if ip.startswith(CLAB_SUBNET_PREFIX):
            return ip
    except Exception:
        pass

    # fallback: scan all IPs via getaddrinfo
    try:
        hostname = socket.gethostname()
        for res in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = res[4][0]
            if ip.startswith(CLAB_SUBNET_PREFIX):
                return ip
    except Exception:
        pass

    return ""


class TransactionPoller:
    """
    Reads the Redis stream "emulate" and injects workloads into the emulation
    timeline for new OnGoing transactions targeting this seller node.
    """

    def __init__(self, timeline):
        self._timeline = timeline
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._rd: Optional[redis.Redis] = None

        # Resolve seller IP: env var first, then auto-detect
        self._seller_ip = SELLER_NODE_IP.strip()
        if not self._seller_ip:
            self._seller_ip = detect_clab_ip()

        if self._seller_ip:
            logger.info(f"Seller IP resolved: {self._seller_ip}")
        else:
            logger.warning(
                "Could not determine seller IP — "
                "transaction poller will match ALL sellers. "
                "Ensure hostNetwork: true is set in pod spec."
            )

        # Consumer name is the seller IP (or hostname as fallback)
        self._consumer_name = self._seller_ip or socket.gethostname()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="tx-poller"
        )
        self._thread.start()
        logger.info(
            f"Transaction poller started | "
            f"seller_ip={self._seller_ip or 'ANY'} | "
            f"interval={POLL_INTERVAL_S}s | "
            f"redis={REDIS_HOST}:{REDIS_PORT} | "
            f"stream={STREAM_KEY} | "
            f"group={CONSUMER_GROUP} | "
            f"consumer={self._consumer_name}"
        )

    def stop(self):
        self._running = False

    # ── Redis connection ──────────────────────────────────────────────────────

    def _connect(self) -> bool:
        """Connect to Redis and create the consumer group if it doesn't exist."""
        try:
            self._rd = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            self._rd.ping()

            # Create consumer group; MKSTREAM creates the stream if absent
            try:
                self._rd.xgroup_create(
                    STREAM_KEY,
                    CONSUMER_GROUP,
                    id="0",          # start from the very beginning
                    mkstream=True
                )
                logger.info(
                    f"Consumer group '{CONSUMER_GROUP}' created on "
                    f"stream '{STREAM_KEY}'"
                )
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists — normal on restart
                    logger.debug(
                        f"Consumer group '{CONSUMER_GROUP}' already exists"
                    )
                else:
                    raise

            return True

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            self._rd = None
            return False

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            # Ensure Redis is connected
            if self._rd is None:
                if not self._connect():
                    logger.warning(
                        "Redis unavailable — retrying in "
                        f"{POLL_INTERVAL_S}s"
                    )
                    time.sleep(POLL_INTERVAL_S)
                    continue

            poll_start = time.time()
            try:
                self._poll_once()
            except redis.exceptions.ConnectionError as e:
                logger.error(f"Redis connection lost: {e} — will reconnect")
                self._rd = None
            except Exception as e:
                logger.error(f"Transaction poller error: {e}", exc_info=True)

            elapsed   = time.time() - poll_start
            remaining = POLL_INTERVAL_S - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _poll_once(self):
        # Do not process transactions during calibration phase
        from api import state
        if not state.get("calibration_done", False):
            logger.debug("Calibration in progress — skipping transaction poll")
            return

        # Read new messages: ">" means only undelivered messages
        results = self._rd.xreadgroup(
            groupname=CONSUMER_GROUP,
            consumername=self._consumer_name,
            streams={STREAM_KEY: ">"},
            count=100,       # max messages per poll
            block=None       # non-blocking
        )
        received_unix = time.time()   # captured immediately after Redis delivers messages

        if not results:
            return

        # results structure: [ (stream_key, [(msg_id, {field: value}), ...]) ]
        _, messages = results[0]
        now       = datetime.now(timezone.utc)
        new_count = 0

        for msg_id, fields in messages:
            try:
                raw_json = fields.get(STREAM_FIELD)
                if not raw_json:
                    logger.warning(
                        f"Message {msg_id} missing field '{STREAM_FIELD}' "
                        f"— acknowledging and skipping"
                    )
                    self._rd.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                    continue

                tx_record = json.loads(raw_json)

                if not self._is_relevant(tx_record):
                    # Acknowledge even irrelevant messages so they don't
                    # accumulate in the pending list
                    self._rd.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                    continue

                tx_hash      = tx_record.get("tx_hash", "")
                tx           = tx_record.get("tx", {})
                lease_dur    = int(tx.get("lease_duration", 0))
                tx_start_str = tx.get("tx_start_ts", "")

                lifetime_s = self._remaining_lifetime(tx_start_str, lease_dur, now)
                if lifetime_s <= 0:
                    logger.info(
                        f"Skipping expired tx {tx_hash[:12]}… "
                        f"(lease already ended)"
                    )
                    self._rd.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                    continue

                buyer_name = tx.get("buyer", {}).get("name", "unknown") or "unknown"
                job = self._timeline.add_job(
                    app_type         = "hotel",
                    lifetime_seconds = lifetime_s,
                    buyer_name       = buyer_name,
                )
                new_count += 1
                logger.info(
                    f"Transaction → Job: hash={tx_hash[:12]}… | "
                    f"job_id={job.job_id} | buyer={buyer_name} | "
                    f"lifetime={lifetime_s}s (original={lease_dur}s)"
                )

                _event_logger.log(
                    tx_record     = tx_record,
                    msg_id        = msg_id,
                    job           = job,
                    received_unix = received_unix,
                    composition   = self._timeline.get_composition(),
                )

                # Acknowledge after successful processing
                self._rd.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)

            except Exception as e:
                logger.error(
                    f"Failed to process message {msg_id}: {e}",
                    exc_info=True
                )
                # Do NOT acknowledge on error — message stays pending
                # and can be inspected or reprocessed manually

        if new_count:
            logger.info(f"Transaction poller: {new_count} new job(s) added")

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _is_relevant(self, tx_record: dict) -> bool:
        status = tx_record.get("status", "")
        if status.lower() != "ongoing":
            return False

        tx = tx_record.get("tx", {})
        if tx.get("type") != "transfer":
            return False

        if self._seller_ip:
            seller_ip = tx.get("seller", {}).get("ip", "")
            if seller_ip != self._seller_ip:
                return False

        return True

    # ── Lifetime calculation ──────────────────────────────────────────────────

    def _remaining_lifetime(
        self,
        tx_start_str: str,
        lease_duration_s: int,
        now: datetime,
    ) -> int:
        if not tx_start_str:
            return lease_duration_s
        try:
            ts = tx_start_str.replace("Z", "+00:00")
            try:
                start_dt = datetime.fromisoformat(ts)
            except ValueError:
                ts_clean = ts.split("+")[0].split("Z")[0]
                start_dt = datetime.fromisoformat(ts_clean).replace(
                    tzinfo=timezone.utc
                )
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            elapsed   = (now - start_dt).total_seconds()
            remaining = int(lease_duration_s - elapsed)
            return max(remaining, 0)
        except Exception as e:
            logger.warning(
                f"Could not parse tx_start_ts '{tx_start_str}': {e} "
                f"— using full lease_duration"
            )
            return lease_duration_s