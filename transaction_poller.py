"""
transaction_poller.py

Background thread that polls the transaction module every POLL_INTERVAL_S
seconds and adds hotel workloads to the timeline for new Ongoing transactions
belonging to this seller node.

Seller IP detection (automatic, no hardcoding):
  1. SELLER_NODE_IP env var if set (manual override)
  2. UDP socket trick to find the 10.0.1.x interface IP automatically
  3. If neither works, matches ALL sellers (logs a warning)

Requires hostNetwork: true in pod spec so localhost:26657 reaches cometbft.
"""

import base64
import json
import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Set
import urllib.request
import urllib.error

from config import TRANSACTION_API_URL, SELLER_NODE_IP, POLL_INTERVAL_S

logger = logging.getLogger(__name__)

ONGOING_STATUSES    = {"ongoing", "Ongoing", "ONGOING", "OnGoing"}
CLAB_SUBNET_PREFIX  = "10.0."          # ContainerLab experiment network
CLAB_PROBE_TARGET   = "10.0.1.1"       # probe target to find local interface


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
    Polls the transaction module API and injects hotel workloads
    into the emulation timeline for new ongoing transactions.
    """

    def __init__(self, timeline):
        self._timeline     = timeline
        self._seen_hashes: Set[str] = set()
        self._lock         = threading.Lock()
        self._running      = False
        self._thread: Optional[threading.Thread] = None

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
            f"url={TRANSACTION_API_URL}"
        )

    def stop(self):
        self._running = False

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            poll_start = time.time()
            try:
                self._poll_once()
            except Exception as e:
                logger.error(f"Transaction poller error: {e}", exc_info=True)
            elapsed   = time.time() - poll_start
            remaining = POLL_INTERVAL_S - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _poll_once(self):
        raw = self._fetch()
        if raw is None:
            return
        transactions = self._decode(raw)
        if transactions is None:
            return

        now       = datetime.now(timezone.utc)
        new_count = 0

        for tx_record in transactions:
            try:
                if not self._is_relevant(tx_record):
                    continue

                tx_hash = tx_record.get("TxHash", "")
                with self._lock:
                    if tx_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(tx_hash)

                tx           = tx_record.get("Tx", {})
                lease_dur    = int(tx.get("lease_duration", 0))
                tx_start_str = tx.get("tx_start_ts", "")

                lifetime_s = self._remaining_lifetime(tx_start_str, lease_dur, now)
                if lifetime_s <= 0:
                    logger.info(
                        f"Skipping expired tx {tx_hash[:12]}… "
                        f"(lease already ended)"
                    )
                    continue

                job = self._timeline.add_job(
                    app_type         = "hotel",
                    lifetime_seconds = lifetime_s,
                )
                new_count += 1
                logger.info(
                    f"Transaction → Job: hash={tx_hash[:12]}… | "
                    f"job_id={job.job_id} | "
                    f"lifetime={lifetime_s}s (original={lease_dur}s)"
                )

            except Exception as e:
                logger.error(
                    f"Failed to process tx "
                    f"{tx_record.get('TxHash','?')[:12]}: {e}"
                )

        if new_count:
            logger.info(f"Transaction poller: {new_count} new job(s) added")

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _is_relevant(self, tx_record: dict) -> bool:
        status = tx_record.get("Status", "")
        if status not in ONGOING_STATUSES:
            return False

        tx = tx_record.get("Tx", {})
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

    # ── HTTP fetch ────────────────────────────────────────────────────────────

    def _fetch(self) -> Optional[dict]:
        try:
            req = urllib.request.Request(
                TRANSACTION_API_URL,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw_bytes = resp.read()
            return json.loads(raw_bytes.decode("utf-8"))
        except urllib.error.URLError as e:
            logger.warning(f"Transaction API unreachable: {e}")
            return None
        except Exception as e:
            logger.error(f"Transaction API fetch error: {e}")
            return None

    # ── Base64 decode ─────────────────────────────────────────────────────────

    def _decode(self, response: dict) -> Optional[list]:
        try:
            value_b64 = (
                response
                .get("result", {})
                .get("response", {})
                .get("value", "")
            )
            if not value_b64:
                logger.debug("Transaction API returned empty value field")
                return None

            decoded_bytes = base64.b64decode(value_b64)
            decoded_str   = decoded_bytes.decode("utf-8", errors="ignore").rstrip("\x00")
            json_end      = decoded_str.rfind("]") + 1
            if json_end == 0:
                logger.warning("No JSON array in decoded transaction value")
                return None

            transactions = json.loads(decoded_str[:json_end])
            if not isinstance(transactions, list):
                logger.warning("Decoded transaction value is not a list")
                return None
            return transactions

        except Exception as e:
            logger.error(f"Failed to decode transaction value: {e}")
            return None