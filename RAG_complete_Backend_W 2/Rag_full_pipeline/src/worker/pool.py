"""
Worker Pool  —  RabbitMQ feeders for the StagePipeline
=======================================================
PDFWorker threads consume messages from RabbitMQ and submit DocJobs into
the StagePipeline.  All OCR, embedding, and storage happens inside the
stage workers — PDFWorkers themselves do NO heavy processing.

Architecture
------------
  PDFWorker-0 ─┐
  PDFWorker-1 ─┼──→  StagePipeline._doc_q  →  stage workers
  PDFWorker-2 ─┘

Why separate feeder threads?
  RabbitMQ message acks must happen on the same thread/channel that received
  the message (pika is not thread-safe).  Each feeder owns its own pika
  connection+channel, so acks are correct.

Heartbeat safety
  pika processes heartbeats only inside process_data_events().  Any blocking
  work inside _on_message() (file I/O, queue put) starves heartbeats and
  drops the MQ connection after 60 s.

  Fix: _on_message() enqueues (body, ch, method) into a local _inbox queue
  and returns immediately.  A dedicated _submit_thread does all file I/O and
  pipeline.submit(); it schedules the ack back onto the pika thread via
  connection.add_callback_threadsafe() so acks stay on the correct thread.
"""

import logging
import threading
import time
import uuid
from pathlib import Path
from queue import Queue, Empty

import pika

from src.config import MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE, MAX_RETRIES
from src.ingestion.pipeline.datatypes import DocJob
from src.models.schemas import JobPayload
from src.database.rabbitmq_broker import rabbit_connect, publish_job

logger = logging.getLogger(__name__)

# Max messages buffered locally before _on_message blocks (should never fill
# because prefetch_count=1 limits how many unacked msgs pika holds).
_INBOX_MAXSIZE = 8


class PDFWorker:
    """
    Single RabbitMQ consumer thread.
    Reads job messages and submits DocJobs to the shared StagePipeline.
    Does NOT load any ML models — all heavy work is done by stage workers.
    """

    def __init__(self, worker_id: str, rsm, pipeline, shutdown: threading.Event):
        self.worker_id = worker_id
        self.rsm       = rsm
        self.pipeline  = pipeline   # RAGPipeline — exposes .submit(DocJob)
        self.shutdown  = shutdown
        self._conn     = None
        self._ch       = None
        # Local buffer: _on_message drops work here and returns immediately
        self._inbox: Queue = Queue(maxsize=_INBOX_MAXSIZE)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        self._start_heartbeat()
        self._start_submit_thread()
        while not self.shutdown.is_set():
            try:
                self._connect()
                self._consume()
            except pika.exceptions.AMQPConnectionError as e:
                logger.warning("[Feeder %s] RabbitMQ connection lost, retrying in 5s: %s",
                               self.worker_id[:8], e)
                time.sleep(5)
            except Exception as e:
                logger.error("[Feeder %s] Unexpected error: %s",
                             self.worker_id[:8], e, exc_info=True)
                time.sleep(5)
        self._stop_heartbeat()
        logger.info("[Feeder %s] Stopped", self.worker_id[:8])

    def _connect(self):
        self._conn = rabbit_connect()
        self._ch   = self._conn.channel()
        # prefetch_count=1 → fair dispatch; feeder acks after submit to pipeline
        self._ch.basic_qos(prefetch_count=1)
        for q in (MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE):
            self._ch.basic_consume(queue=q, on_message_callback=self._on_message)

    def _consume(self):
        while not self.shutdown.is_set():
            self._conn.process_data_events(time_limit=1)

    # ── Message handler (pika thread — must not block) ────────────────────────

    def _on_message(self, ch, method, props, body):
        """
        Called on the pika I/O thread.  Returns immediately after enqueuing
        so process_data_events() can continue sending heartbeats.
        All file I/O and pipeline submission happens in _submit_thread.
        """
        try:
            self._inbox.put((body, ch, method), timeout=2)
        except Exception:
            # Inbox full or interrupted — nack so RabbitMQ requeues
            logger.warning("[Feeder %s] Inbox full, nacking message", self.worker_id[:8])
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    # ── Submit thread (off the pika thread — may block) ───────────────────────

    def _start_submit_thread(self):
        threading.Thread(
            target=self._submit_loop,
            daemon=True,
            name=f"submit-{self.worker_id[:8]}",
        ).start()

    def _submit_loop(self):
        while not self.shutdown.is_set():
            try:
                body, ch, method = self._inbox.get(timeout=1)
            except Empty:
                continue
            self._handle(body, ch, method)

    def _handle(self, body: bytes, ch, method):
        job = None
        delivery_tag = method.delivery_tag
        try:
            job   = JobPayload.from_json(body)
            fpath = Path(job.file_path)
            if not fpath.exists():
                raise FileNotFoundError(f"PDF missing on disk: {fpath}")

            raw_bytes = fpath.read_bytes()
            logger.info("[Feeder %s] Submitting '%s' (%.1f KB) to pipeline",
                        self.worker_id[:8], job.filename, len(raw_bytes) / 1024)

            doc_job = DocJob(
                file_id=job.file_id,
                session_id=job.session_id,
                filename=job.filename,
                raw_bytes=raw_bytes,
                user_id=job.user_id,
                dept_id=job.dept_id,
                upload_id=getattr(job, "upload_id", None),
                upload_type=getattr(job, "upload_type", "user"),
            )

            # Submit to stage pipeline (uses _put() internally — non-blocking on shutdown)
            self.pipeline.submit(doc_job)

            # Ack must run on the pika thread — schedule it there
            self._schedule_ack(delivery_tag)

        except Exception as e:
            logger.error("[Feeder %s] Error handling '%s': %s",
                         self.worker_id[:8], getattr(job, "filename", "unknown"),
                         e, exc_info=True)
            self._schedule_ack(delivery_tag)

            if job and getattr(job, "retry", 0) < MAX_RETRIES:
                job.retry = getattr(job, "retry", 0) + 1
                logger.info("[Feeder %s] Requeueing '%s' (attempt %d/%d)",
                            self.worker_id[:8], job.filename, job.retry, MAX_RETRIES)
                publish_job(job)
            elif job:
                self.rsm.update_stage(job.file_id, job.session_id, "error", 0,
                                      extra={"error": str(e)})
                self.rsm.incr_stat("total_failed")

    def _schedule_ack(self, delivery_tag):
        """Schedule basic_ack on the pika I/O thread via add_callback_threadsafe."""
        conn = self._conn
        if conn is None or not conn.is_open:
            return
        try:
            ch = self._ch
            conn.add_callback_threadsafe(
                lambda: ch.basic_ack(delivery_tag=delivery_tag)
                if ch and ch.is_open else None
            )
        except Exception as e:
            logger.warning("[Feeder %s] Could not schedule ack: %s", self.worker_id[:8], e)

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def _start_heartbeat(self):
        self._hb_stop = threading.Event()
        threading.Thread(target=self._hb_loop, daemon=True,
                         name=f"hb-{self.worker_id[:8]}").start()

    def _stop_heartbeat(self):
        self._hb_stop.set()

    def _hb_loop(self):
        while not self._hb_stop.is_set():
            try:
                self.rsm.worker_heartbeat(self.worker_id)
            except Exception:
                pass
            self._hb_stop.wait(timeout=5)


class WorkerPool:
    """
    Manages a pool of PDFWorker (RabbitMQ feeder) threads.
    Feeders are lightweight — 3 feeders is sufficient to keep the
    stage pipeline's _doc_q saturated at all times.
    """

    def __init__(self, rsm, pipeline, n: int = 3):
        self.rsm      = rsm
        self.pipeline = pipeline
        self.n        = n
        self.shutdown = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self):
        logger.info("Starting WorkerPool with %d RabbitMQ feeder(s)", self.n)
        for i in range(self.n):
            wid    = str(uuid.uuid4())
            worker = PDFWorker(wid, self.rsm, self.pipeline, self.shutdown)
            t = threading.Thread(
                target=worker.run,
                daemon=True,
                name=f"mq-feeder-{i}",
            )
            self._threads.append(t)
            t.start()
            logger.info("  Feeder %d started (id=%s)", i, wid[:8])

    def stop(self, timeout: float = 30.0):
        logger.info("Stopping WorkerPool…")
        self.shutdown.set()
        for t in self._threads:
            t.join(timeout=timeout)
        logger.info("WorkerPool stopped")
