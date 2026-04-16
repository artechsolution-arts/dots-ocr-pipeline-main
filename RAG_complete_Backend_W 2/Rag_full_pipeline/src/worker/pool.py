import threading, uuid, logging, time, pika
from pathlib import Path
from typing import Optional, List
from src.config import MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE, MQ_QUEUE_DEAD, MAX_RETRIES
from src.models.schemas import JobPayload
from src.database.rabbitmq_broker import rabbit_connect

logger = logging.getLogger(__name__)

class PDFWorker:
    def __init__(self, worker_id, rsm, pipeline, shutdown):
        self.worker_id, self.rsm, self.pipeline, self.shutdown = worker_id, rsm, pipeline, shutdown
        self._conn, self._ch = None, None

    def run(self):
        self._start_heartbeat()
        while not self.shutdown.is_set():
            try: self._connect(); self._consume()
            except Exception as e: logger.error(f"[Worker] Error: {e}"); time.sleep(5)
        self._stop_heartbeat()

    def _connect(self):
        self._conn = rabbit_connect(); self._ch = self._conn.channel(); self._ch.basic_qos(prefetch_count=1)
        for q in (MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE): self._ch.basic_consume(queue=q, on_message_callback=self._on_message)

    def _consume(self):
        while not self.shutdown.is_set(): self._conn.process_data_events(time_limit=1)

    def _on_message(self, ch, method, props, body):
        job = None
        try:
            job = JobPayload.from_json(body)
            if not job.file_path or not Path(job.file_path).exists():
                raise FileNotFoundError(f"File not found: {job.file_path}")
                
            raw = Path(job.file_path).read_bytes()
            self.pipeline.process_pdf(raw_bytes=raw, filename=job.filename, user_id=job.user_id, dept_id=job.dept_id, file_id=job.file_id, session_id=job.session_id, upload_type=job.upload_type, chat_id=job.chat_id, retry=job.retry, upload_id=job.upload_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logger.error(f"[Worker] Error processing {getattr(job, 'filename', 'unknown')}: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)  # Ack original to prevent DLX duplication
            if job:
                if job.retry < MAX_RETRIES:
                    job.retry += 1
                    from src.database.rabbitmq_broker import publish_job
                    publish_job(job)
                else:
                    self.rsm.update_stage(job.file_id, job.session_id, "error", 0, extra={"error": str(e)})
                    self.rsm.incr_stat("total_failed")

    def _start_heartbeat(self):
        self._hb_stop = threading.Event(); threading.Thread(target=self._hb_loop, daemon=True).start()
    def _stop_heartbeat(self): self._hb_stop.set()
    def _hb_loop(self):
        while not self._hb_stop.is_set():
            try: self.rsm.worker_heartbeat(self.worker_id)
            except: pass
            self._hb_stop.wait(timeout=5)

class WorkerPool:
    def __init__(self, rsm, pipeline, n=4):
        self.rsm, self.pipeline, self.n, self.shutdown, self._threads = rsm, pipeline, n, threading.Event(), []
    def start(self):
        for i in range(self.n):
            wid = str(uuid.uuid4()); worker = PDFWorker(wid, self.rsm, self.pipeline, self.shutdown)
            t = threading.Thread(target=worker.run, daemon=True); self._threads.append(t); t.start()
    def stop(self, timeout=30.0):
        self.shutdown.set(); [t.join(timeout=timeout) for t in self._threads]
