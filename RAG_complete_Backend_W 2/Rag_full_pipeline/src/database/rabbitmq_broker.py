import pika, time, logging, json
from typing import List
from src.config import RABBIT_HOST, RABBIT_PORT, RABBIT_USER, RABBIT_PASS, RABBIT_VHOST, \
    MQ_EXCHANGE_JOBS, MQ_EXCHANGE_DLX, MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE, MQ_QUEUE_DEAD, \
    RK_PRIORITY, RK_NORMAL, RK_LARGE, PRIORITY_MAX_KB
from src.models.schemas import JobPayload

logger = logging.getLogger(__name__)

def _rabbit_params():
    return pika.ConnectionParameters(
        host=RABBIT_HOST, port=RABBIT_PORT, virtual_host=RABBIT_VHOST,
        credentials=pika.PlainCredentials(RABBIT_USER, RABBIT_PASS),
        heartbeat=60, 
        blocked_connection_timeout=30, 
        connection_attempts=3, 
        retry_delay=2,
        socket_timeout=10,
    )

def rabbit_connect():
    params = _rabbit_params()
    try:
        conn = pika.BlockingConnection(params)
        logger.info(f"[RabbitMQ] Connected {RABBIT_HOST}:{RABBIT_PORT}")
        return conn
    except Exception as e:
        logger.warning(f"RabbitMQ connection failed: {e}")
        raise e

def setup_topology(conn):
    if not conn or not conn.is_open: return
    ch = conn.channel()
    ch.exchange_declare(MQ_EXCHANGE_JOBS, exchange_type="topic", durable=True)
    ch.exchange_declare(MQ_EXCHANGE_DLX,  exchange_type="fanout", durable=True)
    ch.queue_declare(MQ_QUEUE_DEAD, durable=True)
    ch.queue_bind(MQ_QUEUE_DEAD, MQ_EXCHANGE_DLX)
    dlx_args = {"x-dead-letter-exchange": MQ_EXCHANGE_DLX, "x-message-ttl": 3_600_000}
    ch.queue_declare(MQ_QUEUE_PRIORITY, durable=True, arguments={**dlx_args, "x-max-priority": 10})
    ch.queue_bind(MQ_QUEUE_PRIORITY, MQ_EXCHANGE_JOBS, routing_key=RK_PRIORITY)
    ch.queue_declare(MQ_QUEUE_NORMAL, durable=True, arguments=dlx_args)
    ch.queue_bind(MQ_QUEUE_NORMAL, MQ_EXCHANGE_JOBS, routing_key=RK_NORMAL)
    ch.queue_declare(MQ_QUEUE_LARGE, durable=True, arguments=dlx_args)
    ch.queue_bind(MQ_QUEUE_LARGE, MQ_EXCHANGE_JOBS, routing_key=RK_LARGE)
    ch.close()

def publish_batch(jobs: List[JobPayload]):
    """Creates a fresh connection per batch for maximum stability."""
    if not jobs: return
    conn = None
    try:
        conn = rabbit_connect()
        ch = conn.channel()
        ch.confirm_delivery()
        for job in jobs:
            rk = job.routing_key()
            props = pika.BasicProperties(
                delivery_mode=2, content_type="application/json", message_id=job.job_id,
                headers={"session_id": job.session_id, "file_id": job.file_id, "filename": job.filename, "retry": job.retry},
            )
            ch.basic_publish(MQ_EXCHANGE_JOBS, rk, job.to_json().encode(), props)
        ch.close()
    except Exception as e:
        logger.error(f"Failed to publish batch: {e}")
        raise e
    finally:
        if conn and conn.is_open:
            try: conn.close()
            except: pass

def publish_job(job: JobPayload):
    """Fallback for single job publishing."""
    publish_batch([job])
