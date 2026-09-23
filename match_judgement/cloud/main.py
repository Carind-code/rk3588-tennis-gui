import json
import logging
import os
import socket
import threading
import time
from queue import Empty, Queue

from iot_client import AliyunIoTClient
from oss_handler import OSSHandler


SOCKET_PATH = "/tmp/tennis_ipc.sock"
SENT_COUNT = 0
count_lock = threading.Lock()


def iter_json_messages(text):
    """Accept one JSON object or newline-delimited JSON objects."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines and text.strip():
        lines = [text.strip()]
    for line in lines:
        yield json.loads(line)


def handle_connection(conn, data_queue):
    chunks = []
    while True:
        data = conn.recv(65536)
        if not data:
            break
        chunks.append(data)

    if not chunks:
        return

    text = b"".join(chunks).decode("utf-8").strip()
    if not text:
        return

    for payload in iter_json_messages(text):
        if isinstance(payload, dict):
            data_queue.put(payload)
        else:
            logging.warning("IPC payload is not a JSON object: %r", payload)


def socket_server_thread(data_queue):
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o777)
    server.listen(16)
    logging.info("UDS Server started: %s", SOCKET_PATH)

    while True:
        conn, _ = server.accept()
        try:
            handle_connection(conn, data_queue)
        except Exception:
            logging.exception("Failed to read IPC message")
        finally:
            conn.close()


def monitor_thread(data_queue):
    global SENT_COUNT
    last_count = 0
    last_time = time.time()

    while True:
        time.sleep(1)
        now = time.time()
        with count_lock:
            current_total = SENT_COUNT

        delta = current_total - last_count
        speed = delta / max(now - last_time, 1e-6)
        logging.info(
            "[Monitor] coord_speed=%.2f pts/s queue=%d total=%d",
            speed,
            data_queue.qsize(),
            current_total,
        )
        last_count = current_total
        last_time = now


def upload_result_task(oss, iot, task_data):
    task_id = task_data.get("task_id") or f"task_{int(time.time())}"
    video_file = task_data.get("video_path")
    csv_file = task_data.get("csv_path")
    event_csv_file = task_data.get("event_csv_path")
    judgement_json_file = task_data.get("judgement_json_path")

    result_msg = {
        "type": "video_result",
        "task_id": task_id,
        "status": "done",
        "created_at": time.time(),
    }

    if video_file:
        if os.path.exists(video_file):
            logging.info("[%s] uploading video: %s", task_id, video_file)
            video_url = oss.upload_task_file(video_file, task_id, file_type="video")
            if video_url:
                result_msg["video_url"] = video_url
            else:
                result_msg["status"] = "upload_failed"
        else:
            result_msg["status"] = "video_missing"
            logging.warning("[%s] video file missing: %s", task_id, video_file)

    if csv_file:
        if os.path.exists(csv_file):
            logging.info("[%s] uploading trajectory: %s", task_id, csv_file)
            csv_url = oss.upload_task_file(csv_file, task_id, file_type="trajectory")
            if csv_url:
                result_msg["trajectory_url"] = csv_url
            elif result_msg["status"] == "done":
                result_msg["status"] = "upload_failed"
        else:
            logging.warning("[%s] csv file missing: %s", task_id, csv_file)

    for local_path, file_type, result_key in (
        (event_csv_file, "bounce_events", "bounce_events_url"),
        (judgement_json_file, "judgement", "judgement_url"),
    ):
        if not local_path:
            continue
        if not os.path.exists(local_path):
            logging.warning("[%s] artifact missing: %s", task_id, local_path)
            continue
        logging.info("[%s] uploading %s: %s", task_id, file_type, local_path)
        url = oss.upload_task_file(local_path, task_id, file_type=file_type)
        if url:
            result_msg[result_key] = url
        elif result_msg["status"] == "done":
            result_msg["status"] = "upload_failed"

    result_msg.update(
        {
            key: task_data[key]
            for key in ("frames", "valid_points", "fps", "bounce_count", "judgement_count", "score")
            if key in task_data
        }
    )
    iot.send_data(result_msg)
    logging.info("[%s] result notification sent: %s", task_id, result_msg["status"])


def flush_coords(iot, config, batch_buffer):
    global SENT_COUNT
    payload = {
        "type": "coord_batch",
        "device_id": config["iot"]["device_name"],
        "count": len(batch_buffer),
        "data_list": batch_buffer,
        "sent_at": time.time(),
    }
    iot.send_data(payload)
    with count_lock:
        SENT_COUNT += len(batch_buffer)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        logging.error("Failed to read config.json: %s", exc)
        return

    oss = OSSHandler(config["oss"])
    iot = AliyunIoTClient(config["iot"])
    iot.connect()

    data_queue = Queue()
    threading.Thread(target=socket_server_thread, args=(data_queue,), daemon=True).start()
    threading.Thread(target=monitor_thread, args=(data_queue,), daemon=True).start()

    logging.info("Cloud process is ready. Waiting for AI process IPC messages...")

    batch_size = int(config.get("coord_batch_size", 20))
    max_wait_time = float(config.get("coord_max_wait_time", 0.5))
    batch_buffer = []
    last_flush_time = time.time()

    while True:
        try:
            task_data = data_queue.get(timeout=0.1)
            msg_type = task_data.get("type", "coord")

            if msg_type == "video_task":
                threading.Thread(
                    target=upload_result_task,
                    args=(oss, iot, task_data),
                    daemon=True,
                ).start()
            elif msg_type == "coord":
                batch_buffer.append(task_data)
            else:
                logging.warning("Unknown IPC message type: %s", msg_type)
        except Empty:
            pass
        except Exception:
            logging.exception("Failed to handle queued IPC message")

        now = time.time()
        if batch_buffer and (
            len(batch_buffer) >= batch_size or now - last_flush_time > max_wait_time
        ):
            flush_coords(iot, config, batch_buffer)
            batch_buffer = []
            last_flush_time = now


if __name__ == "__main__":
    run()
