import logging
import os

import oss2


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class OSSHandler:
    def __init__(self, conf):
        auth = oss2.Auth(conf["access_key_id"], conf["access_key_secret"])
        self.bucket = oss2.Bucket(auth, conf["endpoint"], conf["bucket_name"])
        self.url_prefix = conf["url_prefix"].rstrip("/") + "/"
        self.env = conf.get("env", "dev")

    def upload_task_file(self, local_path, task_id, file_type="video"):
        if not os.path.exists(local_path):
            logging.error("File not found: %s", local_path)
            return None

        ext = os.path.splitext(local_path)[1].lstrip(".") or "dat"
        if file_type == "video":
            remote_path = f"videos/{self.env}/tasks/{task_id}/video/main.{ext}"
        else:
            remote_path = f"videos/{self.env}/tasks/{task_id}/artifacts/{file_type}.{ext}"

        try:
            logging.info("Uploading %s -> %s", local_path, remote_path)
            oss2.resumable_upload(
                self.bucket,
                remote_path,
                local_path,
                multipart_threshold=100 * 1024,
                part_size=100 * 1024,
                num_threads=2,
            )
            file_url = f"{self.url_prefix}{remote_path}"
            logging.info("Upload succeeded: %s", file_url)
            return file_url
        except Exception as exc:
            logging.error("Upload failed: %s", exc)
            return None
