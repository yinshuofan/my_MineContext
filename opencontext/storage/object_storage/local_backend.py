# -*- coding: utf-8 -*-

"""
Local filesystem object storage backend
"""

import os

from opencontext.storage.object_storage.base import IObjectStorage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LocalBackend(IObjectStorage):
    def __init__(self, base_dir: str = "./uploads/media"):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"LocalBackend initialized, base_dir={self.base_dir}")

    async def upload(self, data: bytes, key: str, content_type: str) -> str:
        """Write data to {base_dir}/{key}, return absolute file path."""
        file_path = os.path.join(self.base_dir, key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(data)
        logger.debug(f"Uploaded {len(data)} bytes to {file_path}")
        return file_path

    def get_url(self, key: str) -> str:
        """Return absolute file path for the given key."""
        return os.path.join(self.base_dir, key)

    async def delete(self, key: str) -> bool:
        """Delete file at {base_dir}/{key}. Return True on success, False if not found."""
        file_path = os.path.join(self.base_dir, key)
        if not os.path.exists(file_path):
            logger.warning(f"File not found for deletion: {file_path}")
            return False
        os.remove(file_path)
        logger.debug(f"Deleted {file_path}")
        return True

    async def close(self) -> None:
        """No-op for local backend."""
        pass
