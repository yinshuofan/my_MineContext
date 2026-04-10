
"""
Global object storage singleton
"""

import os
import threading

from opencontext.storage.object_storage.base import IObjectStorage
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GlobalObjectStorage:
    _instance: IObjectStorage | None = None
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def initialize(cls) -> IObjectStorage | None:
        """Initialize from config. Call once at startup. Returns the instance or None."""
        if cls._initialized:
            return cls._instance
        with cls._lock:
            if cls._initialized:
                return cls._instance
            cls._instance = cls._create_from_config()
            cls._initialized = True
            return cls._instance

    @classmethod
    def _create_from_config(cls) -> IObjectStorage | None:
        """Create backend from config/config.yaml object_storage section."""
        from opencontext.config.global_config import get_config

        config = get_config("object_storage")
        if not config:
            logger.info("object_storage not configured, disabled")
            return None

        backend_type = config.get("backend", "local")

        if backend_type == "s3":
            from opencontext.storage.object_storage.s3_backend import S3CompatibleBackend

            s3_config = config.get("s3", {})
            access_key_id = s3_config.get("access_key_id", "")
            secret_access_key = s3_config.get("secret_access_key", "")

            # Fallback: reuse Volcengine credentials from VikingDB config or env vars
            if not access_key_id or not secret_access_key:
                storage_config = get_config("storage") or {}
                for b in storage_config.get("backends", []):
                    if b.get("backend") == "vikingdb":
                        b_config = b.get("config", {})
                        if not access_key_id:
                            access_key_id = b_config.get("access_key_id", "")
                        if not secret_access_key:
                            secret_access_key = b_config.get("secret_access_key", "")
                        break
                if not access_key_id:
                    access_key_id = os.environ.get("VIKINGDB_ACCESS_KEY_ID", "")
                if not secret_access_key:
                    secret_access_key = os.environ.get("VIKINGDB_SECRET_ACCESS_KEY", "")

            instance = S3CompatibleBackend(
                endpoint=s3_config.get("endpoint", "tos-s3-cn-beijing.volces.com"),
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                bucket=s3_config.get("bucket", "opencontext-media"),
                region=s3_config.get("region", "cn-beijing"),
                use_https=s3_config.get("use_https", True),
            )
            logger.info("Object storage initialized with S3-compatible backend")
            return instance
        elif backend_type == "local":
            from opencontext.storage.object_storage.local_backend import LocalBackend

            local_config = config.get("local", {})
            instance = LocalBackend(base_dir=local_config.get("base_dir", "./uploads/media"))
            logger.info("Object storage initialized with local backend")
            return instance
        else:
            logger.warning(f"Unknown object_storage backend: {backend_type}")
            return None

    @classmethod
    def get_instance(cls) -> IObjectStorage | None:
        """Get the singleton. Returns None if not initialized or not configured."""
        if not cls._initialized:
            cls.initialize()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False


def get_object_storage() -> IObjectStorage | None:
    """Convenience function - get object storage singleton. Returns None if not configured."""
    return GlobalObjectStorage.get_instance()
