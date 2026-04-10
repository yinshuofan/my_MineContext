
from opencontext.storage.object_storage.base import IObjectStorage
from opencontext.storage.object_storage.global_object_storage import (
    GlobalObjectStorage,
    get_object_storage,
)

__all__ = ["IObjectStorage", "get_object_storage", "GlobalObjectStorage"]
