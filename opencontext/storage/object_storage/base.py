"""
Object storage abstract base class
"""

from abc import ABC, abstractmethod


class IObjectStorage(ABC):
    @abstractmethod
    async def upload(self, data: bytes, key: str, content_type: str) -> str:
        """Upload binary data, return HTTPS URL (or local path for LocalBackend)"""

    @abstractmethod
    def get_url(self, key: str) -> str:
        """Get HTTPS access URL for the given key"""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an object by key"""

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources (e.g. close HTTP sessions)"""
