#!/usr/bin/env python

"""
Event Model Definition
Unified streaming event structure
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from opencontext.utils.time_utils import now as tz_now

from .enums import EventType, NodeType, WorkflowStage


@dataclass
class StreamEvent:
    """Unified streaming event - the single representation for all event types"""

    type: EventType  # Event type
    content: str  # Event content/message
    stage: WorkflowStage | None = None  # Workflow stage
    node: NodeType | None = None  # Node type (if it's a node event)
    progress: float = 0.0  # Progress (0.0-1.0)
    timestamp: datetime = field(default_factory=tz_now)
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamEvent":
        """Create an event from a dictionary"""
        return cls(
            type=EventType(data["type"]),
            content=data["content"],
            stage=WorkflowStage(data.get("stage")) if data.get("stage") else None,
            node=NodeType(data.get("node")) if data.get("node") else None,
            progress=data.get("progress", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization"""
        result = {
            "type": self.type.value,
            "content": self.content,
            "progress": self.progress,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.stage:
            result["stage"] = self.stage.value
        if self.node:
            result["node"] = self.node.value
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_json_string(self) -> str:
        """Convert to a JSON string"""
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def create_node_event(
        cls,
        node_type: NodeType,
        event_type: EventType,
        stage: WorkflowStage,
        message: str,
        duration: float | None = None,
        error: str | None = None,
        **kwargs,
    ) -> "StreamEvent":
        """Create a node event"""
        metadata = kwargs
        if duration is not None:
            metadata["duration"] = duration
        if error is not None:
            metadata["error"] = error

        return cls(type=event_type, content=message, stage=stage, node=node_type, metadata=metadata)

    @classmethod
    def create_workflow_event(
        cls, event_type: EventType, stage: WorkflowStage, message: str, workflow_id: str, **kwargs
    ) -> "StreamEvent":
        """Create a workflow event"""
        metadata = {"workflow_id": workflow_id, **kwargs}
        return cls(type=event_type, content=message, stage=stage, metadata=metadata)

    @classmethod
    def create_chunk(
        cls, content: str, index: int, total: int | None = None, is_final: bool = False, **kwargs
    ) -> "StreamEvent":
        """Create a streaming content chunk"""
        event_type = EventType.STREAM_COMPLETE if is_final else EventType.STREAM_CHUNK
        progress = (index + 1) / total if total else 0.0

        metadata = {"index": index, "total": total, "is_final": is_final, **kwargs}

        return cls(type=event_type, content=content, progress=progress, metadata=metadata)


@dataclass
class EventBuffer:
    """Event buffer"""

    events: list[StreamEvent] = field(default_factory=list)
    max_size: int = 1000

    def add(self, event: StreamEvent):
        """Add an event to the buffer"""
        self.events.append(event)
        # Limit buffer size
        if len(self.events) > self.max_size:
            self.events = self.events[-self.max_size :]

    def get_recent(self, n: int = 10) -> list[StreamEvent]:
        """Get the n most recent events"""
        return self.events[-n:] if len(self.events) >= n else self.events

    def clear(self):
        """Clear the buffer"""
        self.events.clear()

    def filter_by_type(self, event_type: EventType) -> list[StreamEvent]:
        """Filter events by type"""
        return [e for e in self.events if e.type == event_type]

    def filter_by_node(self, node_type: NodeType) -> list[StreamEvent]:
        """Filter events by node"""
        return [e for e in self.events if e.node == node_type]
