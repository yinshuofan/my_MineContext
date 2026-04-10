# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Define core data models used in OpenContext
"""

import base64
import datetime
import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from opencontext.models.enums import ContentFormat, ContextSource, ContextType
from opencontext.utils.logging_utils import get_logger
from opencontext.utils.time_utils import now as _tz_now

logger = get_logger(__name__)


class Chunk(BaseModel):
    """
    Represents a chunk split from a document or text
    """

    text: str | None = None
    image: bytes | None = None
    chunk_index: int = 0
    keywords: list[str] = Field(default_factory=list)  # keywords
    entities: list[str] = Field(default_factory=list)  # entities


class RawContextProperties(BaseModel):
    content_format: ContentFormat
    source: ContextSource
    create_time: datetime.datetime
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_path: str | None = None  # file path if ContentFormat is VIDEO or IMAGE; None if TEXT
    content_type: str | None = None  # content type, e.g. "text", "image", "video"
    content_text: str | None = None  # text content if ContentFormat is TEXT; None otherwise
    filter_path: str | None = None  # filter path
    additional_info: dict[str, Any] | None = None  # additional information
    enable_merge: bool = True
    # Multi-user support fields
    user_id: str | None = None  # User identifier
    device_id: str | None = None  # Device identifier
    agent_id: str | None = None  # Agent identifier

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawContextProperties":
        """Create model from dictionary"""
        return cls.model_validate(data)


class ExtractedData(BaseModel):
    """
    Represents information extracted from context data
    """

    title: str | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)  # keywords
    entities: list[str] = Field(default_factory=list)  # entities
    context_type: ContextType  # context type
    confidence: int = Field(default=0)  # confidence (0-10)
    importance: int = Field(default=0)  # importance (0-10)
    agent_commentary: str | None = None  # agent's subjective commentary on this event

    @field_validator("confidence", "importance", mode="before")
    @classmethod
    def clamp_score(cls, v):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        return max(0, min(10, v))

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedData":
        """Create model from dictionary"""
        return cls.model_validate(data)


class ContextProperties(BaseModel):
    """
    Represents context data attributes
    """

    raw_properties: list[RawContextProperties] = Field(
        default_factory=list
    )  # raw context properties
    create_time: datetime.datetime  # creation time
    event_time_start: datetime.datetime  # event time range start
    event_time_end: datetime.datetime  # event time range end (equals start for point events)
    is_processed: bool = False  # whether processed
    has_compression: bool = False  # whether compressed
    update_time: datetime.datetime  # update time
    call_count: int = 0  # call count, updated during online service calls
    merge_count: int = 0  # merge count
    duration_count: int = 1  # context duration count
    enable_merge: bool = False
    last_call_time: datetime.datetime | None = (
        None  # last call time, updated during online service calls
    )
    # position: Optional[Dict[str, Any]] = None # context position in original data

    # Document tracking fields
    file_path: str | None = None  # file path (empty for documents)
    raw_type: str | None = None  # raw type (e.g. 'vaults')
    raw_id: str | None = None  # raw ID (ID in vaults table)

    # Multi-user support fields
    user_id: str | None = None  # User identifier
    device_id: str | None = None  # Device identifier
    agent_id: str | None = None  # Agent identifier

    # Hierarchy indexing fields (event type only, for time-based hierarchical summaries)
    hierarchy_level: int = 0  # 0=original, 1=daily summary, 2=weekly summary, 3=monthly summary
    refs: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _default_event_time_end(cls, data):
        if isinstance(data, dict) and "event_time_end" not in data and "event_time_start" in data:
            data["event_time_end"] = data["event_time_start"]
        return data


class VideoInput(BaseModel):
    """视频输入"""

    url: str  # HTTP URL, TOS path, or data:video/...;base64,...
    fps: float = 1.0  # 0.2-5.0, frame extraction rate


class Vectorize(BaseModel):
    """
    Vectorization configuration — supports text, image, video, and multimodal content.
    Uses Ark API content parts format as unified internal representation.
    """

    input: list[dict[str, Any]] = Field(default_factory=list)
    vector: list[float] | None = None
    content_format: ContentFormat = ContentFormat.TEXT

    def get_modality_string(self) -> str:
        """Return a human-readable modality descriptor based on input content types.

        Examples: "text", "text and image", "text and image and video", "image", etc.
        Falls back to "text" when no content is present.
        """
        types = {item.get("type") for item in self.input}
        parts: list[str] = []
        if "text" in types:
            parts.append("text")
        if "image_url" in types:
            parts.append("image")
        if "video_url" in types:
            parts.append("video")
        return " and ".join(parts) if parts else "text"

    def build_ark_input(self) -> list[dict[str, Any]]:
        """Build the input list for the Ark multimodal embedding API.

        Local file paths in image_url/video_url items are converted to
        base64 data URIs since remote APIs cannot access local files.
        """
        result: list[dict[str, Any]] = []
        for item in self.input:
            item_type = item.get("type")
            if item_type == "image_url":
                url = item["image_url"]["url"]
                if _is_local_path(url):
                    url = _file_to_data_uri(url)
                result.append({"type": "image_url", "image_url": {"url": url}})
            elif item_type == "video_url":
                vid_info = item["video_url"]
                url = vid_info["url"]
                if _is_local_path(url):
                    url = _file_to_data_uri(url)
                new_vid = {k: v for k, v in vid_info.items()}
                new_vid["url"] = url
                result.append({"type": "video_url", "video_url": new_vid})
            else:
                result.append(item)
        return result

    def get_text(self) -> str | None:
        """Extract text content from input items. Returns concatenated text or None."""
        texts = [item["text"] for item in self.input if item.get("type") == "text"]
        return "\n".join(texts) if texts else None


class ProcessedContext(BaseModel):
    """
    Represents processed context data
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    properties: ContextProperties
    extracted_data: ExtractedData
    vectorize: Vectorize
    metadata: dict[str, Any] | None = Field(
        default_factory=dict
    )  # metadata for storing structured entity information

    def get_llm_context_string(self) -> str:
        """Get context information string for LLM input"""
        parts = []
        ed = self.extracted_data
        parts.append(f"id: {self.id}")
        if ed.title:
            parts.append(f"title: {ed.title}")
        if ed.summary:
            parts.append(f"summary: {ed.summary}")
        if ed.keywords:
            parts.append(f"keywords: {', '.join(ed.keywords)}")
        if ed.entities:
            parts.append(f"entities: {', '.join(ed.entities)}")
        if ed.context_type:
            parts.append(f"context type: {ed.context_type.value}")
        if self.metadata:
            parts.append(f"metadata: {json.dumps(self.metadata, ensure_ascii=False)}")

        # Raw properties
        # raw_contexts_props = self.properties.raw_properties
        # for i, raw_prop in enumerate(raw_contexts_props):
        #     source = raw_prop.source.value if raw_prop.source else 'N/A'
        #     parts.append(f"raw context source {i+1}: {source}")
        create_time = self.properties.create_time
        parts.append(f"create time: {create_time.isoformat()}")
        event_time_start = self.properties.event_time_start
        parts.append(f"event time: {event_time_start.isoformat()}")
        if self.properties.event_time_end != event_time_start:
            parts.append(f"event time end: {self.properties.event_time_end.isoformat()}")
        duration_count = self.properties.duration_count
        parts.append(f"duration count: {duration_count}")

        # Hierarchy info (for event summaries)
        if self.properties.hierarchy_level > 0:
            parts.append(f"hierarchy level: {self.properties.hierarchy_level}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary"""
        return self.model_dump(exclude_none=True)

    def dump_json(self) -> str:
        """Convert model to JSON string"""
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessedContext":
        """Create model from dictionary"""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ProcessedContext":
        """Create model from JSON string"""
        return cls.model_validate_json(json_str)


class RawContextModel(BaseModel):
    """
    Raw context data model for API responses
    """

    object_id: str
    content_format: str
    source: str
    create_time: str
    content_path: str | None = None
    content_text: str | None = None
    additional_info: dict[str, Any] | None = None

    @classmethod
    def from_raw_context_properties(
        cls, rcp: "RawContextProperties", project_root: Path
    ) -> "RawContextModel":
        """Create API model from RawContextProperties object"""
        content_path = None
        if rcp.content_path:
            try:
                # Convert to relative path from project root
                relative_path = Path(rcp.content_path).relative_to(project_root)
                content_path = str(relative_path)
            except ValueError:
                # If path is not under project root, use absolute path
                content_path = rcp.content_path

        return cls(
            object_id=rcp.object_id,
            content_format=rcp.content_format.value,
            source=rcp.source.value,
            create_time=rcp.create_time.isoformat(),
            content_path=content_path,
            content_text=rcp.content_text,
            additional_info=rcp.additional_info,
        )


class ProcessedContextModel(BaseModel):
    """
    Processed context data model for API responses
    """

    id: str
    title: str | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    context_type: str
    confidence: int
    importance: int
    agent_commentary: str | None = None
    is_processed: bool
    call_count: int
    enable_merge: bool = False
    merge_count: int  # merge count
    last_call_time: str | None = None
    create_time: str
    update_time: str
    event_time_start: str
    event_time_end: str
    embedding: list[float] | None = None
    raw_contexts: list["RawContextModel"] = Field(default_factory=list)
    duration_count: int  # context duration count
    metadata: dict[str, Any] | None = None  # metadata information
    # Multi-user support fields
    user_id: str | None = None  # User identifier
    device_id: str | None = None  # Device identifier
    agent_id: str | None = None  # Agent identifier
    # Hierarchy fields
    hierarchy_level: int = 0
    refs: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def from_processed_context(
        cls, pc: "ProcessedContext", project_root: Path
    ) -> "ProcessedContextModel":
        """Create API model from ProcessedContext object"""

        # Generate title
        title = pc.extracted_data.title

        # Create raw context model list
        raw_contexts = [
            RawContextModel.from_raw_context_properties(rcp, project_root)
            for rcp in pc.properties.raw_properties
        ]
        # logger.info(f"raw_contexts duration_count: {pc.properties.duration_count}")

        return cls(
            id=pc.id,
            title=title,
            summary=pc.extracted_data.summary,
            keywords=pc.extracted_data.keywords,
            entities=pc.extracted_data.entities,
            context_type=pc.extracted_data.context_type.value,
            confidence=pc.extracted_data.confidence,
            importance=pc.extracted_data.importance,
            agent_commentary=pc.extracted_data.agent_commentary,
            is_processed=pc.properties.is_processed,
            call_count=pc.properties.call_count,
            enable_merge=pc.properties.enable_merge,  # set enable merge
            merge_count=pc.properties.merge_count,  # set merge count
            duration_count=pc.properties.duration_count,  # set duration count
            last_call_time=(
                pc.properties.last_call_time.strftime("%Y-%m-%d %H:%M:%S")
                if pc.properties.last_call_time
                else None
            ),
            create_time=pc.properties.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            update_time=pc.properties.update_time.strftime("%Y-%m-%d %H:%M:%S"),
            event_time_start=pc.properties.event_time_start.strftime("%Y-%m-%d %H:%M:%S"),
            event_time_end=pc.properties.event_time_end.strftime("%Y-%m-%d %H:%M:%S"),
            embedding=pc.vectorize.vector,
            raw_contexts=raw_contexts,
            metadata=pc.metadata,  # add metadata
            # Multi-user support fields
            user_id=pc.properties.user_id,
            device_id=pc.properties.device_id,
            agent_id=pc.properties.agent_id,
            # Hierarchy fields
            hierarchy_level=pc.properties.hierarchy_level,
            refs=pc.properties.refs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessedContextModel":
        """Create model from dictionary"""
        return cls.model_validate(data)


class ProfileData(BaseModel):
    """User profile — stored in relational DB (composite key: user_id + device_id + agent_id)"""

    user_id: str  # Composite primary key part 1
    device_id: str = "default"  # Composite primary key part 2
    agent_id: str = (
        "default"  # Composite primary key part 3 (different agents can have different profiles)
    )
    factual_profile: str  # Factual profile text (LLM-merged result)
    behavioral_profile: str | None = None  # Behavioral profile text
    entities: list[str] = Field(default_factory=list)
    importance: int = 0
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    created_at: datetime.datetime = Field(default_factory=_tz_now)
    updated_at: datetime.datetime = Field(default_factory=_tz_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileData":
        """Create model from dictionary"""
        return cls.model_validate(data)


class KnowledgeContextMetadata(BaseModel):
    """Knowledge context additional information"""

    knowledge_source: str = ""
    knowledge_file_path: str = ""
    knowledge_title: str = ""
    knowledge_raw_id: str = ""


# ============================================================================
# Local file path → base64 data URI conversion helpers
# Used when LocalBackend stores files locally but remote APIs need base64.
# ============================================================================


def _is_local_path(url: str) -> bool:
    """Check if a URL is a local file path (not HTTP/HTTPS/data URI)."""
    if not url:
        return False
    return not url.startswith(("http://", "https://", "data:"))


def _file_to_data_uri(file_path: str) -> str:
    """Read a local file and convert to a base64 data URI.

    Returns the original path if the file cannot be read.
    """
    try:
        abs_path = os.path.abspath(file_path)
        mime_type, _ = mimetypes.guess_type(abs_path)
        if not mime_type:
            ext = os.path.splitext(abs_path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".mp4": "video/mp4",
                ".avi": "video/avi",
                ".mov": "video/quicktime",
            }.get(ext, "application/octet-stream")
        with open(abs_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to convert file to data URI: {file_path}: {e}")
        return file_path
