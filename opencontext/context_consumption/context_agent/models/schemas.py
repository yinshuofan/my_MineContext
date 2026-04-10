#!/usr/bin/env python

"""
Core Data Model Definition
Use dataclass to define all data structures to ensure type safety
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from opencontext.utils.time_utils import now as tz_now

from .enums import ActionType, ContextSufficiency, DataSource, QueryType, ReflectionType, TaskStatus


@dataclass
class ChatMessage:
    """Chat message"""

    role: str  # user, assistant, system
    content: str
    # timestamp: datetime = field(default_factory=tz_now)
    # metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebSearchResult:
    """Web search result"""

    title: str
    url: str
    snippet: str
    relevance_score: float = 1.0
    source: str = "web"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """User query encapsulation"""

    text: str
    query_type: QueryType | None = None
    user_id: str | None = None
    session_id: str | None = None
    selected_content: str | None = None
    document_id: str | None = None


@dataclass
class Intent:
    """Intent analysis result"""

    original_query: str
    query_type: QueryType
    enhanced_query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextItem:
    """A single context item"""

    source: DataSource
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str | None = None
    relevance_score: float = 1.0
    timestamp: datetime = field(default_factory=tz_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_relevant: bool = True
    relevance_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.value,
            "content": self.content,
            "title": self.title,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_relevant": self.is_relevant,
            "relevance_reason": self.relevance_reason,
        }


@dataclass
class DocumentInfo:
    """Document context"""

    id: str | None = None
    title: str | None = None
    content: str | None = None
    summary: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "tags": self.tags,
        }


@dataclass
class ContextCollection:
    """Context collection"""

    items: list[ContextItem] = field(default_factory=list)
    sufficiency: ContextSufficiency = ContextSufficiency.UNKNOWN
    missing_sources: set[DataSource] = field(default_factory=set)
    collection_metadata: dict[str, Any] = field(default_factory=dict)
    current_document: DocumentInfo | None = None
    chat_history: list[ChatMessage] = field(default_factory=list)
    selected_content: str | None = None

    def add_item(self, item: ContextItem):
        """Add a context item"""
        self.items.append(item)

    def get_by_source(self, source: DataSource) -> list[ContextItem]:
        """Get context by data source"""
        return [item for item in self.items if item.source == source]

    def is_sufficient(self) -> bool:
        """Check if the context is sufficient"""
        return self.sufficiency == ContextSufficiency.SUFFICIENT

    def prepare_context(self) -> dict[str, Any]:
        import json
        from dataclasses import asdict

        """Prepare execution context, optimize loading strategy based on query_type"""
        context = {}
        # Convert ChatMessage objects to dictionaries before serializing
        chat_history_dicts = [asdict(msg) for msg in self.chat_history]
        context["chat_history"] = json.dumps(chat_history_dicts, ensure_ascii=False)
        if self.current_document:
            context["current_document"] = json.dumps(
                self.current_document.to_dict(), ensure_ascii=False
            )
        if self.selected_content:
            context["selected_content"] = self.selected_content or ""
        if self.items:
            context["collected_contexts"] = json.dumps(
                [item.to_dict() for item in self.items], ensure_ascii=False
            )
        return context

    def get_summary(self) -> str:
        """Get a summary of the context"""
        source_counts = {}
        for item in self.items:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1

        summary_parts = [f"{source.value}: {count}" for source, count in source_counts.items()]
        return f"Collected {len(self.items)} context items ({', '.join(summary_parts)}), sufficiency: {self.sufficiency.value}"

    def get_chat_history(self) -> list[dict[str, str]]:
        """Get chat history as a list of dictionaries"""
        return [{"role": msg.role, "content": msg.content} for msg in self.chat_history]


@dataclass
class ExecutionStep:
    """Execution step"""

    action: ActionType
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self):
        """Convert to a dictionary"""
        return {
            "action": self.action.value,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class ExecutionPlan:
    """Execution plan"""

    steps: list[ExecutionStep] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary"""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "current_step": self.current_step,
            "total_steps": self.total_steps,
        }

    def add_step(self, step: ExecutionStep):
        """Add an execution step"""
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def get_current_step(self) -> ExecutionStep | None:
        """Get the current step"""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance(self):
        """Advance to the next step"""
        if self.current_step < self.total_steps - 1:
            self.current_step += 1


@dataclass
class ExecutionResult:
    """Execution result"""

    success: bool
    plan: ExecutionPlan
    outputs: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionResult:
    """Reflection result"""

    reflection_type: ReflectionType
    success_rate: float
    summary: str
    issues: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    should_retry: bool = False
    retry_strategy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
