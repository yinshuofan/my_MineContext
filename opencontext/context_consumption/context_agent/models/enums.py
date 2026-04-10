#!/usr/bin/env python

"""
Enum Definition Module
Defines all enumeration types to avoid magic strings
"""

from enum import StrEnum


class NodeType(StrEnum):
    """Node type enumeration"""

    INTENT = "intent"
    CONTEXT = "context"
    EXECUTE = "execute"
    REFLECT = "reflect"


class WorkflowStage(StrEnum):
    """Workflow stage enumeration"""

    INIT = "init"
    INTENT_ANALYSIS = "intent_analysis"
    CONTEXT_GATHERING = "context_gathering"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    COMPLETED = "completed"
    FAILED = "failed"
    NEXT = "next"


class DataSource(StrEnum):
    """Data source type enumeration"""

    DOCUMENT = "document"  # Document library
    WEB_SEARCH = "web_search"  # Web search
    AGENT_MEMORY = "agent_memory"  # Agent memory
    CONTEXT_DB = "context_db"  # Context database
    CHAT_HISTORY = "chat_history"  # Chat history
    PROCESSED = "processed"  # Processed context
    UNKNOWN = "unknown"  # Unknown source


class TaskStatus(StrEnum):
    """Task status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INSUFFICIENT_INFO = "insufficient_info"


class ActionType(StrEnum):
    """Action type enumeration"""

    ANSWER = "answer"
    EDIT = "edit"
    CREATE_DOC = "create_doc"
    GENERATE = "generate"


class EventType(StrEnum):
    """Event type enumeration"""

    THINKING = "thinking"
    RUNNING = "running"
    DONE = "done"
    FAIL = "fail"
    COMPLETED = "completed"

    STREAM_CHUNK = "stream_chunk"
    STREAM_COMPLETE = "stream_complete"


class QueryType(StrEnum):
    """Query type enumeration - five categories"""

    SIMPLE_CHAT = "simple_chat"  # Simple chat (daily greetings, small talk, etc.)
    # Document editing and rewriting (preserving existing facts/not introducing new info)
    DOCUMENT_EDIT = "document_edit"
    # Q&A (covering summarization, analysis, and dialogue based on documents and complex context)
    QA_ANALYSIS = "qa_analysis"
    CONTENT_GENERATION = (
        "content_generation"  # Document content generation/expansion (allowing new information)
    )
    CLARIFICATION_NEEDED = (
        "clarification_needed"  # Query is too vague or ambiguous, needs user clarification
    )


class ContextSufficiency(StrEnum):
    """Context sufficiency enumeration"""

    SUFFICIENT = "sufficient"  # Sufficient
    PARTIAL = "partial"  # Partially sufficient
    INSUFFICIENT = "insufficient"  # Insufficient
    UNKNOWN = "unknown"  # Unknown


class ReflectionType(StrEnum):
    """Reflection type enumeration"""

    SUCCESS = "success"  # Successfully completed
    PARTIAL_SUCCESS = "partial_success"  # Partially successful
    FAILURE = "failure"  # Failed
    NEED_MORE_INFO = "need_more_info"  # Needs more information
    NEED_RETRY = "need_retry"  # Needs retry
