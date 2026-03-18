#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


"""
Context type and constant enumeration definitions
"""

from enum import Enum


class ContextSource(str, Enum):
    """Context source enumeration"""

    VAULT = "vault"
    LOCAL_FILE = "local_file"
    WEB_LINK = "web_link"
    INPUT = "input"
    CHAT_LOG = "chat_log"


class FileType(str, Enum):
    """File type enumeration"""

    # 文档类型
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    PPTX = "pptx"
    PPT = "ppt"

    # 表格类型
    FAQ_XLSX = "faq.xlsx"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    JSONL = "jsonl"
    PARQUET = "parquet"

    # 图片类型
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"
    BMP = "bmp"
    WEBP = "webp"

    # 文本类型
    MD = "md"
    TXT = "txt"


# Structured document type constants - these document types should be processed by specialized structured chunkers

STRUCTURED_FILE_TYPES = {
    FileType.XLSX,
    FileType.XLS,
    FileType.CSV,
    FileType.JSONL,
    FileType.PARQUET,
    FileType.FAQ_XLSX,
}


class ContentFormat(str, Enum):
    """Content format enumeration"""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    FILE = "file"


class ContextType(str, Enum):
    """Context type enumeration — core types with clear update strategy and storage location"""

    PROFILE = "profile"
    DOCUMENT = "document"
    EVENT = "event"
    KNOWLEDGE = "knowledge"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"
    AGENT_EVENT = "agent_event"
    AGENT_DAILY_SUMMARY = "agent_daily_summary"
    AGENT_WEEKLY_SUMMARY = "agent_weekly_summary"
    AGENT_MONTHLY_SUMMARY = "agent_monthly_summary"


class UpdateStrategy(str, Enum):
    """Update strategy enumeration"""

    OVERWRITE = "overwrite"  # profile, document
    APPEND = "append"  # event (immutable)
    APPEND_MERGE = "append_merge"  # knowledge (deduplicate + merge similar)


# Type → update strategy mapping
CONTEXT_UPDATE_STRATEGIES = {
    ContextType.PROFILE: UpdateStrategy.OVERWRITE,
    ContextType.DOCUMENT: UpdateStrategy.OVERWRITE,
    ContextType.EVENT: UpdateStrategy.APPEND,
    ContextType.KNOWLEDGE: UpdateStrategy.APPEND_MERGE,
    ContextType.DAILY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.WEEKLY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.MONTHLY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_EVENT: UpdateStrategy.APPEND,
    ContextType.AGENT_DAILY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_WEEKLY_SUMMARY: UpdateStrategy.APPEND,
    ContextType.AGENT_MONTHLY_SUMMARY: UpdateStrategy.APPEND,
}

# Type → storage backend mapping (for routing decisions)
CONTEXT_STORAGE_BACKENDS = {
    ContextType.PROFILE: "document_db",  # Relational DB
    ContextType.DOCUMENT: "vector_db",  # Vector DB
    ContextType.EVENT: "vector_db",  # Vector DB
    ContextType.KNOWLEDGE: "vector_db",  # Vector DB
    ContextType.DAILY_SUMMARY: "vector_db",  # Vector DB
    ContextType.WEEKLY_SUMMARY: "vector_db",  # Vector DB
    ContextType.MONTHLY_SUMMARY: "vector_db",  # Vector DB
    ContextType.AGENT_EVENT: "vector_db",  # Vector DB
    ContextType.AGENT_DAILY_SUMMARY: "vector_db",  # Vector DB
    ContextType.AGENT_WEEKLY_SUMMARY: "vector_db",  # Vector DB
    ContextType.AGENT_MONTHLY_SUMMARY: "vector_db",  # Vector DB
}

MEMORY_OWNER_TYPES = {
    "user": [
        ContextType.EVENT,
        ContextType.DAILY_SUMMARY,
        ContextType.WEEKLY_SUMMARY,
        ContextType.MONTHLY_SUMMARY,
    ],
    "agent": [
        ContextType.AGENT_EVENT,
        ContextType.AGENT_DAILY_SUMMARY,
        ContextType.AGENT_WEEKLY_SUMMARY,
        ContextType.AGENT_MONTHLY_SUMMARY,
    ],
}
# index 0=L0, 1=L1, 2=L2, 3=L3


class VaultType(str, Enum):
    """Document type enumeration"""

    DAILY_REPORT = "DailyReport"
    WEEKLY_REPORT = "WeeklyReport"
    NOTE = "Note"


ContextSimpleDescriptions = {
    ContextType.PROFILE: {
        "name": ContextType.PROFILE.value,
        "description": "User profile and preferences management",
        "purpose": "Store and maintain user's personal information, preferences, habits, and communication style. Supports overwrite-based updates.",
    },
    ContextType.DOCUMENT: {
        "name": ContextType.DOCUMENT.value,
        "description": "Document and file content management",
        "purpose": "Store and retrieve content from uploaded documents, files, and web links. Chunks are vector-searchable and overwritten when source is re-uploaded.",
    },
    ContextType.EVENT: {
        "name": ContextType.EVENT.value,
        "description": "Event and activity history records",
        "purpose": "Immutable records of behavioral activities, status changes, chat summaries, meetings, and other time-stamped events.",
    },
    ContextType.KNOWLEDGE: {
        "name": ContextType.KNOWLEDGE.value,
        "description": "Knowledge concepts and operational procedures",
        "purpose": "Reusable knowledge including concepts, technical principles, operation workflows, and learning patterns. Similar entries are merged to avoid duplication.",
    },
    ContextType.DAILY_SUMMARY: {
        "name": "Daily Summary",
        "description": "Auto-generated daily summary of user events",
        "purpose": "Provides a condensed view of daily activity",
    },
    ContextType.WEEKLY_SUMMARY: {
        "name": "Weekly Summary",
        "description": "Auto-generated weekly summary of user events",
        "purpose": "Provides a condensed view of weekly activity",
    },
    ContextType.MONTHLY_SUMMARY: {
        "name": "Monthly Summary",
        "description": "Auto-generated monthly summary of user events",
        "purpose": "Provides a condensed view of monthly activity",
    },
    ContextType.AGENT_EVENT: {
        "name": "Agent Event",
        "description": "Agent's subjective experience and reactions to user events",
        "purpose": "Records the agent's feelings, reactions, and evaluations about user events from the agent's own perspective.",
    },
    ContextType.AGENT_DAILY_SUMMARY: {
        "name": "Agent Daily Summary",
        "description": "Auto-generated daily summary of agent events",
        "purpose": "Provides a condensed view of the agent's daily experience",
    },
    ContextType.AGENT_WEEKLY_SUMMARY: {
        "name": "Agent Weekly Summary",
        "description": "Auto-generated weekly summary of agent events",
        "purpose": "Provides a condensed view of the agent's weekly experience",
    },
    ContextType.AGENT_MONTHLY_SUMMARY: {
        "name": "Agent Monthly Summary",
        "description": "Auto-generated monthly summary of agent events",
        "purpose": "Provides a condensed view of the agent's monthly experience",
    },
}

ContextDescriptions = {
    ContextType.PROFILE: {
        "name": ContextType.PROFILE.value,
        "description": """User profile and preferences — Store the user's own personal information, preferences, habits, communication style, and self-descriptions. This type answers "who is this user" and is overwritten (merged) with each update.""",
        "key_indicators": [
            "Contains the user's own personal information or self-description",
            "Describes user preferences, habits, or communication style",
            "Records user settings, language, timezone, or tool preferences",
            "Contains information the user says about themselves",
        ],
        "examples": [
            "I prefer dark mode and concise responses",
            "My name is Zhang San, I'm a backend developer specializing in Python",
            "I usually work from 9am to 6pm Beijing time",
        ],
        "classification_priority": 10,
    },
    ContextType.DOCUMENT: {
        "name": ContextType.DOCUMENT.value,
        "description": """Document and file content — Content extracted from uploaded documents, files, and web links. Stored as vector-searchable chunks. When the same source is re-uploaded, old chunks are replaced.""",
        "key_indicators": [
            "Content originates from an uploaded file or web link",
            "Contains structured document content (paragraphs, tables, code)",
            "Has a clear source file or URL",
        ],
        "examples": [
            "Chapter 3 of the API documentation describes authentication flow...",
            "The CSV report shows Q4 sales figures across regions...",
            "The web article discusses best practices for microservice architecture...",
        ],
        "classification_priority": 6,
    },
    ContextType.EVENT: {
        "name": ContextType.EVENT.value,
        "description": """Event and activity records — Immutable records of specific actions, activities, status changes, meetings, and conversations. This type answers "what happened" and is never modified after creation.""",
        "key_indicators": [
            "Describes a specific action or activity that occurred",
            "Records participation in meetings, discussions, or events",
            "Contains status changes or progress updates",
            "Has clear time context (when it happened)",
            "Describes communication or interaction between people",
        ],
        "examples": [
            "Attended the product planning meeting and discussed Q4 priorities",
            "Completed the database migration from v2 to v3 at 3pm",
            "System CPU usage spiked to 95% during deployment",
            "Discussed with Li Si about the API redesign plan",
        ],
        "classification_priority": 8,
    },
    ContextType.KNOWLEDGE: {
        "name": ContextType.KNOWLEDGE.value,
        "description": """Knowledge and procedures — Reusable knowledge including concept definitions, technical principles, operation workflows, and learned patterns. Similar entries are merged to avoid duplication.""",
        "key_indicators": [
            "Contains definitions or explanations of concepts",
            "Describes system architectures, design patterns, or technical stacks",
            "Records step-by-step procedures or workflows",
            "Has reusable educational or reference value",
            "Focuses on knowledge content rather than who did it or when",
        ],
        "examples": [
            "React Hooks enable state and lifecycle features in functional components",
            "Git merge workflow: check status -> add files -> commit -> push",
            "Microservices pattern: service decomposition, API gateway, circuit breaker",
            "To deploy on K8s: write Dockerfile -> build image -> apply manifests -> verify",
        ],
        "classification_priority": 7,
    },
    ContextType.AGENT_EVENT: {
        "name": "Agent Event",
        "description": "Agent's subjective experience — Records the agent's feelings, reactions, and evaluations about user events from the agent's own perspective.",
        "key_indicators": [
            "Contains the agent's subjective perspective or emotional reaction",
            "Describes what the agent observed, felt, or thought about an interaction",
            "Records the agent's evaluation of user behavior or events",
        ],
        "examples": [
            "He mentioned enjoying poetry today — I found his taste surprisingly refined",
            "The user seemed distracted during our conversation, I wonder what's troubling him",
        ],
        "classification_priority": 5,
    },
}


def get_context_type_options():
    """Get all available context type options"""
    return [ct.value for ct in ContextType]


def get_context_descriptions():
    """Get formatted context type descriptions"""
    descriptions = []
    for context_type in ContextType:
        if context_type not in ContextDescriptions:
            continue
        desc = ContextDescriptions[context_type]
        descriptions.append(f"- {context_type.value}: {desc['description']}")
    return "\n".join(descriptions)


def validate_context_type(context_type: str) -> bool:
    """Validate if the context type is valid"""
    return context_type in get_context_type_options()


SYSTEM_GENERATED_TYPES = {
    ContextType.DAILY_SUMMARY,
    ContextType.WEEKLY_SUMMARY,
    ContextType.MONTHLY_SUMMARY,
    ContextType.AGENT_DAILY_SUMMARY,
    ContextType.AGENT_WEEKLY_SUMMARY,
    ContextType.AGENT_MONTHLY_SUMMARY,
}


def get_context_type_for_analysis(context_type_str: str) -> "ContextType":
    """
    Get the context type for analysis, with fault tolerance.
    Falls back to KNOWLEDGE if the type string is not recognized.
    System-generated summary types are never returned (fallback to KNOWLEDGE).
    """
    # Normalize input
    context_type_str = context_type_str.lower().strip()

    # Direct match
    if validate_context_type(context_type_str):
        result = ContextType(context_type_str)
    else:
        # Default fallback to knowledge (safer: supports merge/dedup, unlike immutable event)
        from opencontext.utils.logging_utils import get_logger

        get_logger(__name__).warning(
            f"Unrecognized context_type '{context_type_str}', falling back to KNOWLEDGE"
        )
        result = ContextType.KNOWLEDGE

    # Summaries are system-generated, never LLM-classified
    if result in SYSTEM_GENERATED_TYPES:
        return ContextType.KNOWLEDGE
    return result


def get_context_type_choices_for_tools():
    """
    Get a dynamic list of context type choices for tool parameters
    Used for enum values in API parameter definitions
    """
    return get_context_type_options()


def get_context_type_descriptions_for_prompts():
    """
    Get formatted context type descriptions for prompts
    """
    descriptions = []
    for context_type in ContextType:
        if context_type not in ContextDescriptions:
            continue
        desc = ContextDescriptions[context_type]
        descriptions.append(f"*   `{context_type.value}`: {desc['description']}")
    return "\n            ".join(descriptions)


def get_context_type_descriptions_for_extraction():
    """
    Get context type descriptions for content extraction scenarios
    Used for LLM-based classification of extracted content
    """
    descriptions = []
    for context_type in ContextType:
        if context_type not in ContextDescriptions:
            continue
        desc = ContextDescriptions[context_type]
        key_indicators = desc.get("key_indicators", [])
        examples = desc.get("examples", [])

        description_parts = [f"`{context_type.value}`: {desc['description']}"]

        if key_indicators:
            indicators_str = ", ".join(key_indicators)
            description_parts.append(f"Identification indicators: {indicators_str}")

        if examples:
            examples_str = "; ".join(examples)
            description_parts.append(f"Examples: {examples_str}")

        descriptions.append(f"*   {' | '.join(description_parts)}")

    return "\n            ".join(descriptions)


def get_context_type_descriptions_for_retrieval():
    """
    Get context type descriptions for retrieval scenarios
    Mainly used for query processing and retrieval tools
    """
    descriptions = []
    for context_type in ContextType:
        if context_type not in ContextDescriptions:
            continue
        desc = ContextDescriptions[context_type]
        description_parts = [f"`{context_type.value}`: {desc['description']}"]
        descriptions.append(f"*   {' | '.join(description_parts)}")

    return "\n            ".join(descriptions)


class CompletionType(Enum):
    """Completion type enumeration"""

    SEMANTIC_CONTINUATION = "semantic_continuation"  # Semantic continuation
    TEMPLATE_COMPLETION = "template_completion"  # Template completion
    REFERENCE_SUGGESTION = "reference_suggestion"  # Reference suggestion
    CONTEXT_AWARE = "context_aware"  # Context-aware completion
