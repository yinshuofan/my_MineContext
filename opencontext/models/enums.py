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
    FILE = "file"


class ContextType(str, Enum):
    """Context type enumeration — 5 types with clear update strategy and storage location"""

    PROFILE = "profile"
    ENTITY = "entity"
    DOCUMENT = "document"
    EVENT = "event"
    KNOWLEDGE = "knowledge"


class UpdateStrategy(str, Enum):
    """Update strategy enumeration"""

    OVERWRITE = "overwrite"  # profile, entity, document
    APPEND = "append"  # event (immutable)
    APPEND_MERGE = "append_merge"  # knowledge (deduplicate + merge similar)


# Type → update strategy mapping
CONTEXT_UPDATE_STRATEGIES = {
    ContextType.PROFILE: UpdateStrategy.OVERWRITE,
    ContextType.ENTITY: UpdateStrategy.OVERWRITE,
    ContextType.DOCUMENT: UpdateStrategy.OVERWRITE,
    ContextType.EVENT: UpdateStrategy.APPEND,
    ContextType.KNOWLEDGE: UpdateStrategy.APPEND_MERGE,
}

# Type → storage backend mapping (for routing decisions)
CONTEXT_STORAGE_BACKENDS = {
    ContextType.PROFILE: "document_db",  # Relational DB
    ContextType.ENTITY: "document_db",  # Relational DB
    ContextType.DOCUMENT: "vector_db",  # Vector DB
    ContextType.EVENT: "vector_db",  # Vector DB
    ContextType.KNOWLEDGE: "vector_db",  # Vector DB
}


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
    ContextType.ENTITY: {
        "name": ContextType.ENTITY.value,
        "description": "Entity profile information management",
        "purpose": "Record and manage profile information of various entities (people, projects, teams, organizations). Supports alias management and relationship tracking.",
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
    ContextType.ENTITY: {
        "name": ContextType.ENTITY.value,
        "description": """Entity profile information — Record and manage profile information of various entities (people other than current user, projects, teams, organizations). This type answers "who/what is this entity" and is overwritten per entity.""",
        "key_indicators": [
            "Contains information about entities other than the current user",
            "Describes basic attributes, roles, and characteristics of people/projects/teams",
            "Records entity aliases, abbreviations, and full names",
            "Involves relationships between entities",
        ],
        "examples": [
            "Li Si is a senior frontend engineer specializing in React and TypeScript",
            "Project Alpha is our company's new CRM system, deadline Q3 2026",
            "The AI team consists of 5 members led by Wang Wu",
        ],
        "classification_priority": 9,
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


def get_context_type_for_analysis(context_type_str: str) -> "ContextType":
    """
    Get the context type for analysis, with fault tolerance.
    Falls back to KNOWLEDGE if the type string is not recognized.
    """
    # Normalize input
    context_type_str = context_type_str.lower().strip()

    # Direct match
    if validate_context_type(context_type_str):
        return ContextType(context_type_str)

    # Default fallback to knowledge (safer: supports merge/dedup, unlike immutable event)
    from opencontext.utils.logging_utils import get_logger

    get_logger(__name__).warning(
        f"Unrecognized context_type '{context_type_str}', falling back to KNOWLEDGE"
    )
    return ContextType.KNOWLEDGE


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
