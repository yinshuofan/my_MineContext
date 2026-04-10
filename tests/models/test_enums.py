"""Unit tests for opencontext.models.enums"""

import pytest

from opencontext.models.enums import (
    CONTEXT_STORAGE_BACKENDS,
    CONTEXT_UPDATE_STRATEGIES,
    ContextType,
    UpdateStrategy,
    get_context_type_for_analysis,
    validate_context_type,
)


@pytest.mark.unit
class TestContextType:
    """Tests for ContextType enum and related utilities."""

    def test_all_context_types_have_update_strategy(self):
        for ct in ContextType:
            assert ct in CONTEXT_UPDATE_STRATEGIES, f"{ct} missing from CONTEXT_UPDATE_STRATEGIES"

    def test_all_context_types_have_storage_backend(self):
        for ct in ContextType:
            assert ct in CONTEXT_STORAGE_BACKENDS, f"{ct} missing from CONTEXT_STORAGE_BACKENDS"

    def test_profile_types_use_document_db(self):
        profile_types = [
            ContextType.PROFILE,
            ContextType.AGENT_PROFILE,
            ContextType.AGENT_BASE_PROFILE,
        ]
        for ct in profile_types:
            assert CONTEXT_STORAGE_BACKENDS[ct] == "document_db", f"{ct} should use document_db"

    def test_event_types_use_vector_db(self):
        assert CONTEXT_STORAGE_BACKENDS[ContextType.EVENT] == "vector_db"
        assert CONTEXT_STORAGE_BACKENDS[ContextType.KNOWLEDGE] == "vector_db"
        assert CONTEXT_STORAGE_BACKENDS[ContextType.DOCUMENT] == "vector_db"

    def test_profile_uses_overwrite_strategy(self):
        assert CONTEXT_UPDATE_STRATEGIES[ContextType.PROFILE] == UpdateStrategy.OVERWRITE

    def test_event_uses_append_strategy(self):
        assert CONTEXT_UPDATE_STRATEGIES[ContextType.EVENT] == UpdateStrategy.APPEND

    def test_knowledge_uses_append_merge_strategy(self):
        assert CONTEXT_UPDATE_STRATEGIES[ContextType.KNOWLEDGE] == UpdateStrategy.APPEND_MERGE


@pytest.mark.unit
class TestValidateContextType:
    """Tests for validate_context_type()."""

    def test_valid_types(self):
        assert validate_context_type("profile") is True
        assert validate_context_type("event") is True
        assert validate_context_type("knowledge") is True
        assert validate_context_type("document") is True

    def test_invalid_types(self):
        assert validate_context_type("invalid") is False
        assert validate_context_type("") is False
        assert validate_context_type("PROFILE") is False  # case sensitive


@pytest.mark.unit
class TestGetContextTypeForAnalysis:
    """Tests for get_context_type_for_analysis()."""

    def test_known_type_returned(self):
        assert get_context_type_for_analysis("profile") == ContextType.PROFILE
        assert get_context_type_for_analysis("event") == ContextType.EVENT

    def test_case_insensitive(self):
        assert get_context_type_for_analysis("PROFILE") == ContextType.PROFILE
        assert get_context_type_for_analysis("  Event  ") == ContextType.EVENT

    def test_unknown_falls_back_to_knowledge(self):
        assert get_context_type_for_analysis("nonexistent") == ContextType.KNOWLEDGE

    def test_summary_types_fall_back_to_knowledge(self):
        assert get_context_type_for_analysis("daily_summary") == ContextType.KNOWLEDGE
        assert get_context_type_for_analysis("weekly_summary") == ContextType.KNOWLEDGE
        assert get_context_type_for_analysis("monthly_summary") == ContextType.KNOWLEDGE
