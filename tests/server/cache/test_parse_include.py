"""Tests for memory_cache route _parse_include helper."""

import pytest

from opencontext.server.routes.memory_cache import (
    DEFAULT_SECTIONS,
    VALID_SECTIONS,
    _parse_include,
)


@pytest.mark.unit
def test_agent_prompt_is_a_valid_section():
    assert "agent_prompt" in VALID_SECTIONS


@pytest.mark.unit
def test_default_sections_include_agent_prompt():
    assert {"profile", "agent_prompt", "events", "accessed"} == DEFAULT_SECTIONS


@pytest.mark.unit
def test_parse_agent_prompt_only():
    assert _parse_include("agent_prompt") == {"agent_prompt"}


@pytest.mark.unit
def test_parse_profile_and_agent_prompt():
    assert _parse_include("profile,agent_prompt") == {"profile", "agent_prompt"}


@pytest.mark.unit
def test_parse_all_expands_to_every_valid_section():
    assert _parse_include("all") == {"profile", "agent_prompt", "events", "accessed"}


@pytest.mark.unit
def test_parse_none_returns_default():
    assert _parse_include(None) == {"profile", "agent_prompt", "events", "accessed"}


@pytest.mark.unit
def test_unknown_values_fall_back_to_default():
    assert _parse_include("bogus") == {"profile", "agent_prompt", "events", "accessed"}
