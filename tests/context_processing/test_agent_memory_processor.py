"""Unit tests for AgentMemoryProcessor integration with RecallAgent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opencontext.context_processing.processor.agent_memory_processor import (
    AgentMemoryProcessor,
)
from opencontext.models.enums import ContextSource, ContextType


def _make_raw_context():
    return SimpleNamespace(
        user_id="u1",
        device_id="d1",
        agent_id="bot1",
        source=ContextSource.CHAT_LOG,
        content_text="hello bot",
    )


def _make_event_ctx(idx: int):
    return SimpleNamespace(
        id=f"evt{idx}",
        extracted_data=SimpleNamespace(
            context_type=ContextType.EVENT,
            title=f"Event {idx}",
            summary=f"summary {idx}",
            agent_commentary=None,
        ),
    )


_COMMENTARY_PROMPT_GROUP = {
    "system": (
        "You are {agent_name}.\nPersona: {agent_persona}\nPast memories: {related_memories}\n"
    ),
    "user": ("Current time: {current_time}\nEvents: {event_list}\nConversation: {chat_history}"),
}


@pytest.mark.unit
async def test_processor_uses_recall_agent_output_in_commentary_prompt():
    """RecallAgent.recall result is injected into {related_memories} placeholder."""
    processor = AgentMemoryProcessor()
    raw = _make_raw_context()
    events = [_make_event_ctx(0), _make_event_ctx(1)]

    fake_storage = MagicMock()
    fake_storage.get_agent = AsyncMock(return_value={"name": "Bot"})
    fake_storage.get_profile = AsyncMock(return_value={"factual_profile": "helpful agent"})

    commentary_response = '{"commentaries": {"0": "felt nice", "1": "also interesting"}}'

    captured_messages: list = []

    async def fake_generate(messages, **kwargs):
        captured_messages.append(messages)
        return commentary_response

    with (
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_storage",
            return_value=fake_storage,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_prompt_group",
            return_value=_COMMENTARY_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.generate_with_messages",
            new=fake_generate,
        ),
        patch.object(
            processor._recall_agent,
            "recall",
            new=AsyncMock(return_value="[2026-04-10] some past event\nsum"),
        ),
    ):
        result = await processor._process_async(raw, events)

    assert len(captured_messages) == 1
    system_content = captured_messages[0][0]["content"]
    assert "[2026-04-10] some past event" in system_content
    assert events[0].extracted_data.agent_commentary == "felt nice"
    assert events[1].extracted_data.agent_commentary == "also interesting"
    assert len(result) == 2


@pytest.mark.unit
async def test_processor_falls_back_to_base_profile_when_user_profile_missing():
    """If user-specific profile missing, processor tries __base__ agent_base_profile."""
    processor = AgentMemoryProcessor()
    raw = _make_raw_context()
    events = [_make_event_ctx(0)]

    fake_storage = MagicMock()
    fake_storage.get_agent = AsyncMock(return_value={"name": "Bot"})
    fake_storage.get_profile = AsyncMock(side_effect=[None, {"factual_profile": "base persona"}])

    with (
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_storage",
            return_value=fake_storage,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_prompt_group",
            return_value=_COMMENTARY_PROMPT_GROUP,
        ),
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.generate_with_messages",
            new=AsyncMock(return_value='{"commentaries": {"0": "ok"}}'),
        ),
        patch.object(processor._recall_agent, "recall", new=AsyncMock(return_value="")),
    ):
        await processor._process_async(raw, events)

    second_call_kwargs = fake_storage.get_profile.await_args_list[1].kwargs
    assert second_call_kwargs["user_id"] == "__base__"
    assert second_call_kwargs["context_type"] == "agent_base_profile"


@pytest.mark.unit
async def test_processor_skips_when_agent_id_is_default():
    """agent_id == 'default' -> early return, RecallAgent not called."""
    processor = AgentMemoryProcessor()
    raw = SimpleNamespace(
        user_id="u1",
        device_id="d1",
        agent_id="default",
        source=ContextSource.CHAT_LOG,
        content_text="hi",
    )

    with patch.object(processor._recall_agent, "recall", new=AsyncMock()) as recall_mock:
        result = await processor._process_async(raw, [_make_event_ctx(0)])

    assert result == []
    assert recall_mock.await_count == 0


@pytest.mark.unit
async def test_processor_skips_when_no_events_in_prior_results():
    """No EVENT contexts in prior_results -> early return, RecallAgent not called."""
    processor = AgentMemoryProcessor()
    raw = _make_raw_context()

    fake_storage = MagicMock()
    fake_storage.get_agent = AsyncMock(return_value={"name": "Bot"})

    with (
        patch(
            "opencontext.context_processing.processor.agent_memory_processor.get_storage",
            return_value=fake_storage,
        ),
        patch.object(processor._recall_agent, "recall", new=AsyncMock()) as recall_mock,
    ):
        result = await processor._process_async(raw, [])

    assert result == []
    assert recall_mock.await_count == 0
