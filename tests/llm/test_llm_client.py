"""Unit tests for LLMClient._openai_chat_completion model override."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from opencontext.llm.llm_client import LLMClient, LLMType


def _make_client() -> LLMClient:
    config = {
        "provider": "openai",
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "model": "default-model",
        "max_concurrent": 1,
    }
    return LLMClient(llm_type=LLMType.CHAT, config=config)


@pytest.mark.unit
async def test_model_override_via_kwarg_applied():
    """Passing model=<name> as kwarg overrides self.model in create_params."""
    client = _make_client()
    fake_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    client.client.chat.completions.create = fake_create

    await client._openai_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="override-model",
    )

    assert fake_create.await_count == 1
    kwargs = fake_create.await_args.kwargs
    assert kwargs["model"] == "override-model"


@pytest.mark.unit
async def test_model_override_absent_uses_default():
    """Without model kwarg, create_params.model is self.model."""
    client = _make_client()
    fake_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    client.client.chat.completions.create = fake_create

    await client._openai_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
    )

    kwargs = fake_create.await_args.kwargs
    assert kwargs["model"] == "default-model"
