"""Unit tests for VikingDB query optimization (combined context-type queries)."""

import pytest

from opencontext.storage.backends.vikingdb_backend import VikingDBBackend


@pytest.mark.unit
class TestSplitTypesByUserScope:
    """Tests for _split_types_by_user_scope helper."""

    def test_no_user_id_returns_single_group(self):
        types = ["document", "event", "knowledge", "agent_base_event"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id=None, device_id=None)
        assert len(result) == 1
        assert result[0] == (types, False)

    def test_with_user_id_splits_into_two_groups(self):
        types = ["document", "event", "agent_base_event", "agent_base_l1_summary"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id="u1", device_id=None)
        assert len(result) == 2
        regular_group, base_group = result
        assert regular_group == (["document", "event"], False)
        assert base_group == (["agent_base_event", "agent_base_l1_summary"], True)

    def test_with_device_id_only_splits(self):
        types = ["event", "agent_base_event"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id=None, device_id="d1")
        assert len(result) == 2

    def test_only_regular_types_returns_single_group(self):
        types = ["document", "event", "knowledge"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id="u1", device_id=None)
        assert len(result) == 1
        assert result[0] == (types, False)

    def test_only_base_types_returns_single_group(self):
        types = ["agent_base_event", "agent_base_l1_summary"]
        result = VikingDBBackend._split_types_by_user_scope(types, user_id="u1", device_id=None)
        assert len(result) == 1
        assert result[0] == (types, True)

    def test_empty_types_returns_empty(self):
        result = VikingDBBackend._split_types_by_user_scope([], user_id="u1", device_id=None)
        assert result == []
