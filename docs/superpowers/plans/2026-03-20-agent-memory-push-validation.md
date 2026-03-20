# Agent Memory Push Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return HTTP 400 when `push/chat` requests `agent_memory` processor with an unregistered `agent_id`, instead of silently skipping.

**Architecture:** Single validation check added to `push_chat()` handler, before chat batch persistence. Uses existing `storage.get_agent()` to verify agent existence. No new files, no new abstractions.

**Tech Stack:** Python / FastAPI / existing `UnifiedStorage.get_agent()`

**Spec:** `docs/superpowers/specs/2026-03-20-agent-memory-push-validation.md`

---

### Task 1: Add agent existence validation to push_chat

**Files:**
- Modify: `opencontext/server/routes/push.py:399-403` (insert after `__base__` check)

- [ ] **Step 1: Add validation logic**

In `push_chat()`, after the existing `__base__` reserved check (line 402) and before the multimodal processing (line 405), add:

```python
        # Validate agent exists when agent_memory processor is requested
        if (
            "agent_memory" in request.processors
            and request.agent_id is not None
            and request.agent_id != "default"
        ):
            storage = get_storage()
            agent = await storage.get_agent(request.agent_id)
            if not agent:
                raise HTTPException(
                    400,
                    f"Agent '{request.agent_id}' is not registered. "
                    f"Please register the agent via POST /api/agents before using agent_memory processor.",
                )
```

Note: `get_storage()` is already imported in this file (used at line 409). The `storage` variable here is scoped to this validation block; the later `storage = get_storage()` at line 409 is fine — both are cheap singleton lookups.

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile opencontext/server/routes/push.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add opencontext/server/routes/push.py
git commit -m "feat(push): validate agent exists before agent_memory processing"
```

---

### Task 2: Update API documentation

**Files:**
- Modify: `docs/api_reference.md` (push/chat error responses section, around line 93)
- Modify: `docs/curls.sh` (push/chat section)

- [ ] **Step 1: Add 400 error case to api_reference.md**

In the push/chat response section, after the existing timeout error example (line 101), add:

```markdown
**Agent 未注册**
```json
{
  "detail": "Agent 'my_agent' is not registered. Please register the agent via POST /api/agents before using agent_memory processor."
}
```
> 当 `processors` 包含 `"agent_memory"` 且 `agent_id` 指定了未注册的 agent 时返回 400。
```

- [ ] **Step 2: Fix curls.sh agent_id inconsistency**

`docs/curls.sh` uses `agent_id: "agent_001"` in push/chat examples (lines 111, 332) and memory-cache (line 369), but the agents CRUD section registers `assistant_01`. With the new validation, the dual-processor push example at line 102-114 would return 400 because `agent_001` is never registered.

Fix: Change all `agent_001` references in curls.sh to `assistant_01` to be consistent with the agents CRUD section. This affects lines 111, 332, and 369.

- [ ] **Step 3: Commit**

```bash
git add docs/api_reference.md docs/curls.sh
git commit -m "docs: document 400 error for unregistered agent in push/chat"
```
