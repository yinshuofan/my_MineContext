# Narrative-to-Base-Events Skill Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Claude Code Skill (`SKILL.md` + reference file) that provides a systematic 7-stage methodology for extracting character events from narrative text and generating hierarchical Base Events JSON.

**Architecture:** Two markdown files in `.claude/skills/narrative-to-base-events/`. `SKILL.md` contains the full methodology (frontmatter + 7-stage pipeline with decision trees, extraction rules, and validation checks). `references/base-event-schema.md` contains the `BaseEventsRequest` JSON schema derived from the codebase, kept separate to avoid bloating the main skill file.

**Tech Stack:** Markdown (Claude Code Skill format)

**Spec:** `docs/superpowers/specs/2026-03-21-narrative-to-base-events-skill-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `.claude/skills/narrative-to-base-events/SKILL.md` | Create | Main methodology: frontmatter, overview, 7-stage pipeline, common mistakes |
| `.claude/skills/narrative-to-base-events/references/base-event-schema.md` | Create | `BaseEventsRequest` JSON schema, field descriptions, complete example |

---

## Task 1: Create base-event-schema.md Reference File

**Files:**
- Create: `.claude/skills/narrative-to-base-events/references/base-event-schema.md`

This file is created first because SKILL.md references it.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p .claude/skills/narrative-to-base-events/references
```

- [ ] **Step 2: Write base-event-schema.md**

The file must contain:
1. The `BaseEventItem` field table (name, type, required/optional, default value, description) derived from `opencontext/server/routes/agents.py:69-78`. Include defaults: `event_time_start` defaults to current time when omitted, `event_time_end` defaults to `event_time_start` for L0 nodes, `importance` defaults to 5, `hierarchy_level` defaults to 0
2. The `BaseEventsRequest` wrapper structure: `events: List[BaseEventItem]` (min 1 item), this is the top-level request body for `POST /api/agents/{agent_id}/base/events`
3. Validation rules summary (from the server-side `_validate_base_event_tree`), explicitly enumerating:
   - `hierarchy_level > 0` requires `event_time_end` and non-empty `children`
   - `hierarchy_level == 0` cannot have `children`
   - Each child's `hierarchy_level` must equal `parent.hierarchy_level - 1` (no skipping levels)
   - `event_time_start <= event_time_end` on the same node
   - Parent's time range must cover all direct children's time ranges
   - Maximum `hierarchy_level` is 3; maximum nesting depth is 4 levels
   - Total event count across all levels must not exceed 500
4. A complete nested JSON example showing all 4 levels (L0-L3) with all fields populated — use a well-known fictional character (e.g., a classic literary or film character) so the example is immediately recognizable and illustrative
5. A minimal L0-only example (flat list, no hierarchy)
6. Notes on server-side behavior: `refs` auto-constructed by server (do not include in JSON), `event_time_end` defaults to `event_time_start` for L0 nodes, 500-event limit per request

Source material:
- Field definitions: `opencontext/server/routes/agents.py:69-82`
- Validation rules: `opencontext/server/routes/agents.py:85-186`
- Design spec: `docs/superpowers/specs/2026-03-21-narrative-to-base-events-skill-design.md` (Stage 7, Constraints)

- [ ] **Step 3: Verify**

Read the file back to confirm:
- All `BaseEventItem` fields are documented
- JSON examples are valid JSON (no trailing commas, correct nesting)
- `hierarchy_level` values in examples match nesting depth
- `event_time_start`/`event_time_end` coverage rules are stated

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/narrative-to-base-events/references/base-event-schema.md
git commit -m "docs: add BaseEventsRequest schema reference for narrative skill"
```

---

## Task 2: Create SKILL.md — Frontmatter & Overview

**Files:**
- Create: `.claude/skills/narrative-to-base-events/SKILL.md`

This task writes the frontmatter and the top sections (Overview, When to Use, Prerequisites, Pipeline Overview table). The stage details are added in Task 3.

- [ ] **Step 1: Write frontmatter + overview sections**

The file must start with:

```yaml
---
name: narrative-to-base-events
description: Use when extracting a character's experiences from narrative text (novels, scripts, stories) to generate structured hierarchical Base Events JSON for MineContext agent memory.
---
```

Followed by sections:

1. **Overview** — What this skill is (methodology, not prompt), core principle in 2 sentences
2. **When to Use** — Trigger conditions (user asks to extract character events, create agent memory from fiction, convert narrative to structured events)
3. **When NOT to Use** — Not for real-world event logs, not for non-character-centric extraction
4. **Prerequisites** — What user must provide (narrative text, character name, optional output path)
5. **Pipeline Overview** — The 7-stage table from the spec (Stage, Name, Checkpoint columns only — keep it scannable)
6. **Hierarchy Levels** — The 4-level definition table (L0: concrete events, L1: plot segments, L2: plot units, L3: character stages) with examples

Source material:
- Spec: `docs/superpowers/specs/2026-03-21-narrative-to-base-events-skill-design.md` (Skill Metadata, Trigger Conditions, Prerequisites, Staged Pipeline, Stage 5 hierarchy definitions)

- [ ] **Step 2: Verify**

- Frontmatter has exactly 2 fields: `name` and `description`
- `description` starts with "Use when"
- Total frontmatter < 1024 characters
- No implementation details leaked into overview sections

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/narrative-to-base-events/SKILL.md
git commit -m "docs: add SKILL.md frontmatter and overview for narrative-to-base-events"
```

---

## Task 3: Write SKILL.md — Stage Details (Stages 1-4)

**Files:**
- Modify: `.claude/skills/narrative-to-base-events/SKILL.md`

Append Stages 1-4 to the SKILL.md file. Each stage section follows the pattern: purpose, decision logic, concrete instructions, checkpoint.

- [ ] **Step 1: Write Stage 1 (Text Ingestion & Segmentation)**

Content:
- Short text path vs long text path decision (threshold: ~half of context window, ~50K chars)
- Short path: read full text, proceed
- Long path: segment by natural boundaries (chapters/acts/episodes), produce segmentation plan table, present to user
- Checkpoint: user confirms segmentation strategy

- [ ] **Step 2: Write Stage 2 (Character Research)**

Content:
- Use WebSearch to research character (arc, personality, relationships, key events)
- Produce character reference card (basic info, personality, relationships, arc overview)
- This card is the **anchor** for all subsequent stages
- Fallback: if WebSearch yields nothing (obscure/original work), build card from text only, inform user
- Checkpoint: user confirms reference card accuracy

- [ ] **Step 3: Write Stage 3 (L0 Event Extraction)**

Content:
- L0 granularity definition: concrete, meaningful action/decision/encounter
- Extraction rules: direct participation only, title (verb phrase) + summary (1-2 sentences), no time/keywords yet, output in narrative order (the order events appear in the text; chronological reordering happens in Stage 4)
- Long text: process segments sequentially, deduplicate across segments (same action in same scene = merge; recurring themes = keep separate)
- Checkpoint: present complete L0 list (title + summary), user confirms completeness

- [ ] **Step 4: Write Stage 4 (Timeline Construction)**

Content:
- Three strategies in priority order:
  1. Direct Extraction (explicit dates in text)
  2. Anchor-Based Inference (time anchors + relative references)
  3. Synthetic Sequential (from `2000-01-01`, intervals by textual hints)
- Strategies can be mixed within one work
- Non-linear narratives: arrange in story-chronological order, not narrative order
- Checkpoint: present timeline overview, user confirms

- [ ] **Step 5: Verify**

Read Stages 1-4 in the file. Confirm:
- Each stage has: purpose statement, concrete instructions, checkpoint
- Stage 2 has the fallback for obscure works
- Stage 3 has long-text deduplication guidance
- Stage 4 has all 3 strategies + non-linear narrative handling

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/narrative-to-base-events/SKILL.md
git commit -m "docs: add Stages 1-4 to narrative-to-base-events skill"
```

---

## Task 4: Write SKILL.md — Stage Details (Stages 5-7) & Common Mistakes

**Files:**
- Modify: `.claude/skills/narrative-to-base-events/SKILL.md`

Append Stages 5-7 and a Common Mistakes section.

- [ ] **Step 1: Write Stage 5 (Hierarchy Organization)**

Content:
- Build bottom-up: L0 → L1 (plot segments, 3-8 L0s) → L2 (plot units, 2-5 L1s) → L3 (character stages, 2-5 L2s)
- Construction rules: bottom-up only, title captures theme, summary synthesizes children (2-4 sentences)
- Time rules: `event_time_start` = min of children's starts; `event_time_end` = max of children's ends (fallback to `event_time_start` for L0)
- Coverage rule: every L0 in exactly one L1, every L1 in one L2, every L2 in one L3
- Checkpoint: present tree outline (titles only at each level)

- [ ] **Step 2: Write Stage 6 (Field Completion & Validation)**

Content:
- Complete for all nodes: `keywords` (3-5), `entities` (people/places/orgs), `importance` (1-10 scale with guidance)
- Information sources: text + internet research only, no fabrication
- Validation checks: consistency with reference card, monotonic timeline, complete hierarchy coverage, time constraints at every parent-child boundary
- Checkpoint: present sample nodes from each level for spot-check

- [ ] **Step 3: Write Stage 7 (JSON Generation)**

Content:
- Assemble nested JSON per `BaseEventsRequest` format
- Include a brief structural JSON snippet showing the nesting pattern (L3 → L2 → L1 → L0), but reference `references/base-event-schema.md` for full field-level schema details
- Key format rules: L3 at top level, `children` + `event_time_end` required for L1+, `refs` omitted (server auto-constructs), ISO 8601 with timezone
- Write to file, report path and event counts per level
- 500-event limit: if exceeded, split into complete sub-trees per file
- Checkpoint: user confirms output

- [ ] **Step 4: Write Common Mistakes section**

A table of common mistakes and corrections:

| Mistake | Correction |
|---------|-----------|
| Including events the character didn't directly participate in | Only extract events where the character is present, acts, or speaks |
| Fabricating details not in the text or verifiable sources | All content must be traceable to source text or internet research |
| Arranging flashback events in narrative order instead of chronological | Always use story-chronological order |
| Skipping the character research stage | The reference card prevents drift — never skip it |
| Creating hierarchy levels without constituent events | Always build bottom-up: L0 first, then aggregate upward |
| Splitting hierarchy trees across multiple JSON files | Each file must contain complete sub-trees (own L3 + full children) |
| Using `event_time` instead of `event_time_start` | The field is `event_time_start` (ISO 8601 with timezone) |

- [ ] **Step 5: Verify SKILL.md**

Read the complete SKILL.md. Confirm:
- All 7 stages are present with checkpoints
- Stage 7 includes a brief structural JSON snippet and references `references/base-event-schema.md`
- Common mistakes table covers key pitfalls from the spec
- Total file length is reasonable (aim for < 500 lines)

- [ ] **Step 6: Cross-file consistency check**

Verify SKILL.md and `references/base-event-schema.md` together:
- Stage 7 in SKILL.md correctly references `references/base-event-schema.md`
- Hierarchy level definitions in SKILL.md (L0-L3) match the examples in the schema file
- Field names used throughout SKILL.md match the schema (e.g., `event_time_start` not `event_time`)
- The structural JSON snippet in Stage 7 is consistent with the full example in the schema file

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/narrative-to-base-events/SKILL.md
git commit -m "docs: complete narrative-to-base-events skill with all stages and common mistakes"
```

---

## Implementation Notes

### What does NOT need to be created
- No Python code — this is a pure methodology skill
- No test files — skill correctness is verified by reading and reviewing
- No configuration changes — skills are auto-discovered by Claude Code from `.claude/skills/`

### Key design decisions
1. **Schema in separate file**: `base-event-schema.md` is a reference file loaded on demand, keeping SKILL.md focused on methodology. SKILL.md references it in Stage 7 rather than duplicating the schema.
2. **Complete JSON example in schema file**: The schema reference includes a realistic multi-level example (not placeholder text) so Claude has a concrete model to follow when generating output.
3. **SKILL.md < 500 lines**: The skill is loaded into context on every invocation. Keeping it concise ensures it doesn't crowd out the narrative text the user provides.
