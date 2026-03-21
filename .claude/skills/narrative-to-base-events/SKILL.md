---
name: narrative-to-base-events
description: Use when extracting a character's experiences from narrative text (novels, scripts, stories) to generate structured hierarchical Base Events JSON for MineContext agent memory.
---

# Narrative to Base Events

## Overview

This skill is a structured multi-stage methodology for extracting a fictional character's experiences from narrative text and encoding them as hierarchical Base Events for MineContext agent memory. The core principle: treat the character as the sole first-person subject, extracting only events they directly participated in, then organize those events across four hierarchy levels (L0–L3).

For long texts, the skill uses a **subagent architecture** — the main agent coordinates the process while subagents handle segment-level work in parallel, overcoming context window limitations.

## When to Use

- User asks to extract character events from a novel, script, or story
- User wants to create Base Events / agent memory from fictional content
- User mentions converting narrative text to structured events

## When NOT to Use

- Source material is real-world event logs or factual records (use a data-import flow instead)
- Extraction is not character-centric (e.g., extracting world-building facts or plot summaries without a focal character)

## Prerequisites

The user must provide:

- **Narrative text** — file path or pasted directly into the conversation
- **Target character name** — the character whose perspective anchors all extraction
- **Output file path** — optional; defaults to `{character_name}_base_events.json`

## Pipeline Overview

| Stage | Name | Checkpoint |
|-------|------|------------|
| 1 | Text Ingestion & Segmentation | User confirms segmentation strategy |
| 2 | Narrative Comprehension | User confirms narrative summary accuracy |
| 3 | Character Research | User confirms character understanding |
| 4 | L0 Event Extraction | User reviews event completeness |
| 5 | Timeline Construction | User confirms timeline |
| 6 | Hierarchy Organization | User confirms hierarchy grouping |
| 7 | Field Completion & Validation | User spot-checks quality |
| 8 | JSON Generation | User confirms output |

## Hierarchy Levels

| Level | Dimension | Granularity | Example |
|-------|-----------|-------------|---------|
| L0 | Concrete event | A meaningful action, decision, or encounter | "Met mentor and accepted the mission" |
| L1 | Plot segment | Several related events forming a continuous sequence | "Exploring the forbidden chamber" |
| L2 | Plot unit | A complete narrative arc (conflict → resolution) | "Unraveling the mystery of the Stone" |
| L3 | Character stage | A major phase of character development | "Innocent Youth", "Awakening" |

## Coordination Model

The main agent acts as **controller** — it coordinates the pipeline, dispatches subagents, synthesizes their outputs, and handles all user checkpoints. Subagents are used for segment-level work that would exceed a single context window.

### Role Assignment

| Role | Responsibility | When Used |
|------|---------------|-----------|
| **Main Agent** | Pipeline coordination, segmentation, character research, synthesis, hierarchy, validation, JSON generation, all user checkpoints | Always |
| **Comprehension Subagent** | Read one segment, produce segment summary | Long text: one per segment (Stage 2) |
| **Extraction Subagent** | Read one segment, extract L0 events for the target character | Long text: one per segment (Stage 4) |
| **Revision Subagent** | Review the full merged L0 list against the narrative summary for completeness and consistency | After Stage 4 merge |

### Short Text Path

When the text fits in a single context window, the main agent performs all stages directly. No subagents are dispatched.

### Long Text Path

```
Main Agent                          Subagents
──────────                          ─────────
Stage 1: Segment text
Stage 2: Dispatch comprehension ──→ [Subagent per segment: read & summarize]
         Synthesize summaries   ←── [Return segment summaries]
         → User checkpoint
Stage 3: Character research
         → User checkpoint
Stage 4: Dispatch extraction   ──→ [Subagent per segment: extract L0 events]
         Merge & deduplicate    ←── [Return L0 event lists]
         Dispatch revision     ──→ [Revision subagent: review merged list]
         Apply corrections      ←── [Return corrections]
         → User checkpoint
Stage 5–8: Main agent handles
         (timeline, hierarchy, fields, JSON)
```

### Subagent Context

Each subagent receives a precisely crafted prompt containing only what it needs:

- **Comprehension subagent**: the segment text, segment number, total segments
- **Extraction subagent**: the segment text, character reference card, narrative summary, extraction rules (from Stage 4), segment number
- **Revision subagent**: the full merged L0 list, narrative summary, character reference card

Subagents do NOT receive the full pipeline context or conversation history.

---

## Stage 1: Text Ingestion & Segmentation

**Purpose**: Determine whether the full text fits in a single pass or requires segmented processing, then produce a clear plan before any extraction begins.

### Decision Logic

**Threshold**: approximately half the available context window (~50,000 characters or ~60–80 pages).

- **Short text path** (below threshold): read the full text in one pass, proceed directly to Stage 2 with no segmentation step.
- **Long text path** (above threshold): segment by natural boundaries — chapters, acts, or episodes. Produce a segmentation plan table and present it to the user before proceeding.

**Segmentation plan table format:**

| Segment | Chapters / Sections Covered |
|---------|-----------------------------|
| 1 | Chapters 1–8 |
| 2 | Chapters 9–17 |
| … | … |

### Checkpoint

Present the segmentation strategy (or confirm "no segmentation needed" for short texts). Proceed only after user confirms.

---

## Stage 2: Narrative Comprehension

**Purpose**: Read and understand the full narrative before any extraction begins. This builds the foundational understanding of plot, themes, characters, and narrative structure that informs all subsequent stages — without it, extraction is blind pattern-matching.

### Short Text

Read the entire text end-to-end. Produce the narrative summary directly.

### Long Text — Subagent Dispatch

Dispatch one **comprehension subagent** per segment. Each subagent receives:
- The segment text
- Segment number and total segment count
- Instruction: "Read this segment and produce a segment summary covering: key plot developments, characters involved, setting/time clues, and where the segment ends narratively."

After all subagents return, the main agent **synthesizes** the segment summaries into a unified narrative summary.

### Narrative Summary

Whether produced directly (short text) or synthesized from subagent outputs (long text), the narrative summary must contain:

- **Plot overview**: The story's main conflict, major plot beats, and resolution (5-10 sentences)
- **Key characters**: All significant characters and their roles (protagonist, antagonist, mentor, ally, etc.)
- **Setting & time period**: Where and when the story takes place, any major setting changes
- **Narrative structure**: Is the story linear, non-linear (flashbacks/flash-forwards), multi-POV, episodic?
- **Themes**: 3-5 core themes (e.g., redemption, betrayal, coming-of-age)

This summary serves two purposes:
1. It grounds Stage 3 (Character Research) — internet research becomes verification and enrichment rather than the primary source of understanding
2. It is passed to extraction subagents in Stage 4 as global context, so each subagent understands where its segment fits in the overall story

### Checkpoint

Present the narrative summary to the user. Proceed only after user confirms it accurately reflects the source material.

---

## Stage 3: Character Research

**Purpose**: Build a character reference card that serves as the anchor for all subsequent extraction. This prevents character drift — misidentifying events, misreading motivations, or losing track of the character's arc across long or segmented texts.

### Process

Use WebSearch to research the target character:

- Complete story arc and ending
- Personality analysis and core traits
- Key relationships (allies, antagonists, mentors)
- Major plot points and turning points

Produce a **character reference card** with these sections:

- **Basic info**: name, identity, faction/affiliation
- **Core personality traits**: 3–5 adjectives with brief evidence
- **Key relationships**: character name, role, nature of relationship
- **Arc overview**: starting state → key turning points → ending state

This card is referenced in every subsequent stage. Do not skip or abbreviate it.

### Fallback for Obscure or Original Works

If WebSearch yields insufficient results (fan fiction, unpublished works, niche titles):

1. Build the card from source text only
2. Explicitly inform the user: "Limited external information found. Reference card is built from source text alone — it may be incomplete."
3. User must explicitly confirm before proceeding

### Checkpoint

Present the reference card. Proceed only after user confirms accuracy.

---

## Stage 4: L0 Event Extraction

**Purpose**: Extract every concrete, meaningful event in which the target character directly participates — this is the raw material for all higher-level structure.

### Granularity Rule

L0 granularity is a **concrete, meaningful action, decision, or encounter**. It is coarser than a single dialogue line and finer than a full scene summary. Ask: "Did something change for this character — their situation, relationship, knowledge, or resolve?" If yes, it is an L0 event.

### Extraction Rules

- **Direct participation only**: the character must be present, act, speak, or be directly affected. Do not include events the character only hears about secondhand.
- **Fields to fill at this stage**: `title` (a short verb phrase) and `summary` (1–2 sentences).
- **Fields to leave empty**: time, keywords, entities, importance — these are filled in later stages.
- **Output order**: narrative order (the order events appear in the text). Chronological reordering happens in Stage 5, not here.

### Short Text

Main agent extracts L0 events directly from the full text.

### Long Text — Subagent Dispatch

Dispatch one **extraction subagent** per segment. Each subagent receives:
- The segment text
- The **character reference card** (from Stage 3)
- The **narrative summary** (from Stage 2) — so the subagent understands the full story context, not just its segment
- The extraction rules above (granularity rule, direct participation, fields to fill)
- Segment number and total segment count

Each subagent returns a list of L0 events (title + summary) found in its segment.

### Merge & Deduplication

After all extraction subagents return, the main agent merges their outputs:

- Concatenate all segment event lists in segment order
- **Deduplicate**: if the same character action in the same scene appears across segment boundaries (due to overlap), merge into one event. Recurring themes (e.g., multiple separate battles) are distinct events — keep them separate.

### Revision Subagent

After merging, dispatch a **revision subagent** that receives:
- The full merged L0 event list
- The narrative summary
- The character reference card

The revision subagent checks for:
- **Completeness**: any significant character events from the narrative summary that are missing from the L0 list?
- **Consistency**: any events that contradict the narrative summary or character reference card?
- **Granularity**: any events that should be split (too coarse) or merged (too fine)?
- **Scope**: any events included where the character didn't directly participate?

The revision subagent returns a list of corrections (additions, removals, edits). The main agent applies corrections to produce the final L0 list.

### Checkpoint

Present the complete L0 list (title + summary only, no other fields). User confirms completeness — are there missing events, events to remove, or events to split/merge?

---

## Stage 5: Timeline Construction

**Purpose**: Assign concrete timestamps to every L0 event, enabling the hierarchical time-bucketing that MineContext requires.

### Three Strategies (apply in priority order)

**Strategy 1 — Direct Extraction**: The text contains explicit calendar dates (e.g., "September 1, 1998", "Monday, March 5"). Use them directly. This is the most reliable strategy; apply wherever explicit dates exist.

**Strategy 2 — Anchor-Based Inference**: Establish time anchors from the text or from internet research (e.g., a known historical event the story references), then use relative references in the text ("three days later", "that winter", "two years after the war") to infer dates for other events.

**Strategy 3 — Synthetic Sequential**: No real-world timeline is determinable. Assign a synthetic timeline starting from `2000-01-01T00:00:00+08:00`, with intervals inferred from textual pacing hints:

| Textual hint | Approximate interval |
|---|---|
| Same scene / same day | Hours apart |
| "A few days later" / "next week" | Days apart |
| "Months passed" | Weeks to months apart |
| "Years passed" / major life transition | 1–5 years apart |

Strategies may be mixed within one work (e.g., explicit dates for some events, anchor-based inference for others, synthetic for undatable sections).

### Non-Linear Narratives

For stories with flashbacks, flash-forwards, or non-chronological structure: arrange events in **story-chronological order** — the order they actually happened within the story world, not the order they appear in the text.

### Checkpoint

Present a timeline overview table:

| # | Event Title | Assigned Time |
|---|-------------|---------------|
| 1 | … | 1998-09-01T09:00:00+08:00 |
| … | … | … |

User confirms the timeline before proceeding.

---

## Stage 6: Hierarchy Organization

**Purpose**: Organise the confirmed L0 events into a three-level summary hierarchy (L1 → L2 → L3) that reflects the character's development arc.

### Construction Process

Build bottom-up, grouping by character development arc:

- **L0 → L1 (Plot Segments)**: group related L0 events into continuous plot segments. Typical L1 contains 3–8 L0 events.
- **L1 → L2 (Plot Units)**: group L1 segments into complete narrative arcs (conflict → resolution). Typical L2 contains 2–5 L1 segments.
- **L2 → L3 (Character Stages)**: group plot units into major phases of character development. Typical L3 contains 2–5 L2 units. A work typically produces 2–5 L3 nodes.

### Construction Rules

- **Bottom-up only** — never create a summary level without its constituent lower-level events already defined.
- Each summary node's `title` captures the theme of that group.
- Each summary node's `summary` synthesises what happens across its children (2–4 sentences).
- `event_time_start` = minimum `event_time_start` of all direct children.
- `event_time_end` = for each direct child, take its `event_time_end` if present, otherwise its `event_time_start`; the parent's `event_time_end` is the maximum of these values.
- Every L0 must belong to exactly one L1; every L1 to exactly one L2; every L2 to exactly one L3.

### Checkpoint

Present a tree outline showing titles only at each level, for example:

```
L3: Innocent Youth
  L2: Discovery and Arrival
    L1: Hagrid Delivers the Truth
      L0: Hagrid arrives at the hut on the rock
      L0: Harry learns his parents were killed by Voldemort
    L1: Diagon Alley and First Supplies
      L0: ...
  L2: ...
```

User confirms the grouping and titles before proceeding.

---

## Stage 7: Field Completion & Validation

**Purpose**: Complete all remaining fields on every node (L0 through L3) and validate the full tree for consistency and correctness.

### Fields to Complete

For all nodes:

- **`keywords`**: 3–5 keywords drawn from the summary, capturing themes, actions, and concepts.
- **`entities`**: character names, place names, organizations, and artifacts referenced in the event.
- **`importance`**: 1–10 scale:
  - 1–3: Routine, low-impact
  - 4–6: Notable, contributes to story
  - 7–9: Major turning points, critical decisions
  - 10: Life-defining moments

### Information Sources

Use narrative text and internet research only. Do NOT invent information not traceable to these sources.

### Validation Checks

Before presenting the checkpoint, verify all of the following:

- Every summary is consistent with the Stage 3 character reference card (no character drift).
- Timeline is monotonically increasing with no contradictions.
- Hierarchy coverage is complete: every L0 belongs to an L1, every L1 to an L2, every L2 to an L3.
- `event_time_start` / `event_time_end` constraints are satisfied at every parent-child boundary: `parent.event_time_start` ≤ `min(children[*].event_time_start)` and `parent.event_time_end` ≥ `max(children[*].effective_end_time)`.

### Checkpoint

Present a sample of completed nodes — a few from each level — for spot-check. User confirms quality before proceeding.

---

## Stage 8: JSON Generation

**Purpose**: Assemble the validated hierarchy into a `BaseEventsRequest`-format JSON file and write it to the output path.

### Structure

Assemble as a nested JSON with L3 nodes at the top level, each containing its full child tree. The nesting pattern is:

```json
{
  "events": [
    {
      "title": "Character Stage Title",
      "hierarchy_level": 3,
      "event_time_start": "...", "event_time_end": "...",
      "children": [
        {
          "hierarchy_level": 2,
          "children": [
            {
              "hierarchy_level": 1,
              "children": [
                { "hierarchy_level": 0 }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

For the full field-level schema and all validation rules, see `references/base-event-schema.md`.

### Key Format Rules

- L3 nodes appear at the top level of the `events` array.
- `children` and `event_time_end` are required for all L1, L2, and L3 nodes.
- L0 nodes have no `children` field.
- The `refs` field must be omitted — the server constructs parent-child reference links automatically.
- All times must be ISO 8601 with timezone offset (e.g., `+08:00`).

### 500-Event Limit

If the total node count across all levels exceeds 500, split into multiple output files. Each file must contain complete sub-trees: own L3 nodes with their full L2 → L1 → L0 descendants. Do not split a tree in the middle of a hierarchy.

### Output

Write to the output file path specified by the user (or the default `{character_name}_base_events.json`). Report the output file path and the event count per level (L0, L1, L2, L3).

### Checkpoint

Present the output file path and event counts. User confirms the output is acceptable.

---

## Common Mistakes

| Mistake | Correction |
|---------|------------|
| Including events the character didn't directly participate in | Only extract events where the character is present, acts, or speaks |
| Fabricating details not in the text or verifiable sources | All content must be traceable to source text or internet research |
| Arranging flashback events in narrative order instead of chronological | Always use story-chronological order |
| Skipping the character research stage | The reference card prevents drift — never skip it |
| Creating hierarchy levels without constituent events | Always build bottom-up: L0 first, then aggregate upward |
| Splitting hierarchy trees across multiple JSON files | Each file must contain complete sub-trees (own L3 + full children) |
| Using `event_time` instead of `event_time_start` | The field is `event_time_start` (ISO 8601 with timezone) |
| Giving extraction subagents only segment text without narrative summary | Subagents need the full narrative summary to understand context beyond their segment |
| Skipping the revision subagent after merging | Revision catches gaps and inconsistencies that segment-level extraction misses |
