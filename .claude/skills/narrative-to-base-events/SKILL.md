---
name: narrative-to-base-events
description: Use when extracting a character's experiences from narrative text (novels, scripts, stories) to generate structured hierarchical Base Events JSON for MineContext agent memory.
---

# Narrative to Base Events

## Overview

This skill is a structured multi-stage methodology for extracting a fictional character's experiences from narrative text and encoding them as hierarchical Base Events for MineContext agent memory. The core principle: treat the character as the sole first-person subject, extracting only events they directly participated in, then organize those events across four time-granularity levels (L0–L3).

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
| 2 | Character Research | User confirms character understanding |
| 3 | L0 Event Extraction | User reviews event completeness |
| 4 | Timeline Construction | User confirms timeline |
| 5 | Hierarchy Organization | User confirms hierarchy grouping |
| 6 | Field Completion & Validation | User spot-checks quality |
| 7 | JSON Generation | User confirms output |

## Hierarchy Levels

| Level | Dimension | Granularity | Example |
|-------|-----------|-------------|---------|
| L0 | Concrete event | A meaningful action, decision, or encounter | "Met mentor and accepted the mission" |
| L1 | Plot segment | Several related events forming a continuous sequence | "Exploring the forbidden chamber" |
| L2 | Plot unit | A complete narrative arc (conflict → resolution) | "Unraveling the mystery of the Stone" |
| L3 | Character stage | A major phase of character development | "Innocent Youth", "Awakening" |

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

## Stage 2: Character Research

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

## Stage 3: L0 Event Extraction

**Purpose**: Extract every concrete, meaningful event in which the target character directly participates — this is the raw material for all higher-level structure.

### Granularity Rule

L0 granularity is a **concrete, meaningful action, decision, or encounter**. It is coarser than a single dialogue line and finer than a full scene summary. Ask: "Did something change for this character — their situation, relationship, knowledge, or resolve?" If yes, it is an L0 event.

### Extraction Rules

- **Direct participation only**: the character must be present, act, speak, or be directly affected. Do not include events the character only hears about secondhand.
- **Fields to fill at this stage**: `title` (a short verb phrase) and `summary` (1–2 sentences).
- **Fields to leave empty**: time, keywords, entities, importance — these are filled in later stages.
- **Output order**: narrative order (the order events appear in the text). Chronological reordering happens in Stage 4, not here.

### Long Text Processing

For segmented texts, process segments sequentially:

- After each segment, append new L0 events to the global list
- **Deduplication rule**: if the same character action in the same scene appears across segment boundaries (due to overlap), merge into one event. Recurring themes (e.g., multiple separate battles) are distinct events — keep them separate.
- After all segments are processed, present the merged global L0 list in narrative order.

### Checkpoint

Present the complete L0 list (title + summary only, no other fields). User confirms completeness — are there missing events, events to remove, or events to split/merge?

---

## Stage 4: Timeline Construction

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
