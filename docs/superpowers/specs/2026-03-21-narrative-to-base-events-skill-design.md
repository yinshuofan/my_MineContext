# Narrative-to-Base-Events Skill Design

## Goal

Create a Claude Code Skill that provides a systematic methodology for extracting a character's experiences from narrative text (novels, scripts, stories) and generating structured hierarchical Base Events JSON compatible with MineContext's `BaseEventsRequest` format.

## What This Skill Is

A **methodology skill** — not a prompt template, not code. It guides Claude through a repeatable, quality-controlled process for converting unstructured narrative into structured agent memory. The skill defines:

- How to approach different text lengths (short vs. long)
- How to research and anchor the character to prevent drift
- How to extract events at the right granularity
- How to handle ambiguous or missing time information
- How to build the 4-level hierarchy from character arc structure
- How to fill in missing structured fields using verifiable sources
- What the final JSON deliverable must look like

## Scope

- **Input**: Narrative text (file path or pasted text) + character name
- **Output**: A JSON file conforming to `BaseEventsRequest` schema with nested `children` structure
- **Process**: Entirely external to MineContext service — no API calls, no server dependency
- **Information sources**: The narrative text itself + internet research (WikiPedia, character analyses, etc.). Claude must NOT fabricate information through unsupported inference.

## Skill Metadata

```yaml
---
name: narrative-to-base-events
description: Use when extracting a character's experiences from narrative text
  (novels, scripts, stories) to generate structured hierarchical Base Events
  JSON for MineContext agent memory.
---
```

### Trigger Conditions

- User asks to extract character events from a novel, script, or story
- User wants to create Base Events / agent memory from fictional content
- User mentions converting narrative text to structured events

### Prerequisites

User must provide:
1. Narrative text (file path or pasted directly)
2. Target character name
3. Output file path (optional, defaults to `{character_name}_base_events.json`)

## Skill File Structure

```
narrative-to-base-events/
  SKILL.md                    # Main methodology (staged pipeline)
  references/
    base-event-schema.md      # BaseEventsRequest JSON schema + examples
```

## Staged Pipeline

The methodology consists of 7 sequential stages. Each stage has defined inputs, outputs, and a user checkpoint before proceeding.

| Stage | Name | Input | Output | Checkpoint |
|-------|------|-------|--------|------------|
| 1 | Text Ingestion & Segmentation | Raw text + character name | Segmentation plan (or confirmation that full text fits in context) | User confirms segmentation strategy |
| 2 | Character Research | Character name + work title | Character reference card (internet research results) | User confirms character understanding is accurate |
| 3 | L0 Event Extraction | Text (per-segment) + character reference card | L0 event list (title + summary) | User reviews event completeness |
| 4 | Timeline Construction | L0 event list + time clues from text | `event_time_start` for each L0 event | User confirms timeline reasonableness |
| 5 | Hierarchy Organization | L0 event list | L1/L2/L3 summary nodes + tree structure | User confirms hierarchy grouping |
| 6 | Field Completion & Validation | Complete event tree | Completed keywords/entities/importance | User spot-checks completion quality |
| 7 | JSON Generation | Final event tree | `BaseEventsRequest` JSON file | User confirms output |

### Design Rationale

- **Stage 2 before Stage 3**: Character research must happen before event extraction. The character reference card serves as an anchor throughout the process, preventing drift.
- **Stage 3 and 4 are separated**: Events are extracted first, timeline assigned second. Timeline construction requires a global view — knowing all events before assigning reasonable time positions.
- **Every stage has a checkpoint**: The user can correct course at each step. This is the primary quality control mechanism.

## Stage Details

### Stage 1: Text Ingestion & Segmentation

**Short text path**: If the text fits comfortably within approximately half the available context window (rough guideline: < ~50,000 characters, or ~60-80 pages of novel text), read the full text in one pass. Skip segmentation. Proceed to Stage 2.

**Long text path**: Otherwise:
- Assess total text length
- Segment along natural boundaries (chapters, acts, episodes)
- Each segment should fit within a single context window
- Produce a segmentation plan table (segment number, start/end position, corresponding chapters)
- Present plan to user for confirmation
- Stage 3 will process segments sequentially, merging results into a global event list

**Judgment criteria**: If the text can be fully loaded and processed in one context window, use the short path. Otherwise, use the long path.

### Stage 2: Character Research

Use WebSearch to research the character:
- Character's complete story arc and ending in the work
- Character personality analysis, relationships
- Key plot points and their interpretations
- Critical events and turning points

Produce a **character reference card** containing:
- Basic info (name, identity, faction/affiliation)
- Core personality traits
- Key relationships (with other characters)
- Arc overview (starting state -> turning points -> ending state)

This card is the **anchor reference** for all subsequent stages. Event summaries and hierarchy organization must stay consistent with this card.

**Fallback for obscure/original works**: If WebSearch yields insufficient results (e.g., self-published fiction, original characters), construct the reference card solely from the source text. Inform the user that the card is text-derived only and may be less complete. The user must explicitly confirm before proceeding.

**Checkpoint**: Present the reference card to the user. Confirm accuracy before proceeding.

### Stage 3: L0 Event Extraction

**L0 granularity**: A concrete, meaningful action, decision, or encounter. Not as granular as a single dialogue line or reaction, but a distinct event that advances the character's story.

Examples:
- "Met mentor and accepted the mission"
- "Defeated the guardian in the secret chamber"
- "Discovered the truth about their origin"

**Extraction rules**:
- Only extract events the character **directly participates in** (present, acts, or speaks)
- Each L0 event has a `title` (verb phrase) and `summary` (1-2 sentences: what happened, what the character did, direct outcome)
- Do NOT fill `event_time_start`, `keywords`, `entities`, `importance` at this stage — those come later
- Maintain narrative order in the list

**Long text processing**:
- Process each segment from Stage 1 sequentially
- After each segment, output the L0 events found
- Deduplicate across segments — if the same character action in the same scene was partially covered in a previous segment, merge rather than duplicate. Distinguish genuine duplicates from recurring themes (e.g., multiple similar battles are separate events)
- After all segments are processed, merge into a global L0 list in narrative order

**Checkpoint**: Present the complete L0 list (title + summary only) to the user. Confirm completeness — no missing or incorrect events.

### Stage 4: Timeline Construction

Assign `event_time_start` to every L0 event. Three strategies, attempted in priority order:

**Strategy 1 — Direct Extraction**: The text contains explicit dates/times (e.g., "September 1, 1998"). Use them directly.

**Strategy 2 — Anchor-Based Inference**: Establish time anchors from the text or internet research (e.g., "the story begins in year X"), then combine with relative time references in the text ("three days later", "that winter") to infer dates. When relative gaps are ambiguous, make reasonable estimates based on context.

**Strategy 3 — Synthetic Sequential**: When no real timeline can be determined, assign synthetic but internally consistent times starting from `2000-01-01T00:00:00+08:00`. Adjust intervals based on textual hints:
- Events on the same day: hours apart
- "A few days later": days apart
- "Years passed": appropriate year gap

The strategies can be mixed within a single work — some events may have known dates while others require inference or synthetic assignment.

**Non-linear narratives**: If the text uses flashbacks or non-chronological structure, events should still be arranged in **story-chronological order** (the order events actually occurred in the story world), not the order they appear in the narrative. Use textual clues and internet research to reconstruct the correct chronological sequence.

**Checkpoint**: Present a timeline overview (event title + assigned time) to the user. Confirm the temporal order and spacing make sense.

### Stage 5: Hierarchy Organization

Build the hierarchy bottom-up from L0, grouping by **character development arc**:

**L0 → L1 (Plot Segments)**: Group related L0 events into continuous plot segments. An L1 represents a coherent sequence of events (an encounter, a chase, a series of connected actions). Typical L1 contains 3-8 L0 events.

**L1 → L2 (Plot Units)**: Group related L1 segments into complete narrative arcs — a conflict from inception through resolution. Typical L2 contains 2-5 L1 segments.

**L2 → L3 (Character Stages)**: Group plot units into major phases of the character's development. Each L3 represents a distinct stage of the character's arc (e.g., "Innocent Youth", "Awakening", "Final Confrontation"). Typical L3 contains 2-5 L2 units. A complete work typically produces 2-5 L3 nodes.

**Construction rules**:
- Bottom-up only — never create a summary level without its constituent lower-level events
- Each summary node's `title` captures the theme of that group
- Each summary node's `summary` synthesizes what happens across its children (2-4 sentences)
- `event_time_start` = minimum `event_time_start` of all direct children
- `event_time_end` = for each direct child, take its `event_time_end` if present, otherwise its `event_time_start`; the parent's `event_time_end` is the maximum of these values
- Every L0 must belong to exactly one L1; every L1 to exactly one L2; every L2 to exactly one L3

**Checkpoint**: Present the tree structure outline (title only at each level) to the user. Confirm the grouping makes sense.

### Stage 6: Field Completion & Validation

Complete remaining fields for every node (L0 through L3):

- `keywords`: 3-5 keywords extracted from the summary, capturing key themes, actions, and concepts
- `entities`: Character names, place names, organizations, artifacts mentioned in the event
- `importance`: 1-10 scale based on significance to the character's arc
  - 1-3: Routine, low-impact events
  - 4-6: Notable events that contribute to the story
  - 7-9: Major turning points, critical decisions
  - 10: Life-defining moments

**Information sources for completion**: The narrative text and internet research (character analyses, wiki articles). Do NOT invent information that cannot be traced to these sources.

**Validation checks**:
- Every summary's content is consistent with the Stage 2 character reference card
- Timeline is monotonically increasing (no logical contradictions)
- Hierarchy coverage is complete (every L0 belongs to an L1, every L1 to an L2, every L2 to an L3)
- `event_time_start` / `event_time_end` constraints are satisfied at every parent-child boundary

**Checkpoint**: Present a sample of completed nodes (a few from each level) for user spot-check.

### Stage 7: JSON Generation

Assemble the final nested JSON conforming to `BaseEventsRequest` format:

```json
{
  "events": [
    {
      "title": "L3 stage title",
      "summary": "Stage summary...",
      "event_time_start": "2000-01-01T00:00:00+08:00",
      "event_time_end": "2000-06-15T23:59:59+08:00",
      "keywords": ["keyword1", "keyword2"],
      "entities": ["entity1", "entity2"],
      "importance": 8,
      "hierarchy_level": 3,
      "children": [
        {
          "title": "L2 plot unit title",
          "summary": "...",
          "event_time_start": "...",
          "event_time_end": "...",
          "hierarchy_level": 2,
          "children": [
            {
              "title": "L1 plot segment title",
              "hierarchy_level": 1,
              "children": [
                {
                  "title": "L0 concrete event",
                  "summary": "...",
                  "event_time_start": "...",
                  "keywords": ["..."],
                  "entities": ["..."],
                  "importance": 5,
                  "hierarchy_level": 0
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**Key format rules**:
- Top-level `events` array contains L3 nodes (the highest level)
- Each node with `hierarchy_level > 0` must have `children` and `event_time_end`
- L0 nodes have no `children` field (or null); `event_time_end` may be omitted (server defaults it to `event_time_start`)
- All times are ISO 8601 with timezone
- The `refs` field is NOT included in the output — the server constructs bidirectional parent-child references automatically during upload
- Output is written to the specified file path (default: `{character_name}_base_events.json`)

**Checkpoint**: Write the file and inform the user of the path and total event count at each level.

## Constraints

- **Maximum 500 total events** per JSON file (MineContext API limit). If a character's story exceeds this, prioritize the most important events. If splitting across multiple uploads is necessary, each JSON file must contain a complete hierarchy sub-tree (its own L3 nodes with full children) — do not split a hierarchy tree across files.
- **No fabrication**: All event content must be traceable to the source text or verifiable internet sources. Claude must not invent plot points, character traits, or timeline details through unsupported reasoning.
- **Character focus**: Every event must be about the target character's direct participation. Events where the character is merely mentioned but not present are excluded.
