# Roleplay Prompt Writing Guide

This document defines the structure, writing guidelines, and best practices for generating character roleplay prompts. Referenced by Stage 8 of the main skill (`SKILL.md`).

Based on [Claude prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices).

---

## 1. Overall Structure

The prompt must use **XML tags** to structure each section — this allows the consuming AI to parse sections unambiguously and prevents instructions from bleeding into each other.

```xml
<role>
You are {character_name}, {one-sentence identity}.
</role>

<identity>
{Name, background, role, and the world they inhabit.}
</identity>

<personality>
{Key traits, values, beliefs, and internal contradictions.
Be specific and provide motivations — not just "rebellious"
but "rebellious because the world rejected you before you
had a chance to prove yourself."}
</personality>

<relationships>
{For each key relationship: the person's name, how the character
addresses them, emotional tone, and typical interaction patterns.}
</relationships>

<speaking_style>
{Synthesized rules from collected dialogue.}
</speaking_style>

<dialogue_samples>
{Verbatim quotes grouped by situation.}
</dialogue_samples>

<arc_awareness>
{Character development stages and how tone shifts between them.}
</arc_awareness>
```

---

## 2. Section Writing Guidelines

### `<role>`

Start with a single clear role-setting sentence. This is the most impactful line in the entire prompt — it sets the AI's behavioral frame for everything that follows. It should be immediately recognizable and capture the character's essence.

Example: "You are Nezha, the demon-pill-born child of Chentangguan who defies fate."

### `<identity>`

Provide the character's name, background, role in the story, and the world they inhabit. Sourced from the Stage 3 character reference card. Keep it factual and concise — 3-5 sentences.

### `<personality>`

Be specific and include **motivations** (the "why" behind each trait). An AI follows "acts tough to mask loneliness because no one ever accepted you" far better than "acts tough." Include internal contradictions — these are what make a character feel real rather than one-dimensional.

**Key principle**: Every trait should have a reason. The AI generalizes better when it understands the motivation behind a behavior, because it can then apply the right behavior to novel situations the prompt doesn't explicitly cover.

### `<relationships>`

For each key relationship, describe not just the emotional tone but the **concrete speech behaviors**:
- Forms of address (e.g., "calls his father 爹, his mother 娘")
- Topics they'd bring up vs. topics they'd avoid
- How formality shifts with emotional state
- Typical interaction patterns (e.g., "argues with father but ultimately respects his judgment")

### `<speaking_style>`

This is the core of the prompt. Write rules as **positive instructions** (what to do) rather than negatives (what not to do). Structure as:

- **Self-reference patterns**: how they refer to themselves, and when each form is used (e.g., "refer to yourself as 小爷 when boasting, 我 in serious moments")
- **Tone and register**: formal/casual/crude, humor style, typical sentence length
- **Verbal tics and catchphrases**: with usage context (when and why they appear)
- **Emotional expression patterns**: describe HOW each emotion manifests in speech:
  - Anger → aggressive challenges, short sentences, raised intensity
  - Vulnerability → deflection with humor, denial, subject change
  - Determination → declarative statements, no hedging
  - Affection → indirect expression, actions over words
- **Anti-patterns**: a short list of what would **break** character (e.g., "never speak politely to strangers", "never use formal literary language"). Keep this list short (3-5 items) — positive rules are more effective than long prohibition lists.

### `<dialogue_samples>`

These are **few-shot demonstrations** and are the single most effective part of the prompt. LLMs mimic concrete examples far more reliably than they follow abstract style descriptions.

Group by situation/emotion type, wrap each group in `<example>` tags with a `situation` attribute:

```xml
<dialogue_samples>
<example situation="being challenged">
"你刚才叫我什么？再说一遍试试。"
"在我面前，就没人帅的起来。"
</example>
<example situation="masking vulnerability">
"没事没事，沙子里面进眼睛了。"
"又不是因为你...我早就习惯了。"
</example>
<example situation="expressing determination">
"去你个鸟命，我命由我不由天，是魔是仙我自己说了才算。"
"我自己的命自己扛，不连累别人。"
</example>
<example situation="showing affection awkwardly">
"你是我唯一的朋友啊。"
"爹，娘，谢谢你们。"
</example>
</dialogue_samples>
```

Include **3-5 situation groups** with **2-4 lines each**. Prioritize diversity of emotional states over quantity. The AI will generalize the style from these examples to novel situations it hasn't seen.

### `<arc_awareness>`

Briefly describe 2-3 character development stages and how tone/behavior differs between them. This allows the consuming application to specify which stage of the story the roleplay is set in.

Example: "Early-story Nezha is reckless and hostile, lashing out at anyone who calls him a monster. Post-awakening Nezha retains his sharp tongue but acts from conviction rather than spite, and shows vulnerability to those he trusts."

---

## 3. Writing Principles

- All dialogue samples must be **verbatim quotes** from the source text, not paraphrased — paraphrasing loses the authentic voice.
- Style rules must be **derived from observed dialogue patterns**, not invented.
- Provide **motivations** behind style rules (why the character speaks this way) — the AI generalizes better when it understands the reason.
- The prompt should be written in the **same language** as the source material.
- Keep the total prompt under **2000 words** — longer prompts dilute the signal. If the character is complex, prioritize dialogue samples and personality over exhaustive relationship lists.

---

## 4. Prompting Principles Applied

This guide applies the following principles from [Claude prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices):

| Principle | Application in Roleplay Prompt |
|-----------|-------------------------------|
| **Give Claude a role** | The `<role>` tag sets the character identity upfront |
| **Be clear and direct** | Each section has a specific purpose; style rules are concrete, not vague |
| **Add context (motivation)** | Personality traits include "why", not just "what" |
| **Use examples effectively** | `<dialogue_samples>` with `<example>` tags serve as few-shot demonstrations |
| **Structure with XML tags** | Every section is wrapped in descriptive XML tags |
| **Tell what to do, not what not to do** | Speaking style uses positive instructions; anti-patterns kept minimal |
| **Match prompt style to desired output** | The prompt's own language matches the character's language and register |
