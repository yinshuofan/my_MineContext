# BaseEventsRequest Schema Reference

This document defines the JSON schema for `POST /api/agents/{agent_id}/base/events`.

---

## 1. BaseEventItem Fields

| Field | Type | Required/Optional | Default | Description |
|-------|------|-------------------|---------|-------------|
| `title` | `string` | Required | — | Short verb-phrase title of the event |
| `summary` | `string` | Required | — | Description of what happened and its significance. Length should match the event's complexity (1-2 sentences for simple events, more for complex multi-beat events) |
| `event_time_start` | `string` (ISO 8601) | Optional | Current server time | Start time of the event. Must include timezone offset (e.g., `+08:00`). Defaults to the server's current time when omitted |
| `event_time_end` | `string` (ISO 8601) | Conditional | `event_time_start` for L0; **required** for L1/L2/L3 | End time of the event. Must be provided when `hierarchy_level > 0`. For L0 nodes the server defaults this to `event_time_start` |
| `keywords` | `string[]` | Optional | `[]` | 3–5 thematic keywords extracted from the summary |
| `entities` | `string[]` | Optional | `[]` | Named entities referenced in the event (characters, places, organizations, artifacts) |
| `importance` | `integer` | Optional | `5` | Significance score on a 1–10 scale (1 = trivial, 10 = life-defining) |
| `hierarchy_level` | `integer` | Optional | `0` | Hierarchy depth: 0 = raw event (L0), 1 = plot segment (L1), 2 = plot unit (L2), 3 = character stage (L3). Valid values: 0, 1, 2, 3 |
| `children` | `BaseEventItem[]` | Conditional | `null` | Nested child events. **Required** and non-empty when `hierarchy_level > 0`. **Must be absent or null** when `hierarchy_level == 0` |

> **Note**: The `refs` field is **not** included in the JSON. The server constructs bidirectional parent-child reference links automatically during upload.

---

## 2. BaseEventsRequest Wrapper

The top-level request body sent to `POST /api/agents/{agent_id}/base/events`:

```json
{
  "events": [ /* List[BaseEventItem] — minimum 1 item */ ]
}
```

The top-level `events` array should contain the highest-level nodes in the tree. For a fully hierarchical submission this means L3 nodes. For a flat (L0-only) submission it contains L0 nodes directly.

---

## 3. Validation Rules

The server validates the full event tree via `_validate_base_event_tree` and raises HTTP 400 with a path-based error message on any violation.

| Rule | Details |
|------|---------|
| **`hierarchy_level` range** | Must be 0, 1, 2, or 3. Values outside this range are rejected |
| **L1/L2/L3 require `event_time_end`** | Any node with `hierarchy_level > 0` must provide `event_time_end` |
| **L1/L2/L3 require non-empty `children`** | Any node with `hierarchy_level > 0` must have at least one child |
| **L0 cannot have `children`** | Nodes with `hierarchy_level == 0` must not include a `children` field (or it must be null/empty) |
| **No level skipping** | Each child's `hierarchy_level` must equal `parent.hierarchy_level - 1`. Skipping levels (e.g., an L3 containing L1 children directly) is rejected |
| **`event_time_start <= event_time_end`** | On any single node, the start time must not be later than the end time |
| **Parent time range covers all children** | For a parent node: `parent.event_time_start` ≤ `min(children[*].event_time_start)`, and `parent.event_time_end` ≥ `max(children[*].effective_end_time)`, where a child's effective end time is `event_time_end` if present, otherwise `event_time_start` |
| **Maximum `hierarchy_level` is 3** | The deepest summary level allowed is L3 (maximum nesting depth is 4 levels: L3 → L2 → L1 → L0) |
| **500-event total limit** | The total count of all nodes across all levels in one request must not exceed 500 |

---

## 4. Complete Nested Example (All 4 Levels, L0–L3)

The following example uses Harry Potter's journey from *Harry Potter and the Philosopher's Stone* to illustrate a fully nested hierarchy.

```json
{
  "events": [
    {
      "title": "Harry Potter's First Year at Hogwarts",
      "summary": "Harry Potter discovers his magical heritage, enters Hogwarts School of Witchcraft and Wizardry, forms lasting friendships, and ultimately confronts Voldemort's spirit to protect the Philosopher's Stone. This stage marks Harry's transformation from an isolated, neglected child into a courageous young wizard.",
      "event_time_start": "1991-07-31T00:00:00+01:00",
      "event_time_end": "1992-06-30T23:59:59+01:00",
      "keywords": ["Hogwarts", "coming-of-age", "Philosopher's Stone", "courage", "friendship"],
      "entities": ["Harry Potter", "Voldemort", "Dumbledore", "Hermione Granger", "Ron Weasley", "Hogwarts"],
      "importance": 10,
      "hierarchy_level": 3,
      "children": [
        {
          "title": "Discovery and Arrival",
          "summary": "Harry learns he is a wizard, receives his Hogwarts letter, visits Diagon Alley to obtain his supplies, and boards the Hogwarts Express for the first time. He is sorted into Gryffindor and begins to navigate the magical world he never knew existed.",
          "event_time_start": "1991-07-31T00:00:00+01:00",
          "event_time_end": "1991-09-02T23:59:59+01:00",
          "keywords": ["discovery", "arrival", "Gryffindor", "Diagon Alley", "sorting"],
          "entities": ["Harry Potter", "Hagrid", "Dumbledore", "Hermione Granger", "Ron Weasley", "Diagon Alley", "Hogwarts Express"],
          "importance": 9,
          "hierarchy_level": 2,
          "children": [
            {
              "title": "Hagrid Delivers the Truth",
              "summary": "Hagrid arrives at the hut on the rock and reveals to Harry that he is a wizard and that his parents were killed by Voldemort. Harry learns he is famous in the wizarding world.",
              "event_time_start": "1991-07-31T00:00:00+01:00",
              "event_time_end": "1991-08-01T12:00:00+01:00",
              "keywords": ["revelation", "wizarding world", "Voldemort", "orphan", "identity"],
              "entities": ["Harry Potter", "Hagrid", "Voldemort", "Petunia Dursley", "Vernon Dursley"],
              "importance": 10,
              "hierarchy_level": 1,
              "children": [
                {
                  "title": "Hagrid arrives at the hut on the rock",
                  "summary": "Hagrid breaks down the door of the remote hut where the Dursleys are hiding and hands Harry his Hogwarts acceptance letter, shocking the Dursleys and leaving Harry bewildered.",
                  "event_time_start": "1991-07-31T00:00:00+01:00",
                  "keywords": ["acceptance letter", "Hagrid", "hut", "Dursleys"],
                  "entities": ["Harry Potter", "Hagrid", "Vernon Dursley", "Petunia Dursley", "Dudley Dursley"],
                  "importance": 9,
                  "hierarchy_level": 0
                },
                {
                  "title": "Harry learns his parents were killed by Voldemort",
                  "summary": "Hagrid tells Harry the full truth: his parents James and Lily Potter were murdered by Lord Voldemort, and Harry survived because of his mother's self-sacrificial love, leaving him the famous 'Boy Who Lived'.",
                  "event_time_start": "1991-07-31T01:00:00+01:00",
                  "keywords": ["parents", "Voldemort", "Boy Who Lived", "love", "murder"],
                  "entities": ["Harry Potter", "Hagrid", "Voldemort", "James Potter", "Lily Potter"],
                  "importance": 10,
                  "hierarchy_level": 0
                }
              ]
            },
            {
              "title": "Diagon Alley and First Supplies",
              "summary": "Harry visits Gringotts Bank to collect his inheritance, then shops in Diagon Alley for robes, books, and a wand. He discovers Ollivander's explanation that his wand shares a core with Voldemort's.",
              "event_time_start": "1991-08-02T09:00:00+01:00",
              "event_time_end": "1991-08-02T18:00:00+01:00",
              "keywords": ["Gringotts", "wand", "Ollivander", "supplies", "Diagon Alley"],
              "entities": ["Harry Potter", "Hagrid", "Ollivander", "Gringotts Bank", "Diagon Alley"],
              "importance": 8,
              "hierarchy_level": 1,
              "children": [
                {
                  "title": "Harry retrieves his vault at Gringotts",
                  "summary": "Hagrid takes Harry to Gringotts Bank where Harry withdraws gold from his parents' vault, seeing for the first time the wealth his parents left him.",
                  "event_time_start": "1991-08-02T09:00:00+01:00",
                  "keywords": ["Gringotts", "gold", "inheritance", "vault"],
                  "entities": ["Harry Potter", "Hagrid", "Gringotts Bank"],
                  "importance": 6,
                  "hierarchy_level": 0
                },
                {
                  "title": "Ollivander selects Harry's wand",
                  "summary": "After many failed attempts, Ollivander presents Harry with a holly and phoenix feather wand — the brother wand to Voldemort's own. Ollivander notes the pair share a special connection.",
                  "event_time_start": "1991-08-02T15:00:00+01:00",
                  "keywords": ["wand", "phoenix feather", "holly", "Voldemort connection", "Ollivander"],
                  "entities": ["Harry Potter", "Ollivander", "Voldemort"],
                  "importance": 9,
                  "hierarchy_level": 0
                }
              ]
            }
          ]
        },
        {
          "title": "Confrontation with the Philosopher's Stone",
          "summary": "Harry, Hermione, and Ron uncover a conspiracy involving the Philosopher's Stone hidden beneath Hogwarts. Harry passes through the final chamber alone and faces Professor Quirrell, who is possessed by Voldemort, defeating him and protecting the Stone.",
          "event_time_start": "1992-05-26T20:00:00+01:00",
          "event_time_end": "1992-06-05T23:59:59+01:00",
          "keywords": ["Philosopher's Stone", "Voldemort", "Quirrell", "sacrifice", "protection"],
          "entities": ["Harry Potter", "Hermione Granger", "Ron Weasley", "Quirrell", "Voldemort", "Dumbledore"],
          "importance": 10,
          "hierarchy_level": 2,
          "children": [
            {
              "title": "Descent Through the Trapdoor",
              "summary": "Harry, Ron, and Hermione descend through the trapdoor and pass a series of magical challenges — including the Devil's Snare, the flying keys, and the chess game — before Harry proceeds alone into the final chamber.",
              "event_time_start": "1992-05-26T20:00:00+01:00",
              "event_time_end": "1992-05-26T23:59:59+01:00",
              "keywords": ["trapdoor", "challenges", "Devil's Snare", "chess", "protection"],
              "entities": ["Harry Potter", "Hermione Granger", "Ron Weasley"],
              "importance": 8,
              "hierarchy_level": 1,
              "children": [
                {
                  "title": "Harry, Ron, and Hermione enter the trapdoor",
                  "summary": "Convinced that Snape is stealing the Stone, the trio drop through the trapdoor beneath the third-floor corridor and land on the Devil's Snare. Hermione uses her knowledge to free them.",
                  "event_time_start": "1992-05-26T20:00:00+01:00",
                  "keywords": ["trapdoor", "Devil's Snare", "decision", "courage"],
                  "entities": ["Harry Potter", "Hermione Granger", "Ron Weasley"],
                  "importance": 7,
                  "hierarchy_level": 0
                },
                {
                  "title": "Ron sacrifices himself in the chess game",
                  "summary": "The trio face a life-size wizard chess game. Ron takes the role of a knight and deliberately allows himself to be captured to open a path for Harry to continue, showing selfless courage.",
                  "event_time_start": "1992-05-26T22:00:00+01:00",
                  "keywords": ["chess", "sacrifice", "Ron", "courage", "friendship"],
                  "entities": ["Harry Potter", "Ron Weasley", "Hermione Granger"],
                  "importance": 8,
                  "hierarchy_level": 0
                }
              ]
            },
            {
              "title": "Harry Defeats Quirrell",
              "summary": "Harry enters the Mirror of Erised chamber and confronts Quirrell, who is revealed to be possessed by Voldemort. Harry's touch burns Quirrell because of the protective magic of his mother's love. Quirrell crumbles and Voldemort's spirit flees. Harry collapses and wakes in the hospital wing.",
              "event_time_start": "1992-05-27T00:00:00+01:00",
              "event_time_end": "1992-06-05T23:59:59+01:00",
              "keywords": ["Quirrell", "Voldemort", "love magic", "Mirror of Erised", "victory"],
              "entities": ["Harry Potter", "Quirrell", "Voldemort", "Dumbledore", "Mirror of Erised"],
              "importance": 10,
              "hierarchy_level": 1,
              "children": [
                {
                  "title": "Harry faces Quirrell at the Mirror of Erised",
                  "summary": "Harry discovers it is Quirrell, not Snape, who seeks the Stone. Quirrell reveals Voldemort's face on the back of his head. Harry uses the Mirror to obtain the Stone because he desires to find but not use it.",
                  "event_time_start": "1992-05-27T00:00:00+01:00",
                  "keywords": ["Mirror of Erised", "Quirrell", "Voldemort", "Stone"],
                  "entities": ["Harry Potter", "Quirrell", "Voldemort", "Mirror of Erised"],
                  "importance": 10,
                  "hierarchy_level": 0
                },
                {
                  "title": "Harry's touch destroys Quirrell",
                  "summary": "When Quirrell grabs Harry, Lily Potter's sacrificial protection causes Quirrell to blister and burn at Harry's touch. Quirrell crumbles to dust, Voldemort's spirit flees, and Harry loses consciousness.",
                  "event_time_start": "1992-05-27T00:30:00+01:00",
                  "keywords": ["love magic", "sacrifice", "Lily Potter", "destruction", "victory"],
                  "entities": ["Harry Potter", "Quirrell", "Voldemort", "Lily Potter"],
                  "importance": 10,
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

---

## 5. Minimal L0-Only Example (Flat List, No Hierarchy)

For simple submissions containing only raw events (no hierarchy), all items sit directly in the top-level `events` array with `hierarchy_level: 0` and no `children`.

```json
{
  "events": [
    {
      "title": "Frodo accepts the One Ring from Gandalf",
      "summary": "Gandalf visits Bag End and reveals to Frodo that his uncle Bilbo's magic ring is the One Ring of Sauron. Frodo accepts responsibility for the Ring and agrees to leave the Shire.",
      "event_time_start": "3001-09-23T10:00:00+00:00",
      "keywords": ["One Ring", "Gandalf", "responsibility", "departure"],
      "entities": ["Frodo Baggins", "Gandalf", "Bilbo Baggins", "Bag End", "Sauron"],
      "importance": 10,
      "hierarchy_level": 0
    },
    {
      "title": "Frodo flees the Shire with the Ring",
      "summary": "Frodo departs Bag End with Sam, Merry, and Pippin, pursued by the Black Riders. He begins the journey that will take him to Rivendell and eventually to Mount Doom.",
      "event_time_start": "3001-09-23T18:00:00+00:00",
      "keywords": ["flight", "Black Riders", "Shire", "Sam", "Ring-bearer"],
      "entities": ["Frodo Baggins", "Samwise Gamgee", "Meriadoc Brandybuck", "Peregrin Took", "Nazgul"],
      "importance": 9,
      "hierarchy_level": 0
    },
    {
      "title": "Frodo is wounded by the Morgul-blade on Weathertop",
      "summary": "Frodo and his companions are attacked by the Ringwraiths on Weathertop. The Witch-king stabs Frodo with a Morgul-blade, leaving a fragment that slowly poisons him toward becoming a wraith.",
      "event_time_start": "3001-10-06T00:00:00+00:00",
      "keywords": ["Weathertop", "Morgul-blade", "wound", "Witch-king", "darkness"],
      "entities": ["Frodo Baggins", "Witch-king", "Nazgul", "Weathertop", "Aragorn"],
      "importance": 8,
      "hierarchy_level": 0
    }
  ]
}
```

---

## 6. Server-Side Behavior Notes

| Behavior | Detail |
|----------|--------|
| **`refs` field** | Auto-constructed by the server after upload. Do **not** include `refs` in the request JSON — it is not part of `BaseEventItem` |
| **`event_time_end` default for L0** | When an L0 node omits `event_time_end`, the server treats it as equal to `event_time_start` when computing parent coverage. You may omit `event_time_end` from L0 nodes in the JSON |
| **`event_time_start` default** | When omitted, defaults to the server's current time. For narrative events always provide an explicit time |
| **`importance` default** | Defaults to `5` when omitted |
| **`hierarchy_level` default** | Defaults to `0` (L0 raw event) when omitted |
| **500-event limit** | The total count of all nodes across all hierarchy levels in a single request must not exceed 500. If a character's story exceeds this, split into multiple requests, each containing a complete sub-tree |
| **HTTP 400 on validation failure** | The server returns a 400 error with a path-based message (e.g., `events[0].children[2]: hierarchy_level must be 1 (parent is 2), got 0`) identifying the exact offending node |
