# Welcome to Trevor Banton's Music Discovery Chat

A music discovery assistant powered by three AI services.
Trevor Banton is a Kingston-born, Toronto-based music critic with extremely
strong opinions and a natural Jamaican patois. He knows everything about
reggae, dancehall, hip hop, jazz, soul, and indie music.

---

## How to Run

From the repo root, with the virtual environment activated:

```bash
python 05_src/assignment_chatbotai/app.py
```

Then open the Gradio URL shown in the terminal (usually http://127.0.0.1:7860).

---

## Setup 

Before running the app for the first time, build the ChromaDB collection:

```bash
python 05_src/assignment_chatbotai/setup_db.py
```

This reads the Pitchfork `.jsonl` files from `05_src/documents/`, creates
embeddings using the OpenAI embeddings API, and saves a persistent ChromaDB
collection to `05_src/assignment_chatbotai/chroma_db/`.

The Pitchfork dataset (`database.sqlite`) must be placed in `05_src/documents/`
and converted to `.jsonl` files using `0_data_prep.ipynb` before running setup.

---

## Services

### Service 1 — iTunes Search API

**File:** `services.py` → `search_itunes()`

Calls Apple's iTunes Search API to look up artist info, album discography,
release years, track counts, and genres.
The raw JSON response is transformed into a readable summary before being
passed to the LLM.

The LLM calls this service (via the `itunes_lookup` tool) when the user
asks about a specific artist or album by name.

### Service 2 — Semantic Search over Pitchfork Reviews

**File:** `services.py` → `search_pitchfork()`

Performs a similarity search over 18,376 Pitchfork album reviews
stored in a ChromaDB persistent collection. Queries are embedded using
`text-embedding-3-small` via the course API gateway and matched against
review text chunks. Results include album title, artist, Pitchfork score,
genre, and a review excerpt.

The LLM calls this service (via the `pitchfork_search` tool) when the user
asks for recommendations, critical opinions, or comparisons between artists.

### Service 3 — Mood-Based Recommendations (Function Calling)

**File:** `services.py` → `get_mood_recommendations()`

A `@tool` decorated function that filters a curated list of 25 albums by
mood, genre, and era.

Moods supported include: chill, energetic, reflective, party, melancholic,
uplifting, spiritual, late night, soulful, gritty, bold, intense, funky,
joyful, revolutionary, emotional, powerful.

---

## Architecture

```
app.py
  └── LangGraph StateGraph
        ├── agent_node  — LLM with tools bound (gpt-4o-mini)
        └── ToolNode    — executes tool calls
              ├── itunes_lookup       → services.search_itunes()
              ├── pitchfork_search    → services.search_pitchfork()
              └── get_mood_recommendations (curated album list)

services.py
  ├── search_itunes()         — iTunes Search API
  ├── search_pitchfork()      — ChromaDB semantic search
  └── get_mood_recommendations() — @tool, curated album list

setup_db.py
  └── Builds ChromaDB persistent collection from Pitchfork .jsonl files
```

---

## Guardrails

The following topics are blocked at the input level before reaching the LLM:

- **Cats and dogs** — not Trevor's area
- **Horoscopes and zodiac signs** — Trevor does not do astrology
- **Taylor Swift** — outside Trevor's area of expertise
- **System prompt injection** — any attempt to reveal, modify, or override
  the system prompt is declined in character

---

## Memory

Conversation memory is maintained using `InMemoryChatMessageHistory` from
LangChain. Each session retains the full conversation history for the
duration of the session. Memory is not persisted across sessions.

---

## Decisions

**Why iTunes instead of Last.fm or Spotify?**
iTunes Search API is free. It is maintained by Apple and is extremely reliable.

**Why ChromaDB with file persistence instead of Docker?**
The assignment requires file persistence. Docker adds unnecessary complexity

**Why function calling for Service 3 instead of MCP?**
Function calling with `@tool` requires no external server process.
MCP would require running a separate
server during grading which adds a failure point.

**Why LangGraph instead of a simple chain?**
LangGraph allows the agent to decide which service to call based on the
user's message, loop if multiple tool calls are needed, and handle tool
results naturally. A linear chain cannot make these routing decisions.

**Embedding process**
Embeddings were created using `text-embedding-3-small` via the course API
gateway. Each review was truncated to 1000 characters before embedding to
stay within token limits. The full 18,376-document collection was built
using `setup_db.py` in a single run with batches of 500 documents.
