# Tally — Conversational Habit Tracker

![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[Live Demo](#) &nbsp;|&nbsp; [Features](#features) &nbsp;|&nbsp; [Quick Start](#how-to-run-the-project)

---

## About

Most habit trackers want you to open an app, find the right habit, tap a button, close
the app. I wanted something that felt more like texting a friend who was keeping score.
So I built Tally — a habit tracker you talk to.

Say "I ran 5km this morning" and it logs it. Say "how's my water habit this week?" and
it tells you — in a real human voice. No forms. No dropdowns. Just a conversation.

What made this genuinely interesting to build wasn't the habit tracking part. It was
making the AI reliable. I learned early that letting an AI freely decide what action to
take is a bad idea — it changes its mind based on how you word things. So I gave the AI
one job: understand what you said. Python handles the rest — a fixed flowchart that
decides what to actually do. That separation is the core architectural decision of the
project, and it's what made everything else predictable and testable.

## Tags

`voice-ai` `habit-tracker` `python` `fastapi` `nextjs` `websocket` `postgresql`
`pydantic-ai` `anthropic` `elevenlabs` `groq` `docker` `real-time` `state-machine`

## Technologies

| What | How |
|------|-----|
| Backend | Python 3.13, FastAPI |
| AI Agent | PydanticAI graph (state machine) |
| Language Model | Anthropic Claude (Sonnet + Haiku) |
| Speech → Text | Groq Whisper |
| Text → Speech | ElevenLabs Flash v2.5 |
| Voice Detection | Silero VAD (runs in your browser) |
| Database | PostgreSQL 17 |
| Frontend | Next.js 15, TypeScript |
| Auth | AuthKit (self-hosted JWT) |
| Deployed | Google Cloud Run (backend), Vercel (frontend) |

## Features

- **Talk or type** — voice and text go through the exact same agent pipeline
- **Understands natural language** — "I finished my run" works just as well as "log 5km running"
- **Talks back** — responses stream as audio in real time, token by token
- **Remembers context** — multi-turn conversations work: "remove the one from yesterday" is understood
- **Confirmation flow** — asks before creating or deleting; handles second thoughts gracefully
- **Full conversation history** — browse and continue past sessions
- **Smart deduplication** — warns you if a habit already exists before creating another one
- **Landing page** — marketing page at `/` with hero, live conversation demo, feature highlights

## Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Space` (hold) | Record voice |
| `Esc` | Cancel recording |

## The Process

**Where it started.** I wanted something that felt like a conversation, not a form.
The first version was simple: user sends text, Claude responds, Claude picks a tool.
That worked — until it didn't. One day it would log a run. The next day, the exact
same message would create a new habit instead. The AI was making routing decisions,
and LLMs don't make consistent routing decisions.

**The fix: separate understanding from control.** I replaced free-form tool calling
with a state machine — a Python flowchart where every intent (log, create, delete,
check progress) has a fixed path. The LLM now has one job: read your message and
extract what you meant. Python decides where that goes. This made the whole system
predictable, testable, and debuggable.

**Adding voice.** I wired in Groq Whisper for speech-to-text (same model as OpenAI,
10x faster, 9x cheaper) and ElevenLabs for text-to-speech. Early versions had an
audible stutter every few words. The cause was surprising: MP3 audio has ~26ms of
silence baked in per chunk by the encoder. Switching to raw PCM — uncompressed audio
samples — eliminated it completely. A tiny detail with a big effect on feel.

**Making voice feel instant.** The browser runs Silero VAD — a small neural network
that detects when you've stopped speaking without sending anything to the server.
No push-to-talk button. No awkward silence threshold. It just knows.

**The threading problem.** ElevenLabs' SDK is synchronous. FastAPI is async. I needed
to stream audio back while both were running at the same time. The solution was a
bridge: a thread-safe queue connects the sync ElevenLabs thread to the async FastAPI
WebSocket. It's a small piece of code, but it took real debugging to understand why
the two worlds couldn't communicate directly.

**Auth.** I used my own AuthKit project — a self-hosted JWT auth system I built
earlier. Connecting two of my own projects in production was a useful test of both.

## What I Learned

- **Separate understanding from routing.** LLMs are great at figuring out what you
  mean. They're unreliable at deciding what to do with it. A state machine for control
  flow, an LLM for language — this is the right division of labor.

- **Audio encoding details matter more than you'd think.** MP3 encoder delay is not
  something most tutorials mention. PCM is simpler and, for real-time streaming,
  noticeably better.

- **Async and sync Python don't naturally coexist.** Threading queues and
  `call_soon_threadsafe` are the bridge. Took real debugging to understand why you
  can't just `await` a synchronous iterator.

- **Multi-turn conversation design is genuinely hard.** Handling "yes", "no",
  "actually never mind", and "show my habits" all in the context of a pending
  confirmation required more thought than any single feature.

- **Good defaults beat smart defaults.** The confirmation flow that asks before every
  create or delete felt annoying at first. After testing with real messages, it turned
  out to be essential — natural language is ambiguous enough that a double-check is
  worth the extra step.

## How It Could Be Improved

- [ ] Habit streaks and visual progress charts
- [ ] Scheduled reminders ("remind me to log water at 3pm")
- [ ] Multi-user support (currently one user per deployed instance)
- [ ] Export habit data as CSV or JSON
- [ ] Cache classifier results for common phrases to reduce latency
- [ ] Open-source alternatives for STT/TTS to reduce API costs at scale

## How to Run the Project

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Node.js 20+](https://nodejs.org/)
- [Python 3.13+](https://python.org/) with [uv](https://docs.astral.sh/uv/)

### Backend

```bash
# Clone the repo
git clone https://github.com/juma-paul/tally
cd tally/habit-tracker

# Copy and fill in your API keys
cp .env.example .env

# Start PostgreSQL
docker compose up -d db

# Install Python dependencies and run
uv sync
uv run uvicorn app.main:app --reload --port 8001
```

### Frontend

```bash
cd tally-client

cp .env.example .env.local   # fill in AUTHKIT_URL, TALLY_URL, etc.
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Full stack with Docker (production-like)

```bash
cd tally/habit-tracker
cp .env.example .env  # fill in all values; set DATABASE_URL host to "db"
docker compose up --build
```

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|:--------:|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `AI_PROVIDER` | `anthropic` or `openai` | Yes |
| `ANTHROPIC_API_KEY` | Anthropic Claude | If using Anthropic |
| `ANTHROPIC_MODEL` | Model ID (default: `claude-sonnet-4-6`) | No |
| `OPENAI_API_KEY` | OpenAI | If using OpenAI |
| `STT_PROVIDER` | `groq`, `whisper`, or `elevenlabs` | Yes |
| `GROQ_API_KEY` | Groq Whisper STT | If using Groq |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS | Yes |
| `ELEVENLABS_VOICE_ID` | Which voice to use | Yes |
| `JWT_SECRET` | Shared secret with AuthKit | Yes |
| `ALLOWED_ORIGINS` | CORS — your frontend URL(s) | Yes |
| `ENVIRONMENT` | `development` or `production` | No |
| `RATE_LIMIT_PER_MINUTE` | API rate limit per client | No (default: 60) |

See `.env.example` for a complete template.

## Running Tests

```bash
# Requires a running PostgreSQL instance (docker compose up -d db)
uv run pytest tests/ -v
```

## CI / CD

Every push to `main` runs the full test suite automatically via GitHub Actions:

1. Starts a PostgreSQL 17 service
2. Runs `ruff` (lint) and `mypy` (types)
3. Applies `app/db/schema.sql` against the test database
4. Runs `pytest` with all tests

Deploying to Cloud Run is triggered by pushing a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The pipeline builds a Docker image, pushes it to Artifact Registry, and runs
`gcloud run deploy`. Required GitHub secrets: `GCP_SA_KEY`, `GCP_REGION`, `GCP_PROJECT_ID`.

## UI Design

The frontend uses a **White + Indigo** palette — white background, zinc-900 text,
indigo-500 (`#6366f1`) as the single accent. The "ll" in every Tally logo is indigo.
Voice waveform bars shift through indigo shades (500 → 300 → 200) across listening,
speaking, and processing states. Everything else is neutral.

## Video

> Demo coming soon — recording in progress.

---

<div align="center">

**Built with Python, FastAPI, and a lot of curiosity about how voice AI actually works.**

If you found this useful, consider giving it a star ⭐

</div>
