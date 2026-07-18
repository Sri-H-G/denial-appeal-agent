# Denial Appeal Agent

Hackathon: Abridge x Anthropic x Lightspeed, SF — built live, single day.

## What it does

A claim gets denied. The agent reads the denial letter, the patient chart,
and the payer's policy, finds the scattered evidence that defeats each
denial reason, and drafts a cited appeal letter. A human reviews and sends
it — the agent never makes an autonomous medical judgment.

## Demo case

Synthetic RA patient (Jane Doe) denied adalimumab. Meridian Health Plan's
denial cites two failures:

1. **RX-114.2** — no completed 3-month conventional DMARD trial.
2. **RX-114.3** — no disease activity score on file.

Both are wrong, but the evidence defeating them is scattered across three
documents, not summarized anywhere:

- Methotrexate was stopped at 14 weeks for documented hepatotoxicity
  (ALT 96, AST 88) — which triggers policy exception **RX-114.2a**
  (intolerance exception), a subclause the denial letter never mentions.
- Leflunomide was then run a full 3 consecutive months at therapeutic
  dose with an explicit "inadequate clinical response" note.
- A CDAI score of 28.4 (high disease activity, threshold is >10.0) was
  documented in June — within the 90-day window — and simply wasn't in
  what got submitted or reviewed.

The interesting part isn't reading comprehension, it's letting the model
trace a policy subclause exception across a chart the reviewer plainly
didn't read closely, and showing that trail live.

## Architecture

Raw Anthropic SDK tool-use loop. No LangChain, no framework, no RAG
pipeline pretending to be an agent.

**Tools (four):**
- `parse_denial()` — extract structured denial reasons + cited policy IDs
- `search_chart(query)` — pull relevant spans from the patient chart
- `get_policy_clause(id)` — fetch a specific policy clause/subclause by ID
- `draft_appeal(rebuttals)` — compose the final cited appeal letter

Loop capped at ~10 iterations. Every tool call — inputs and outputs — is
logged and rendered live in the frontend. That visible reasoning trail
*is* the demo, not a dashboard bolted on afterward.

**Stack:** Python, FastAPI backend, Anthropic SDK, plain HTML/JS frontend
(no framework). Sample data as flat `.txt` files — no vector DB, no
embeddings, no basic-RAG shortcut.

## Build order

1. Core tool-use loop working end to end, on the demo case, first.
2. Only after that works:
   - Adversarial reviewer agent that re-denies the appeal and forces a
     revision.
   - Citation verification against source spans (no unverified claims
     in the letter).
   - Appeal strength scoring.

## Explicitly not doing

- Streamlit
- Basic RAG as the core mechanism
- Dashboard-as-main-feature
- A chatbot wrapper around this

## Setup (done night-before, per hackathon rules — no project code yet)

```bash
git init
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in ANTHROPIC_API_KEY
python hello_world.py           # sanity check only, not project code
```

## Sample data

`sample_data/` contains the three synthetic source documents for the
demo case:
- `denial_letter.txt`
- `policy_rx114.txt`
- `patient_chart.txt`

All marked as synthetic, for demonstration purposes only.

## Day-of reminders

- Check "Partner Provided Resources" at kickoff for Anthropic API credits.
- Grab an Abridge clinician in the first hour to validate the pain point.
