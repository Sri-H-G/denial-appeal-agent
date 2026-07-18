"""
Setup sanity check ONLY. Not project code.
Confirms: venv is active, anthropic SDK is installed, .env loads,
and the API key is valid before tomorrow's build starts.

Run with:
    python hello_world.py
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit(
        "ANTHROPIC_API_KEY not found. Copy .env.example to .env and add your key."
    )

client = Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-sonnet-4-5",  # swap for whatever model the hackathon credits page recommends
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Reply with exactly: setup ok"}
    ],
)

print(response.content[0].text)
print("\nIf you see 'setup ok' above, the key, SDK, and env are all working.")
