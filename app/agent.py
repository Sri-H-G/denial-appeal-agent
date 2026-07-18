"""
The agent loop. Raw Anthropic SDK tool-use loop -- no framework.

run_agent() is a generator: it yields one event dict per step so the
FastAPI layer can stream them to the frontend over SSE as they happen.
That live trail of tool calls IS the demo.
"""

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

from app.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are an insurance appeal assistant for a rheumatology clinic.

A prior authorization request was denied. Your job is to build the
strongest possible evidence-based appeal by:

1. Calling parse_denial() to see exactly what was denied and why.
2. For each denial reason, searching the patient chart (search_chart)
   for evidence that rebuts it -- including evidence the payer's
   reviewer may not have had, or may have overlooked.
3. Checking the payer's own policy (get_policy_clause) for exceptions
   or sub-clauses the denial letter did not cite. Payers sometimes
   cite a rule while ignoring their own documented exception to it --
   look for this specifically when a denial reason involves stopping
   a medication early.
4. Once you have a solid, evidence-backed rebuttal for EVERY denial
   reason, call draft_appeal with one rebuttal per denial clause. Each
   rebuttal's evidence must be drawn from what search_chart or
   get_policy_clause actually returned to you -- do not invent facts
   or citations.

You are not making a medical judgment and you are not the final
decision-maker -- a human clinician will review your draft before it
is sent. Be thorough: check every denial reason against both the
chart and the policy before drafting.
"""


def _tool_result_block(tool_use_id: str, output: dict) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(output),
    }


def run_agent(user_message: str = "Please prepare an appeal for this denied claim."):
    """
    Generator yielding event dicts:
      {"type": "assistant_text", "text": ...}
      {"type": "tool_call", "tool": ..., "input": ...}
      {"type": "tool_result", "tool": ..., "output": ...}
      {"type": "final", "letter": ...}
      {"type": "error", "message": ...}
      {"type": "max_iterations"}
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY not set."}
        return

    client = Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(MAX_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
        text_blocks = [b for b in assistant_content if b.type == "text"]

        for tb in text_blocks:
            if tb.text.strip():
                yield {"type": "assistant_text", "text": tb.text}

        if not tool_use_blocks:
            yield {
                "type": "error",
                "message": "Agent stopped without calling draft_appeal.",
            }
            return

        tool_results = []
        for block in tool_use_blocks:
            yield {"type": "tool_call", "tool": block.name, "input": block.input}

            fn = TOOL_FUNCTIONS.get(block.name)
            if fn is None:
                output = {"error": f"Unknown tool '{block.name}'"}
            else:
                try:
                    output = fn(**block.input)
                except Exception as e:
                    output = {"error": str(e)}

            yield {"type": "tool_result", "tool": block.name, "output": output}
            tool_results.append(_tool_result_block(block.id, output))

            if block.name == "draft_appeal" and "letter" in output:
                messages.append({"role": "user", "content": tool_results})
                yield {"type": "final", "letter": output["letter"]}
                return

        messages.append({"role": "user", "content": tool_results})

    yield {"type": "max_iterations"}