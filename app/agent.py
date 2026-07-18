"""
The agent loop. Raw Anthropic SDK tool-use loop -- no framework.

run_agent() is a generator: it yields one event dict per step so the
FastAPI layer can stream them to the frontend over SSE as they happen.
That live trail of tool calls IS the demo; nothing here hides it.

Design patterns applied, per Anthropic's "Building Effective Agents":
- Augmented LLM: every tool call is the model actively using retrieval.
- Agent (not workflow): the model decides its own tool order/count --
  nothing here hardcodes a fixed call sequence.
- Evaluator-optimizer: the adversarial reviewer critiques the draft and
  the main agent revises based on that feedback, in a loop.
- Routing: drafting (generation, needs real judgment) runs on Sonnet;
  reviewing (evaluation against a checklist) runs on the cheaper, faster
  Haiku model.
- Parallelization (sectioning): independent tool calls in the same turn
  (e.g. two search_chart calls) run concurrently, not one after another.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic
from dotenv import load_dotenv

from app.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

load_dotenv()

MAX_ITERATIONS = 10
MAX_REVIEW_ROUNDS = 2
MODEL = "claude-sonnet-5"
REVIEWER_MODEL = "claude-haiku-4-5"  # routing: evaluation is cheaper than generation

SYSTEM_PROMPT = """You are an insurance appeal assistant for a rheumatology clinic.

A prior authorization request was denied. Your job is to build the
strongest possible evidence-based appeal by:

1. Calling parse_denial() to see exactly what was denied and why.
2. For each denial reason, searching the patient chart (search_chart)
   for evidence that rebuts it -- including evidence the payer's
   reviewer may not have had, or may have overlooked.
3. Checking the payer's own policy for exceptions or alternate
   approval pathways the denial letter did not cite. Payers sometimes
   cite one rule while ignoring their own documented exception or
   alternate criterion that the record actually satisfies -- look for
   this specifically. You have two tools for this, and you will be
   told in your task which one applies:
   - get_policy_clause: for policies with clean lettered clause IDs
     (e.g. "RX-114.2a"). Fetch by exact ID.
   - search_policy: for real-world policy documents with no clean IDs.
     Search by keyword/topic instead, and try a few different queries
     if the first doesn't surface the relevant criterion.
4. Once you have a solid, evidence-backed rebuttal for EVERY denial
   reason, call draft_appeal with one rebuttal per denial clause. Each
   rebuttal's evidence must be drawn from what your tool calls actually
   returned to you -- do not invent facts or citations.

Your draft will be reviewed by a skeptical payer-side reviewer. If it
is rejected, you will receive specific feedback -- read it carefully,
gather any additional evidence you need (you may call search_chart,
get_policy_clause, or search_policy again), and call draft_appeal
again with an improved set of rebuttals.

You are not making a medical judgment and you are not the final
decision-maker -- a human clinician will review your draft before it
is sent. Be thorough: check every denial reason against both the
chart and the policy before drafting.
"""

REVIEWER_SYSTEM_PROMPT = """You are a skeptical payer medical reviewer. You will
be shown a denied prior-authorization appeal letter. Your job is to find any
weakness in it: an unsupported claim, a citation that doesn't actually say what
the letter claims, or a denial reason the letter failed to address at all.

Respond with ONLY a JSON object, no other text:
{"verdict": "APPROVE", "feedback": ""}
or
{"verdict": "REJECT", "feedback": "<specific, actionable weakness to fix>"}

Be genuinely skeptical -- your job is to catch real problems, not rubber-stamp.
But do not invent problems that aren't there; if the letter is well-supported,
approve it.
"""


def _tool_result_block(tool_use_id: str, output: dict) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(output),
    }


def _run_tool(block):
    """
    Execute a single tool call. Used by the ThreadPoolExecutor so
    independent tool calls in the same turn run concurrently rather
    than one at a time -- the "sectioning" parallelization pattern.
    """
    fn = TOOL_FUNCTIONS.get(block.name)
    if fn is None:
        return block, {"error": f"Unknown tool '{block.name}'"}
    try:
        return block, fn(**block.input)
    except Exception as e:  # noqa: BLE001
        return block, {"error": str(e)}


def _review_appeal(client, letter: str, denial_reasons: list) -> dict:
    """
    One adversarial review pass. Returns {"verdict": ..., "feedback": ...}.
    Runs on REVIEWER_MODEL (Haiku) -- routing pattern: evaluating a draft
    against a checklist of denial reasons needs less capability than
    drafting the original evidence-based argument does.
    """
    prompt = (
        f"Denial reasons the appeal must address:\n{json.dumps(denial_reasons, indent=2)}\n\n"
        f"Appeal letter to review:\n{letter}"
    )
    response = client.messages.create(
        model=REVIEWER_MODEL,
        max_tokens=500,
        system=REVIEWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If the reviewer didn't return clean JSON, treat as approved
        # rather than looping forever on a parsing failure.
        return {"verdict": "APPROVE", "feedback": ""}


CASE_INSTRUCTIONS = {
    "synthetic": (
        "Please prepare an appeal for the denied claim. "
        "Call parse_denial and search_chart with case='synthetic' "
        "(or omit case, since 'synthetic' is the default). "
        "The applicable policy is the synthetic RX-114 policy, which "
        "uses clean lettered clause IDs like 'RX-114.2a' -- use "
        "get_policy_clause for this case, not search_policy."
    ),
    "uhc_real": (
        "Please prepare an appeal for the denied claim. "
        "Call parse_denial and search_chart with case='uhc_real'. "
        "The applicable policy is a real UnitedHealthcare document "
        "(policy_file='uhc_adalimumab_policy.txt') that does NOT use "
        "lettered clause IDs -- use search_policy with that policy_file "
        "and a keyword query to find relevant criteria instead of "
        "get_policy_clause. When drafting, also pass case='uhc_real' "
        "to draft_appeal."
    ),
}


def run_agent(case: str = "synthetic"):
    """
    Generator yielding event dicts:
      {"type": "assistant_text", "text": ...}
      {"type": "tool_call", "tool": ..., "input": ...}
      {"type": "tool_result", "tool": ..., "output": ...}
      {"type": "review_start", "round": ...}
      {"type": "review_result", "verdict": ..., "feedback": ...}
      {"type": "final", "letter": ...}
      {"type": "error", "message": ...}
      {"type": "max_iterations"}
    """
    if case not in CASE_INSTRUCTIONS:
        yield {"type": "error", "message": f"Unknown case '{case}'"}
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY not set."}
        return

    client = Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": CASE_INSTRUCTIONS[case]}]

    review_rounds_used = 0
    last_letter = None

    for iteration in range(MAX_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as e:  # noqa: BLE001 -- surface any API error to the demo UI
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
            if last_letter is not None:
                yield {"type": "final", "letter": last_letter}
                return
            yield {
                "type": "error",
                "message": "Agent stopped without calling draft_appeal.",
            }
            return

        tool_results = []
        should_return = False

        # Parallelization: run all independent tool calls from this turn
        # concurrently. executor.map preserves input order in its output,
        # so downstream logic (matching each block to its output) stays
        # deterministic even though execution itself is concurrent.
        with ThreadPoolExecutor(max_workers=4) as executor:
            for block, output in executor.map(_run_tool, tool_use_blocks):
                yield {"type": "tool_call", "tool": block.name, "input": block.input}
                yield {"type": "tool_result", "tool": block.name, "output": output}
                tool_results.append(_tool_result_block(block.id, output))

                if block.name == "draft_appeal" and "letter" in output:
                    last_letter = output["letter"]

                    if review_rounds_used >= MAX_REVIEW_ROUNDS:
                        messages.append({"role": "user", "content": tool_results})
                        yield {"type": "final", "letter": last_letter}
                        should_return = True
                        break

                    denial_reasons = TOOL_FUNCTIONS["parse_denial"](case=case)["denial_reasons"]
                    review_rounds_used += 1
                    yield {"type": "review_start", "round": review_rounds_used}

                    try:
                        review = _review_appeal(client, last_letter, denial_reasons)
                    except Exception as e:  # noqa: BLE001
                        yield {
                            "type": "review_result",
                            "verdict": "APPROVE",
                            "feedback": f"(review skipped: {e})",
                        }
                        messages.append({"role": "user", "content": tool_results})
                        yield {"type": "final", "letter": last_letter}
                        should_return = True
                        break

                    yield {
                        "type": "review_result",
                        "verdict": review.get("verdict", "APPROVE"),
                        "feedback": review.get("feedback", ""),
                    }

                    if review.get("verdict") == "APPROVE":
                        messages.append({"role": "user", "content": tool_results})
                        yield {"type": "final", "letter": last_letter}
                        should_return = True
                        break

                    # Rejected: fold feedback into this turn's results as an
                    # extra text block, then let the normal loop continue --
                    # the model may call any tool next (more evidence, or
                    # straight to draft_appeal again).
                    tool_results.append({
                        "type": "text",
                        "text": (
                            f"A payer medical reviewer rejected this draft: "
                            f"{review.get('feedback', '')} Please gather any "
                            f"additional evidence needed and call draft_appeal "
                            f"again with improved rebuttals."
                        ),
                    })

        if should_return:
            return

        messages.append({"role": "user", "content": tool_results})

    if last_letter is not None:
        yield {"type": "final", "letter": last_letter}
    else:
        yield {"type": "max_iterations"}