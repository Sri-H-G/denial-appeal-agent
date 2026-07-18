"""
Four tools the agent can call. Each is a plain Python function over the
flat sample_data/*.txt files -- no vector DB, no embeddings. Parsing is
regex/structure-based on purpose: these documents have predictable
headers (dated note sections, numbered policy clauses), so we exploit
that structure directly rather than pretending we need semantic search
for a demo with three documents.
"""

import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"

# Each "case" is a self-contained denial scenario: its own denial letter
# and chart. Adding a new case means adding an entry here plus two files
# -- nothing else in this module needs to change.
CASES = {
    "synthetic": {
        "denial": "denial_letter.txt",
        "chart": "patient_chart.txt",
    },
    "uhc_real": {
        "denial": "real_world/uhc_denial_letter.txt",
        "chart": "real_world/uhc_patient_chart.txt",
    },
}


def _read(name: str) -> str:
    return (DATA_DIR / name).read_text()


# ---------------------------------------------------------------------
# Tool 1: parse_denial
# ---------------------------------------------------------------------
def parse_denial(case: str = "synthetic") -> dict:
    """
    Extract structured denial reasons from the denial letter for the
    given case: each cited policy criterion ID plus the rationale
    paragraph the payer gave for it. This is the agent's starting
    point -- it defines what has to be rebutted.
    """
    denial_file = CASES[case]["denial"]
    text = _read(denial_file)

    determination_match = re.search(r"Determination:\s*(.+)", text)
    determination = determination_match.group(1).strip() if determination_match else None

    # Denial reasons are formatted as "Criterion <ID> - Title" followed
    # by a rationale paragraph. The ID pattern is intentionally generic
    # (word characters, dots, hyphens, parens) so it matches both the
    # synthetic RX-114.x style and a plausible real-payer style like
    # RA-2(a) -- we don't want this hardcoded to one insurer's format.
    reason_pattern = re.compile(
        r"Criterion\s+([\w.\-()]+)\s*-\s*(.+?)\n(.*?)(?=\nCriterion\s+[\w.\-()]+\s*-|\nThis determination was made by|\nYou may request a peer-to-peer)",
        re.DOTALL,
    )

    reasons = []
    for m in reason_pattern.finditer(text):
        clause_id, title, rationale = m.groups()
        reasons.append(
            {
                "clause_id": clause_id.strip(),
                "title": title.strip(),
                "rationale": " ".join(rationale.split()),
            }
        )

    ref_match = re.search(r"Reference Number:\s*(.+)", text)

    return {
        "determination": determination,
        "reference_number": ref_match.group(1).strip() if ref_match else None,
        "denial_reasons": reasons,
    }


# ---------------------------------------------------------------------
# Tool 2: search_chart
# ---------------------------------------------------------------------
def _chart_sections(case: str = "synthetic") -> list[dict]:
    """Split the chart into its dated note sections."""
    chart_file = CASES[case]["chart"]
    text = _read(chart_file)
    # Sections are delimited by a dashed line, a date, and a header line,
    # then another dashed line.
    pattern = re.compile(
        r"-{5,}\n(\d{2}/\d{2}/\d{4}) - (.+?) - (.+?)\n-{5,}\n(.*?)(?=\n-{5,}\n\d{2}/\d{2}/\d{4}|\n--- SYNTHETIC)",
        re.DOTALL,
    )
    sections = []
    for m in pattern.finditer(text):
        date, header, provider, body = m.groups()
        sections.append(
            {
                "date": date,
                "header": header.strip(),
                "provider": provider.strip(),
                "text": " ".join(body.split()),
            }
        )
    return sections


def search_chart(query: str, case: str = "synthetic") -> dict:
    """
    Keyword-overlap search over chart note sections. Returns the
    top-scoring sections with their date, so the agent can build a
    timeline of evidence rather than getting one flat blob of chart text.
    """
    sections = _chart_sections(case=case)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

    scored = []
    for s in sections:
        section_terms = re.findall(r"[a-z0-9]+", (s["header"] + " " + s["text"]).lower())
        section_term_set = set(section_terms)
        overlap = query_terms & section_term_set
        if overlap:
            scored.append((len(overlap), s))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = [s for _, s in scored[:5]]
    return {"query": query, "matches": results}


# ---------------------------------------------------------------------
# Tool 3: get_policy_clause
# ---------------------------------------------------------------------
def _policy_clauses() -> dict:
    """
    Line-based parser. A header line is one that STARTS (after stripping
    leading whitespace) with a clause ID immediately followed by an
    all-caps title on the same line, e.g. "RX-114.2b  EXCEPTION -
    INTOLERANCE". Body lines are everything until the next header line
    or a top-level roman-numeral section (III., IV., etc).

    A regex-only approach over the whole text is fragile here: clause
    IDs also appear inline inside wrapped body text (e.g. "...satisfies
    RX-114.2." wrapped onto its own line), which a lookahead-based
    single regex can mistake for the next header. Parsing line by line
    avoids that ambiguity.
    """
    text = _read("policy_rx114.txt")
    header_re = re.compile(r"^\s*(RX-114\.\d[a-z]?)\s{2,}([A-Z][A-Z0-9 ()\-]+)\s*$")
    section_break_re = re.compile(r"^(III|IV|V|VI)\.\s")

    clauses = {}
    current_id = None
    current_title = None
    current_body_lines: list[str] = []

    def flush():
        if current_id is not None:
            clauses[current_id] = {
                "clause_id": current_id,
                "title": current_title.strip(),
                "text": " ".join(" ".join(current_body_lines).split()),
            }

    for line in text.splitlines():
        header_match = header_re.match(line)
        if header_match:
            flush()
            current_id, current_title = header_match.groups()
            current_body_lines = []
            continue
        if section_break_re.match(line.strip()):
            flush()
            current_id = None
            current_body_lines = []
            continue
        if current_id is not None:
            current_body_lines.append(line)

    flush()
    return clauses


def get_policy_clause(clause_id: str) -> dict:
    """
    Fetch a specific policy clause or sub-clause by ID, e.g. 'RX-114.2'
    or 'RX-114.2a'. This is how the agent finds exceptions the denial
    letter didn't cite -- it has to go looking for them.
    """
    clauses = _policy_clauses()
    match = clauses.get(clause_id.strip())
    if match:
        return match
    return {
        "error": f"Clause '{clause_id}' not found.",
        "available_clause_ids": sorted(clauses.keys()),
    }


# ---------------------------------------------------------------------
# Tool 4: draft_appeal
# ---------------------------------------------------------------------
def draft_appeal(rebuttals: list[dict], case: str = "synthetic") -> dict:
    """
    Compose the final appeal letter from a list of rebuttals. Each
    rebuttal must be a dict with:
        denial_clause_id: the criterion being rebutted (e.g. "RX-114.2")
        argument: the agent's rebuttal argument, in its own words
        evidence: list of short quotes/spans pulled from chart/policy
                  tool results, each with a `source` and `text` field

    This tool is intentionally deterministic templating, not another
    model call -- the agent has already done the reasoning; this just
    assembles it into letter format. Keeps the loop auditable: every
    sentence in the output traces back to a specific prior tool result.
    """
    denial = parse_denial(case=case)

    lines = []
    lines.append("APPEAL OF ADVERSE BENEFIT DETERMINATION")
    lines.append(f"Reference Number: {denial.get('reference_number', '[unknown]')}")
    lines.append("")
    lines.append(
        "This letter appeals the denial of the above-referenced prior "
        "authorization request. Each cited basis for denial is addressed "
        "below with supporting documentation already contained in the "
        "member's medical record."
    )
    lines.append("")

    for i, r in enumerate(rebuttals, start=1):
        lines.append(f"{i}. Regarding {r.get('denial_clause_id', '[clause]')}:")
        lines.append(r.get("argument", "[no argument provided]"))
        for ev in r.get("evidence", []):
            lines.append(f"   - [{ev.get('source', 'source')}] {ev.get('text', '')}")
        lines.append("")

    lines.append(
        "Based on the documentation above, the requested service meets "
        "the applicable coverage criteria and the denial should be "
        "reversed. We respectfully request approval of this appeal."
    )

    letter = "\n".join(lines)
    return {"letter": letter, "rebuttal_count": len(rebuttals)}


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> list[str]:
    """
    Generic sliding-window chunker. Makes no assumptions about document
    structure -- no clause IDs, no headers, no blank-line paragraphs
    required. This is what lets the same search technique work on a
    clean synthetic policy AND a messy real-world payer PDF.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def search_policy(policy_file: str, query: str) -> dict:
    """
    Keyword-overlap search over ANY policy document, chunked into
    overlapping windows. Use this for real-world payer documents that
    don't have clean clause IDs like the synthetic RX-114 policy does.

    policy_file: filename under sample_data/ or sample_data/real_world/
    """
    candidate_paths = [DATA_DIR / policy_file, DATA_DIR / "real_world" / policy_file]
    text = None
    for path in candidate_paths:
        if path.exists():
            text = path.read_text()
            break
    if text is None:
        return {"error": f"Policy file '{policy_file}' not found."}

    chunks = _chunk_text(text)
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))

    scored = []
    for chunk in chunks:
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.lower()))
        overlap_count = len(query_terms & chunk_terms)
        if overlap_count:
            scored.append((overlap_count, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_chunks = [" ".join(c.split()) for _, c in scored[:3]]
    return {"policy_file": policy_file, "query": query, "matches": top_chunks}


# ---------------------------------------------------------------------
# Tool schemas for the Anthropic API
# ---------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "parse_denial",
        "description": (
            "Extract the structured denial reasons (policy clause IDs and "
            "rationale) from the denial letter. Call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "case": {
                    "type": "string",
                    "description": "Which case to load: 'synthetic' (default) or 'uhc_real'.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "search_chart",
        "description": (
            "Search the patient's chart for note sections relevant to a "
            "query (e.g. 'methotrexate hepatotoxicity', 'CDAI disease "
            "activity'). Returns matching dated note sections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "case": {
                    "type": "string",
                    "description": "Which case to load: 'synthetic' (default) or 'uhc_real'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_policy_clause",
        "description": (
            "Fetch the exact text of a policy clause or sub-clause by ID, "
            "e.g. 'RX-114.2' or 'RX-114.2a'. ONLY use this for the synthetic "
            "RX-114 policy, which has clean lettered clause IDs. Do not use "
            "this for real-world payer documents -- use search_policy instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string", "description": "e.g. 'RX-114.2a'"}
            },
            "required": ["clause_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search a policy document by keyword when it does NOT use "
            "clean clause IDs (e.g. real-world payer PDFs like "
            "'uhc_adalimumab_policy.txt'). Use get_policy_clause instead "
            "for the synthetic RX-114 policy, which has clean IDs. "
            "policy_file must be a filename under sample_data/ or "
            "sample_data/real_world/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "policy_file": {"type": "string", "description": "e.g. 'uhc_adalimumab_policy.txt'"},
                "query": {"type": "string", "description": "Search terms"},
            },
            "required": ["policy_file", "query"],
        },
    },
    {
        "name": "draft_appeal",
        "description": (
            "Compose the final appeal letter. Call this last, once you have "
            "gathered evidence and a rebuttal for every denial reason. "
            "Provide one rebuttal object per denial clause."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rebuttals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "denial_clause_id": {"type": "string"},
                            "argument": {"type": "string"},
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "text": {"type": "string"},
                                    },
                                    "required": ["source", "text"],
                                },
                            },
                        },
                        "required": ["denial_clause_id", "argument", "evidence"],
                    },
                },
                "case": {
                    "type": "string",
                    "description": "Which case to load: 'synthetic' (default) or 'uhc_real'.",
                },
            },
            "required": ["rebuttals"],
        },
    },
]

TOOL_FUNCTIONS = {
    "parse_denial": lambda **kwargs: parse_denial(**kwargs),
    "search_chart": lambda **kwargs: search_chart(**kwargs),
    "get_policy_clause": lambda **kwargs: get_policy_clause(**kwargs),
    "search_policy": lambda **kwargs: search_policy(**kwargs),
    "draft_appeal": lambda **kwargs: draft_appeal(**kwargs),
}