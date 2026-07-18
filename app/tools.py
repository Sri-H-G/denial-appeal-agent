"""
Four tools the agent can call. Each is a plain Python function over the
flat sample_data/*.txt files -- no vector DB, no embeddings.
"""

import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _read(name: str) -> str:
    return (DATA_DIR / name).read_text()

def parse_denial() -> dict:
    """
    Extract structured denial reasons from the denial letter: each
    cited policy criterion ID plus the rationale paragraph the payer
    gave for it. This is the agent's starting point.
    """
    text = _read("denial_letter.txt")

    determination_match = re.search(r"Determination:\s*(.+)", text)
    determination = determination_match.group(1).strip() if determination_match else None

    reason_pattern = re.compile(
        r"Criterion\s+(RX-[\d.]+[a-z]?)\s*-\s*(.+?)\n(.*?)(?=\nCriterion\s+RX-|\nThis determination was made by)",
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

def _chart_sections() -> list[dict]:
    """Split the chart into its dated note sections."""
    text = _read("patient_chart.txt")
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


def search_chart(query: str) -> dict:
    """
    Keyword-overlap search over chart note sections. Returns the
    top-scoring sections with their date.
    """
    sections = _chart_sections()
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

def _policy_clauses() -> dict:
    """
    Line-by-line parser: a header line starts with a clause ID
    immediately followed by an ALL-CAPS title, e.g.
    "RX-114.2b  EXCEPTION - INTOLERANCE". Everything else is body
    text belonging to whichever clause we're currently inside.
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
    or 'RX-114.2a'.
    """
    clauses = _policy_clauses()
    match = clauses.get(clause_id.strip())
    if match:
        return match
    return {
        "error": f"Clause '{clause_id}' not found.",
        "available_clause_ids": sorted(clauses.keys()),
    }

def draft_appeal(rebuttals: list[dict]) -> dict:
    """
    Compose the final appeal letter from a list of rebuttals. Each
    rebuttal is a dict with:
        denial_clause_id: the criterion being rebutted (e.g. "RX-114.2")
        argument: the agent's rebuttal argument, in its own words
        evidence: list of {source, text} pulled from prior tool results
    """
    denial = parse_denial()

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

TOOL_SCHEMAS = [
    {
        "name": "parse_denial",
        "description": (
            "Extract the structured denial reasons (policy clause IDs and "
            "rationale) from the denial letter. Call this first."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
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
                "query": {"type": "string", "description": "Search terms"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_policy_clause",
        "description": (
            "Fetch the exact text of a policy clause or sub-clause by ID, "
            "e.g. 'RX-114.2' or 'RX-114.2a'. Use this to check for "
            "exceptions the denial letter did not cite."
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
                }
            },
            "required": ["rebuttals"],
        },
    },
]

TOOL_FUNCTIONS = {
    "parse_denial": lambda **kwargs: parse_denial(),
    "search_chart": lambda **kwargs: search_chart(**kwargs),
    "get_policy_clause": lambda **kwargs: get_policy_clause(**kwargs),
    "draft_appeal": lambda **kwargs: draft_appeal(**kwargs),
}