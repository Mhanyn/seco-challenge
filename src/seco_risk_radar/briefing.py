"""
Natural-language inspector briefing.

This is the LLM component. Its job is narrow and honest: it TRANSLATES the
already-computed structured risk output (band, score, top driver factors) into
a short briefing an inspector can read. It does NOT make the prediction.

Design principle: the LLM is an enhancement layer, never on the critical path.
If no API key is configured, we generate a deterministic template briefing from
the same structured factors. The product is therefore fully functional and
reproducible offline; the LLM just makes the prose nicer. This is a deliberate
robustness/cost decision worth defending in the interview.

Providers tried (first available wins):
  * Anthropic   (ANTHROPIC_API_KEY)
  * OpenAI      (OPENAI_API_KEY)
  * Template    (always available, no network)
"""

from __future__ import annotations

import os

_FEATURE_LABELS = {
    "works_type": "works type",
    "structural_system": "structural system",
    "foundation_type": "foundation type",
    "season_started": "construction season",
    "contractor_experience": "contractor experience",
    "site_complexity": "site complexity",
    "existing_structure_age_yrs": "age of existing structure",
    "gross_floor_area_m2": "gross floor area",
    "estimated_cost_eur": "estimated cost",
    "num_floors": "number of floors",
    "canton": "canton",
    "region": "region",
    "building_type": "building type",
    "permit_processing_days": "permit processing time",
}


def _fmt_value(feature: str, value) -> str:
    if feature == "gross_floor_area_m2":
        return f"{int(value):,} m²"
    if feature == "estimated_cost_eur":
        return f"€{int(value):,}"
    if feature in ("contractor_experience", "site_complexity"):
        return f"{float(value):.2f}"
    return str(value)


def _template_briefing(project: dict, band: str, score: float, factors: list[dict]) -> str:
    bullets = []
    for f in factors:
        label = _FEATURE_LABELS.get(f["feature"], f["feature"])
        bullets.append(f"- {label}: {_fmt_value(f['feature'], f['value'])}")
    drivers = "\n".join(bullets) if bullets else "- no single dominant driver"
    return (
        f"Project {project.get('project_id', '?')} "
        f"({project.get('building_type', '?')}, {project.get('canton', '?')}) "
        f"is scored **{band} risk** (priority {score:.2f}).\n\n"
        f"Main risk drivers for this file:\n{drivers}\n\n"
        f"Suggested inspector focus: prioritise verification of the items above, "
        f"in particular the structural and works-type aspects, during the first visit."
    )


def _prompt(project: dict, band: str, score: float, factors: list[dict]) -> str:
    driver_lines = "; ".join(
        f"{_FEATURE_LABELS.get(f['feature'], f['feature'])}={_fmt_value(f['feature'], f['value'])}"
        for f in factors
    )
    return (
        "You are an assistant to a SECO technical-control inspector. In 4-6 "
        "sentences, write a concise, professional risk briefing for ONE "
        "construction project. Be specific, do not invent facts beyond those "
        "given, and end with concrete inspection focus areas.\n\n"
        f"Project: {project.get('project_id')} | type={project.get('building_type')} "
        f"| canton={project.get('canton')} | works={project.get('works_type')} "
        f"| structure={project.get('structural_system')}.\n"
        f"Model risk band: {band} (priority score {score:.2f}).\n"
        f"Top risk drivers: {driver_lines}."
    )


def _try_anthropic(prompt: str) -> str | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception:  # pragma: no cover - network/SDK issues fall back to template
        return None


def _try_openai(prompt: str) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception:  # pragma: no cover
        return None


def generate_briefing(
    project: dict,
    band: str,
    score: float,
    factors: list[dict],
    mode: str = "auto",
) -> tuple[str, str]:
    """Return (briefing_text, source) where source is anthropic/openai/template."""
    if mode != "off":
        prompt = _prompt(project, band, score, factors)
        text = _try_anthropic(prompt)
        if text:
            return text, "anthropic"
        text = _try_openai(prompt)
        if text:
            return text, "openai"
    return _template_briefing(project, band, score, factors), "template"
