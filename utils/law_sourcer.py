"""
utils/law_sourcer.py

Programmatically sources physics laws from Wikipedia, then uses an LLM to
extract structured metadata suitable for NewtonBench module creation.

Pipeline:
  1. Collect candidate law titles from two Wikipedia sources:
       a. Links within "List of scientific laws named after people"
       b. Members of Category:Scientific_laws
  2. Filter candidates by name keywords (law, principle, theorem, effect, …)
  3. For each candidate, fetch the plain-text article intro via MediaWiki API
  4. Pass the intro to an LLM to extract: formula, variables, domain, benchmark suitability
  5. Keep only physics laws; flag ones already in NewtonBench

Usage (Python API):
    from utils.law_sourcer import source_physics_laws
    laws = source_physics_laws(max_laws=30)

Usage (see source_laws.py for the CLI).
"""

import json
import time
import requests
from typing import Optional
from utils.call_llm_api import call_llm_api

# ── Wikipedia API ──────────────────────────────────────────────────────────────

_MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "NewtonBench/1.0 (academic research; https://github.com/NewtonBench)"}

# Primary source: links within Wikipedia's list article
_LIST_ARTICLE = "List of scientific laws named after people"

# Supplementary source: direct category members (also exposed for CLI use)
PHYSICS_CATEGORIES = ["Scientific_laws", "Equations_of_physics"]
_SUPPLEMENTARY_CATEGORIES = PHYSICS_CATEGORIES

# Title keywords used to pre-filter candidates before hitting the LLM
_LAW_KEYWORDS = ["law", "principle", "theorem", "equation", "formula", "effect", "rule", "constant"]

# Laws already implemented in NewtonBench (for duplicate flagging)
_EXISTING_MODULES = {
    "newton's law of universal gravitation",
    "newton's law of gravitation",
    "law of universal gravitation",
    "coulomb's law",
    "magnetic force",
    "lorentz force",
    "fourier's law",
    "fourier's law of heat conduction",
    "snell's law",
    "radioactive decay",
    "harmonic oscillator",
    "malus's law",
    "speed of sound",
    "hooke's law",
    "bose-einstein distribution",
    "bose-einstein statistics",
    "newton's law of cooling",
}

# ── LLM prompts ────────────────────────────────────────────────────────────────

_SYSTEM = """You are an expert physicist and scientific benchmark designer.
Your task is to analyze Wikipedia text about a scientific law and extract structured
metadata to determine if the law is suitable for a physics discovery benchmark."""

_USER = """\
Analyze the following Wikipedia text about "{law_name}" and return a single JSON object.

Wikipedia text:
\"\"\"
{wiki_text}
\"\"\"

Return ONLY a valid JSON object with this exact schema — no extra text, no markdown fences:
{{
  "name": "<canonical name of the law>",
  "domain": "<physics subdomain: one of 'classical mechanics', 'electromagnetism', 'thermodynamics', 'optics', 'fluid dynamics', 'nuclear physics', 'astrophysics', 'quantum mechanics', 'solid state physics', 'waves and acoustics', 'other physics'>",
  "formula_text": "<the core formula in plain text, e.g. 'F = G * m1 * m2 / r^2', or empty string if none>",
  "is_physics": <true if this is a physics law; false if chemistry, biology, economics, statistics, pure mathematics, computer science, etc.>,
  "is_benchmarkable": <true only if ALL of: (a) output = f(inputs) form, (b) output is a single real number scalar, (c) 2 to 5 numeric input variables, (d) NOT a conserved-quantity equality like P1V1=P2V2>,
  "not_benchmarkable_reason": "<if not benchmarkable, brief reason; else empty string>",
  "variables": [
    {{
      "name": "<Python-safe parameter name, e.g. 'mass1'>",
      "symbol": "<math symbol used in formula, e.g. 'm1'>",
      "description": "<plain English description>",
      "constraint": "<valid input range, e.g. 'positive real number', 'real number in [0, pi/2]'>"
    }}
  ],
  "output_description": "<what the function returns, e.g. 'gravitational force magnitude in Newtons'>",
  "function_signature": "<Python def line, e.g. 'def discovered_law(mass1, mass2, distance):'>",
  "param_description": "<multi-line string: one dash-prefixed line per parameter, matching NewtonBench PARAM_DESCRIPTION style>"
}}

Strict rules:
- is_benchmarkable must be false if the law is a ratio/proportionality equality (P1V1=P2V2 style)
- is_benchmarkable must be false if the output is a vector, matrix, or distribution
- is_benchmarkable must be false if any input is categorical (material type, substance name, etc.)
- is_benchmarkable must be false if it requires integrating over a distribution or trajectory
- variables, function_signature, param_description should be empty / empty string if not benchmarkable
- domain must be a physics subdomain; mark is_physics=false for chemistry, biology, economics, etc."""


# ── Wikipedia helpers ──────────────────────────────────────────────────────────

def _get_list_article_links(max_chars_title: int = 80) -> list:
    """Return law-like titles from the Wikipedia list-of-scientific-laws article."""
    all_links: list = []
    params = {
        "action": "query",
        "titles": _LIST_ARTICLE,
        "prop": "links",
        "pllimit": 500,
        "plnamespace": 0,
        "format": "json",
    }
    while True:
        try:
            resp = requests.get(_MEDIAWIKI_API, params=params, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[LawSourcer] Failed fetching list-article links: {e}")
            break

        for page in data.get("query", {}).get("pages", {}).values():
            all_links.extend(l["title"] for l in page.get("links", []))

        cont = data.get("continue", {}).get("plcontinue")
        if not cont:
            break
        params["plcontinue"] = cont

    # Keep titles that look like a law/principle (not person names, etc.)
    return [
        t for t in all_links
        if len(t) <= max_chars_title
        and any(kw in t.lower() for kw in _LAW_KEYWORDS)
        and not any(skip in t for skip in ("(disambiguation)", "List of", "Index of"))
    ]


def _get_category_members(category: str) -> list:
    """Return article titles from a Wikipedia category."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 200,
        "cmnamespace": 0,
        "format": "json",
    }
    try:
        resp = requests.get(_MEDIAWIKI_API, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return [m["title"] for m in resp.json().get("query", {}).get("categorymembers", [])]
    except Exception as e:
        print(f"[LawSourcer] Failed fetching category '{category}': {e}")
        return []


def _fetch_article_intro(title: str, max_chars: int = 3000) -> str:
    """Fetch plain-text introduction of a Wikipedia article."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }
    try:
        resp = requests.get(_MEDIAWIKI_API, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")[:max_chars]
        return ""
    except Exception as e:
        print(f"[LawSourcer] Failed fetching article '{title}': {e}")
        return ""


# ── LLM extraction ─────────────────────────────────────────────────────────────

def _extract_law_metadata(law_name: str, wiki_text: str, model: str) -> Optional[dict]:
    """Use an LLM to extract structured metadata from a Wikipedia article intro."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER.format(law_name=law_name, wiki_text=wiki_text)},
    ]
    try:
        response, _, _ = call_llm_api(messages, model_name=model, temperature=0.1)
        if not response:
            return None

        text = response.strip()

        # Strip markdown code fences if present
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                stripped = part.lstrip("json").strip()
                if stripped.startswith("{"):
                    text = stripped
                    break

        # Try direct parse, then brace-bounded extraction as fallback
        for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue

        print(f"[LawSourcer]   JSON parse failed for '{law_name}'")
        return None
    except Exception as e:
        print(f"[LawSourcer]   LLM call failed for '{law_name}': {e}")
        return None


def _is_already_in_bench(name: str) -> bool:
    name_lower = name.lower()
    return any(existing in name_lower or name_lower in existing
               for existing in _EXISTING_MODULES)


# ── Public API ─────────────────────────────────────────────────────────────────

def source_physics_laws(
    max_laws: int = 50,
    model: str = "cs46",
    api_delay: float = 0.4,
    extra_categories: Optional[list] = None,
) -> list:
    """
    Full pipeline: collect Wikipedia law candidates → fetch intros → LLM extraction → return catalog.

    Args:
        max_laws:          Maximum number of laws to process end-to-end.
        model:             LLM model key for metadata extraction.
        api_delay:         Seconds to wait between Wikipedia API calls.
        extra_categories:  Additional Wikipedia category names to pull from beyond
                           the default supplementary ones.

    Returns:
        List of law metadata dicts. Only physics laws (is_physics=True) are kept;
        each entry has an 'already_in_bench' field.
    """
    supp_categories = _SUPPLEMENTARY_CATEGORIES + (extra_categories or [])

    # ── Collect candidates ─────────────────────────────────────────────────────
    seen: set = set()
    candidates: list = []

    # Primary source: links from the list article
    print("[LawSourcer] Fetching candidates from Wikipedia list article...")
    for title in _get_list_article_links():
        if title not in seen:
            seen.add(title)
            candidates.append(title)
    time.sleep(api_delay)

    # Supplementary: direct category members
    for cat in supp_categories:
        for title in _get_category_members(cat):
            if title not in seen and any(kw in title.lower() for kw in _LAW_KEYWORDS):
                seen.add(title)
                candidates.append(title)
        time.sleep(api_delay)

    print(f"[LawSourcer] {len(candidates)} unique candidates collected")

    # ── Fetch intros + extract metadata ───────────────────────────────────────
    catalog: list = []
    processed = 0

    for title in candidates:
        if processed >= max_laws:
            break

        print(f"[LawSourcer] ({processed + 1}/{max_laws}) {title}")

        wiki_text = _fetch_article_intro(title)
        time.sleep(api_delay)

        if not wiki_text:
            print("[LawSourcer]   skip — no article text")
            continue

        metadata = _extract_law_metadata(title, wiki_text, model=model)
        if metadata is None:
            continue

        if not metadata.get("is_physics", False):
            print(f"[LawSourcer]   skip — not physics")
            continue

        metadata["wiki_title"]      = title
        metadata["wiki_url"]        = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        metadata["already_in_bench"] = _is_already_in_bench(metadata.get("name", title))

        tag = "benchmarkable" if metadata.get("is_benchmarkable") else "not benchmarkable"
        print(f"[LawSourcer]   {tag}: {metadata.get('name', title)}")

        catalog.append(metadata)
        processed += 1
        time.sleep(api_delay)

    benchmarkable = sum(1 for l in catalog if l.get("is_benchmarkable"))
    new_laws      = sum(1 for l in catalog if l.get("is_benchmarkable") and not l.get("already_in_bench"))
    print(f"\n[LawSourcer] Done — {len(catalog)} laws | {benchmarkable} benchmarkable | {new_laws} new to NewtonBench")

    return catalog
