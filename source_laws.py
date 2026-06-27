"""
source_laws.py — CLI for sourcing physics laws from Wikipedia.

Fetches physics law articles from Wikipedia categories, uses an LLM to extract
structured metadata (formula, variables, benchmark suitability), and saves a
JSON catalog.  Only laws with is_physics=True are included; each entry carries
an `already_in_bench` flag for laws already implemented in NewtonBench.

Usage
-----
# Default: 40 laws, all physics categories, save to laws_catalog.json
python source_laws.py

# Specific categories, benchmarkable only, custom output
python source_laws.py \\
    --output results/new_laws.json \\
    --max_laws 60 \\
    --model cs46 \\
    --categories Physics_laws Electromagnetism Optics \\
    --benchmarkable_only

Options
-------
--output              Output JSON file path (default: laws_catalog.json)
--max_laws            Max laws to process end-to-end (default: 40)
--model               LLM model key for extraction (default: cs46)
--categories          Wikipedia category names (default: all PHYSICS_CATEGORIES)
--benchmarkable_only  Only write benchmarkable laws to the output file
--delay               Seconds between Wikipedia API calls (default: 0.4)
"""

import argparse
import json
import os

from utils.law_sourcer import source_physics_laws, PHYSICS_CATEGORIES


def _print_summary(catalog: list) -> None:
    benchmarkable = [l for l in catalog if l.get("is_benchmarkable")]
    new_laws      = [l for l in benchmarkable if not l.get("already_in_bench")]

    print("\n" + "═" * 62)
    print("  SOURCED LAWS — SUMMARY")
    print("═" * 62)
    print(f"  Total physics laws : {len(catalog)}")
    print(f"  Benchmarkable      : {len(benchmarkable)}")
    print(f"  New to NewtonBench : {len(new_laws)}")

    if new_laws:
        print("\n  NEW BENCHMARKABLE LAWS")
        print("  " + "─" * 58)
        for law in new_laws:
            print(f"  • {law['name']}  [{law.get('domain', '?')}]")
            print(f"      Formula    : {law.get('formula_text', 'N/A')}")
            print(f"      Signature  : {law.get('function_signature', 'N/A')}")
            print()

    existing = [l for l in benchmarkable if l.get("already_in_bench")]
    if existing:
        print("  ALREADY IN NEWTONBENCH")
        print("  " + "─" * 58)
        for law in existing:
            print(f"  • {law['name']}  [{law.get('domain', '?')}]")

    print("═" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source physics laws from Wikipedia using LLM extraction"
    )
    parser.add_argument(
        "--output", type=str, default="laws_catalog.json",
        help="Output JSON file path (default: laws_catalog.json)"
    )
    parser.add_argument(
        "--max_laws", type=int, default=40,
        help="Maximum number of laws to process (default: 40)"
    )
    parser.add_argument(
        "--model", type=str, default="cs46",
        help="LLM model key for metadata extraction (default: cs46)"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        metavar="CATEGORY",
        help="Wikipedia category names to pull from (default: all PHYSICS_CATEGORIES)"
    )
    parser.add_argument(
        "--benchmarkable_only", action="store_true",
        help="Only include benchmarkable laws in the saved output"
    )
    parser.add_argument(
        "--delay", type=float, default=0.4,
        help="Seconds between Wikipedia API calls (default: 0.4)"
    )
    args = parser.parse_args()

    categories = args.categories or PHYSICS_CATEGORIES

    print("═" * 62)
    print("  WIKIPEDIA PHYSICS LAW SOURCER")
    print("═" * 62)
    print(f"  Max laws  : {args.max_laws}")
    print(f"  LLM model : {args.model}")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Output    : {args.output}")
    print("═" * 62 + "\n")

    catalog = source_physics_laws(
        max_laws=args.max_laws,
        model=args.model,
        api_delay=args.delay,
        extra_categories=args.categories,
    )

    _print_summary(catalog)

    if args.benchmarkable_only:
        catalog = [l for l in catalog if l.get("is_benchmarkable")]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\n  Saved {len(catalog)} laws to: {args.output}")


if __name__ == "__main__":
    main()
