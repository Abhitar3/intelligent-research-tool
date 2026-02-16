import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rag
from rag import process_urls


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_terms: List[str]
    expected_sources: List[str]
    difficulty: str = "unknown"


PRESETS: Dict[str, Dict[str, object]] = {
    "python_docs_small": {
        "urls": [
            "https://docs.python.org/3/tutorial/introduction.html",
            "https://docs.python.org/3/tutorial/controlflow.html",
            "https://docs.python.org/3/tutorial/datastructures.html",
        ],
        "cases": [
            EvalCase(
                case_id="p1",
                query="How do you write a for loop in Python?",
                expected_terms=["for", "in", "range"],
                expected_sources=["controlflow.html"],
                difficulty="easy",
            ),
            EvalCase(
                case_id="p2",
                query="What does break do in loops?",
                expected_terms=["break", "loop"],
                expected_sources=["controlflow.html"],
                difficulty="easy",
            ),
            EvalCase(
                case_id="p3",
                query="What is a Python list comprehension?",
                expected_terms=["list comprehension"],
                expected_sources=["datastructures.html"],
                difficulty="medium",
            ),
            EvalCase(
                case_id="p4",
                query="How do you define a function in Python?",
                expected_terms=["def", "function"],
                expected_sources=["controlflow.html"],
                difficulty="easy",
            ),
            EvalCase(
                case_id="p5",
                query="What are dictionaries in Python?",
                expected_terms=["dictionary", "key", "value"],
                expected_sources=["datastructures.html"],
                difficulty="medium",
            ),
        ],
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality with resume-friendly metrics (Hit@k, Recall@k, MRR, source hit)."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="python_docs_small",
        help="Use a built-in benchmark preset.",
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="",
        help="Path to benchmark JSON. If provided, this overrides --preset.",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated k values. Example: 1,3,5",
    )
    parser.add_argument(
        "--min-term-matches",
        type=int,
        default=1,
        help="Minimum matched expected terms in top-k to count as a hit.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional output CSV path for per-query metrics.",
    )
    return parser.parse_args()


def _norm(text: str) -> str:
    return text.strip().lower()


def _build_case(raw: Dict[str, object], idx: int) -> EvalCase:
    return EvalCase(
        case_id=str(raw.get("id", f"q{idx + 1}")),
        query=str(raw["query"]),
        expected_terms=[_norm(x) for x in raw.get("expected_terms", [])],
        expected_sources=[_norm(x) for x in raw.get("expected_sources", [])],
        difficulty=str(raw.get("difficulty", "unknown")),
    )


def load_benchmark(args: argparse.Namespace) -> Tuple[List[str], List[EvalCase], str]:
    if args.benchmark_file:
        path = Path(args.benchmark_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        urls = [str(u) for u in data["urls"]]
        cases = [_build_case(raw, idx) for idx, raw in enumerate(data["cases"])]
        return urls, cases, f"file:{path}"

    preset = PRESETS[args.preset]
    urls = [str(u) for u in preset["urls"]]
    cases = list(preset["cases"])
    return urls, cases, f"preset:{args.preset}"


def find_first_term_match_rank(texts: List[str], expected_terms: List[str]) -> Optional[int]:
    for rank, text in enumerate(texts, start=1):
        if any(term in text for term in expected_terms):
            return rank
    return None


def evaluate(
    cases: List[EvalCase],
    ks: List[int],
    min_term_matches: int,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], float, List[Dict[str, object]]]:
    hits = {k: 0 for k in ks}
    term_recalls = {k: [] for k in ks}
    source_hits = {k: 0 for k in ks}
    reciprocal_ranks: List[float] = []
    rows: List[Dict[str, object]] = []

    for case in cases:
        docs = rag.vector_store.similarity_search(case.query, k=max(ks))
        texts = [_norm(d.page_content) for d in docs]
        sources = [_norm(str(d.metadata.get("source", ""))) for d in docs]

        first_match_rank = find_first_term_match_rank(texts, case.expected_terms)
        rr = (1.0 / first_match_rank) if first_match_rank else 0.0
        reciprocal_ranks.append(rr)

        print(f"\n[{case.case_id}] {case.query}")
        for k in ks:
            topk_texts = texts[:k]
            topk_sources = sources[:k]

            found_terms = {term for term in case.expected_terms if any(term in t for t in topk_texts)}
            found_sources = {
                s for s in case.expected_sources if any(s in src for src in topk_sources)
            } if case.expected_sources else set()

            hit = 1 if len(found_terms) >= min_term_matches else 0
            term_recall = (
                len(found_terms) / len(case.expected_terms) if case.expected_terms else 0.0
            )
            source_hit = (
                1 if (case.expected_sources and len(found_sources) > 0) else 0
            )

            hits[k] += hit
            term_recalls[k].append(term_recall)
            source_hits[k] += source_hit

            print(
                f"  k={k}: hit={hit}, term_recall={term_recall:.2f}, "
                f"source_hit={source_hit}, matched_terms={sorted(found_terms)}"
            )

            row = {
                "case_id": case.case_id,
                "query": case.query,
                "difficulty": case.difficulty,
                "k": k,
                "hit": hit,
                "term_recall": f"{term_recall:.4f}",
                "source_hit": source_hit,
                "first_term_match_rank": first_match_rank if first_match_rank else "",
                "reciprocal_rank": f"{rr:.4f}",
                "matched_terms": ";".join(sorted(found_terms)),
                "expected_terms_count": len(case.expected_terms),
                "expected_sources_count": len(case.expected_sources),
            }
            rows.append(row)

    n = len(cases)
    hit_at_k = {k: hits[k] / n for k in ks}
    recall_at_k = {k: sum(term_recalls[k]) / n for k in ks}
    source_hit_at_k = {k: source_hits[k] / n for k in ks}
    mrr = sum(reciprocal_ranks) / n
    return hit_at_k, recall_at_k, source_hit_at_k, mrr, rows


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "query",
        "difficulty",
        "k",
        "hit",
        "term_recall",
        "source_hit",
        "first_term_match_rank",
        "reciprocal_rank",
        "matched_terms",
        "expected_terms_count",
        "expected_sources_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ks = sorted({int(x.strip()) for x in args.k_values.split(",") if x.strip()})

    urls, cases, source_name = load_benchmark(args)
    print(f"Loaded benchmark from {source_name}")
    print(f"Total URLs: {len(urls)} | Total queries: {len(cases)}")
    print("Indexing URLs...")
    for status in process_urls(urls):
        print("-", status)

    hit_at_k, recall_at_k, source_hit_at_k, mrr, rows = evaluate(
        cases=cases,
        ks=ks,
        min_term_matches=args.min_term_matches,
    )

    print("\n=== Aggregate Metrics ===")
    for k in ks:
        print(f"Hit@{k}: {hit_at_k[k]:.2f}")
        print(f"TermRecall@{k}: {recall_at_k[k]:.2f}")
        print(f"SourceHit@{k}: {source_hit_at_k[k]:.2f}")
    print(f"MRR: {mrr:.2f}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        write_csv(rows, out_path)
        print(f"Saved per-query metrics to: {out_path}")


if __name__ == "__main__":
    main()
