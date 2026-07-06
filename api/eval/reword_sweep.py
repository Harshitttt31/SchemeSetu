r"""Reword sweep -- retrieval robustness under paraphrase (Phase 02.5 bridge step).

Phase 02 found that similarity scores on this narrow corpus are COMPRESSED (close
together). This script tests whether that is merely cosmetic by checking RANK
stability: for each intent we retrieve on three different phrasings and ask whether
one and the same chunk stays in the top-3 for every phrasing.

    STABLE   = at least one chunk_id appears in the top-3 of ALL phrasings.
               That shared "anchor" chunk is the observed-correct chunk for the
               intent, and gets harvested into eval_set.seed.json as a first
               labelled row for Phase 04.
    UNSTABLE = no single chunk survives in the top-3 across all phrasings.

It also reports the observed top-hit SCORE RANGE across every query. That range is
what Phase 03 uses to calibrate the not-found sanity floor empirically -- we never
hard-code a threshold from intuition, precisely because the scores are compressed.

We call the retrieval primitives directly (embed_query + search) rather than
retrieve(), so a single Gemini client and a single open collection are reused
across all queries instead of being rebuilt per call. retrieve.py is untouched.

Run (from api/, Windows):
    .\.venv\Scripts\python.exe eval\reword_sweep.py
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

# Make the sibling `rag` package importable no matter how this file is launched
# (`python eval/reword_sweep.py`, `-m eval.reword_sweep`, or from another cwd).
API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from rag import store                      # noqa: E402  (after sys.path tweak)
from rag.embed import embed_query, get_client  # noqa: E402
from rag.retrieve import search            # noqa: E402

logger = logging.getLogger(__name__)

TOP_N = 3                                   # "top-3" per the acceptance definition
DEFAULT_GROUPS = Path(__file__).with_name("reword_groups.json")
DEFAULT_SEED_OUT = Path(__file__).with_name("eval_set.seed.json")


def load_groups(path: Path) -> list[dict]:
    """Load the intent groups: [{intent, phrasings: [q1, q2, q3]}, ...]."""
    groups = json.loads(path.read_text(encoding="utf-8"))
    for g in groups:
        if not g.get("phrasings"):
            raise ValueError(f"Group {g.get('intent')!r} has no phrasings.")
    return groups


def decide_anchors(variants: list[dict]) -> tuple[list[str], str | None, str]:
    """Pure decision step: given each variant's hit list, find the STABLE anchors.

    An anchor is a chunk_id present in the top-N of EVERY phrasing. Anchors are
    ranked strongest-first by lowest mean rank across variants (1 = best position),
    tie-broken by highest mean score. Returns (ranked_anchors, primary, verdict).
    Kept free of any API/collection dependency so it can be tested offline.
    """
    topn_sets = [{h.chunk_id for h in v["hits"]} for v in variants]
    anchors = set.intersection(*topn_sets) if topn_sets else set()

    def mean_rank(cid: str) -> float:
        ranks = []
        for v in variants:
            for i, h in enumerate(v["hits"], start=1):
                if h.chunk_id == cid:
                    ranks.append(i)
                    break
        return statistics.mean(ranks) if ranks else float("inf")

    def mean_score(cid: str) -> float:
        scores = [h.score for v in variants for h in v["hits"] if h.chunk_id == cid]
        return statistics.mean(scores) if scores else float("-inf")

    ranked = sorted(anchors, key=lambda c: (mean_rank(c), -mean_score(c)))
    primary = ranked[0] if ranked else None
    verdict = "STABLE" if ranked else "UNSTABLE"
    return ranked, primary, verdict


def sweep_group(group: dict, collection, genai_client) -> dict:
    """Retrieve top-N for every phrasing of one intent and decide STABLE/UNSTABLE.

    Returns a result dict with per-variant hits, the shared anchor chunk(s), the
    chosen primary anchor, and the verdict.
    """
    variants: list[dict] = []
    for phrasing in group["phrasings"]:
        emb = embed_query(phrasing, client=genai_client)
        hits = search(collection, emb, TOP_N)
        variants.append({"question": phrasing, "hits": hits})

    ranked_anchors, primary, verdict = decide_anchors(variants)
    return {
        "intent": group["intent"],
        "phrasings": group["phrasings"],
        "variants": variants,
        "anchors": ranked_anchors,
        "primary": primary,
        "verdict": verdict,
    }


def _anchor_snippet(result: dict) -> str:
    """First 200 chars of the primary anchor's text (so the human can eyeball it)."""
    if not result["primary"]:
        return ""
    for v in result["variants"]:
        for h in v["hits"]:
            if h.chunk_id == result["primary"]:
                return " ".join(h.text.split())[:200]
    return ""


def print_group(result: dict) -> None:
    """Human-readable per-group table: each variant -> its top-N chunk_ids+scores."""
    print(f"\n=== {result['intent']} ===")
    for v in result["variants"]:
        print(f"  Q: {v['question']}")
        if not v["hits"]:
            print("     (no results)")
        for rank, h in enumerate(v["hits"], start=1):
            mark = " *" if h.chunk_id == result["primary"] else "  "
            print(f"    {rank}.{mark}score={h.score:.3f}  {h.chunk_id}  "
                  f"[{h.source_doc} p{h.page}]")
    print(f"  VERDICT: {result['verdict']}", end="")
    if result["anchors"]:
        print(f"  anchor(s) in top-{TOP_N} of all variants: {result['anchors']}")
        print(f"    primary -> {result['primary']}")
        print(f"    text: {_anchor_snippet(result)!r}")
    else:
        print(f"  no chunk survived in top-{TOP_N} across all variants.")


def top_hit_score_range(results: list[dict]) -> dict:
    """Min/mean/max of the BEST score of every query -- calibrates the not-found floor."""
    top1 = [v["hits"][0].score for r in results for v in r["variants"] if v["hits"]]
    if not top1:
        return {}
    return {
        "n_queries": len(top1),
        "min": min(top1),
        "mean": statistics.mean(top1),
        "max": max(top1),
    }


def build_seed_rows(results: list[dict], score_range: dict) -> list[dict]:
    """Turn STABLE anchors into starter labelled rows for Phase 04's eval set.

    UNSTABLE groups are still written (with their best-effort anchor left empty and
    verdict UNSTABLE) so the human sees exactly which intent needs a chunking fix.
    reference_answer is left blank -- that is the human's job to author.
    """
    rows = []
    for r in results:
        rows.append({
            "intent": r["intent"],
            "question": r["phrasings"][0],
            "variants": r["phrasings"],
            "relevant_chunk_ids": r["anchors"],          # [] when UNSTABLE
            "primary_chunk_id": r["primary"],            # None when UNSTABLE
            "verdict": r["verdict"],
            "reference_answer": "",
            "note": "Harvested by reword_sweep; verify the anchor text before trusting.",
        })
    return {"observed_top_hit_score_range": score_range, "rows": rows}


def run(groups_path: Path, seed_out: Path, k: int) -> int:
    """Run the full sweep. Returns process exit code (0 = all STABLE)."""
    global TOP_N
    TOP_N = k

    groups = load_groups(groups_path)

    client = store.get_client()
    collection = store.get_collection(client)
    if collection.count() == 0:
        raise RuntimeError("Index is empty. Run scripts/build_index.py first.")

    genai_client = get_client()  # one Gemini client reused for every query embedding

    results = [sweep_group(g, collection, genai_client) for g in groups]
    for r in results:
        print_group(r)

    score_range = top_hit_score_range(results)
    seed = build_seed_rows(results, score_range)
    seed_out.write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Summary --------------------------------------------------------------
    stable = [r["intent"] for r in results if r["verdict"] == "STABLE"]
    unstable = [r["intent"] for r in results if r["verdict"] == "UNSTABLE"]

    print("\n" + "=" * 60)
    print(f"STABLE   ({len(stable)}/{len(results)}): {stable}")
    print(f"UNSTABLE ({len(unstable)}/{len(results)}): {unstable}")
    if score_range:
        print(f"\nTop-hit score range over {score_range['n_queries']} queries: "
              f"min={score_range['min']:.3f}  mean={score_range['mean']:.3f}  "
              f"max={score_range['max']:.3f}")
        print("  -> Phase 03 not-found floor: calibrate BELOW this observed min "
              f"({score_range['min']:.3f}); do not hard-code from intuition.")
    print(f"\nSeed labels written: {seed_out}")

    if unstable:
        print("\nACCEPTANCE: UNSTABLE group(s) present. Make exactly ONE chunking "
              "adjustment (smaller chunk = one topic), rebuild the index, and re-run "
              "the sweep for the affected group before proceeding to Phase 03.")
        return 1
    print("\nACCEPTANCE: all groups STABLE -> proceed to Phase 03 with no tuning.")
    return 0


def _cli() -> None:
    p = argparse.ArgumentParser(description="Retrieval robustness reword sweep.")
    p.add_argument("--groups", type=Path, default=DEFAULT_GROUPS,
                   help="JSON list of {intent, phrasings[]} groups.")
    p.add_argument("--seed-out", type=Path, default=DEFAULT_SEED_OUT,
                   help="Where to write harvested starter eval labels.")
    p.add_argument("-k", type=int, default=TOP_N, help="Top-N window (default 3).")
    args = p.parse_args()
    raise SystemExit(run(args.groups, args.seed_out, args.k))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    _cli()
