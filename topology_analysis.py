# topology_analysis.py
#
# Analyze the "topology" / geometry of the hyperparameter landscape:
# - Treat each evaluated config as a node in a graph
# - Connect nodes that differ by only 1 hyperparameter ("1-step mutation")
# - Study connected components of low-loss sublevel sets
# - Compare EA vs random vs BO vs others

import json
import argparse
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple


# ---------- Encoding configs into discrete vectors ----------

ATTN_TYPE_MAP = {
    "full": 0,
    "chunked": 1,
    "hybrid": 2,
}


HP_KEYS_ORDER = [
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
    "attention_type",
    "chunk_size",
    "batch_size",
]


def encode_config(cfg: Dict[str, Any]) -> Tuple:
    """
    Encode a config dict into a discrete tuple we can compare.
    We stick to the keys we actually use in the search.
    """
    vec = []
    for k in HP_KEYS_ORDER:
        if k == "attention_type":
            v = cfg.get(k, "full")
            vec.append(ATTN_TYPE_MAP.get(v, -1))
        else:
            v = cfg.get(k, 0)
            # Cast to int where appropriate, keep dropout as float
            if isinstance(v, float) and k != "dropout":
                v = int(v)
            vec.append(v)
    return tuple(vec)


def hamming_steps(a: Tuple, b: Tuple) -> int:
    """
    Count how many hyperparameters differ between two encoded configs.
    """
    assert len(a) == len(b)
    steps = 0
    for va, vb in zip(a, b):
        if va != vb:
            steps += 1
    return steps


# ---------- Graph construction & components ----------

def build_neighbor_graph(encoded: List[Tuple]) -> Dict[int, List[int]]:
    """
    Build a graph where nodes i,j have an edge if they differ in at most
    1 hyperparameter (hamming_steps <= 1).
    """
    n = len(encoded)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_steps(encoded[i], encoded[j]) <= 1:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def connected_components(nodes: List[int], adj: Dict[int, List[int]]) -> List[List[int]]:
    """
    Compute connected components within a subset of nodes using BFS.
    """
    node_set = set(nodes)
    visited = set()
    comps = []

    for u in nodes:
        if u in visited:
            continue
        comp = []
        q = deque([u])
        visited.add(u)
        while q:
            v = q.popleft()
            comp.append(v)
            for w in adj[v]:
                if w in node_set and w not in visited:
                    visited.add(w)
                    q.append(w)
        comps.append(comp)
    return comps


# ---------- Loading records ----------

def load_jsonl(path: str, source_label: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file. Each line is a record (candidate).
    We attach a 'source' field so we know whether it's EA, random, or BO.
    If the file does not exist, return empty list.
    """
    records = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["source"] = source_label
                records.append(rec)
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}, skipping.")
    return records


def extract_candidate(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a record into a flat dict we can use:
    - config
    - val_loss
    - tokens_per_second
    - num_params
    - source
    """
    cfg = rec.get("config", {})
    val_loss = rec.get("val_loss", None)
    tps = rec.get("tokens_per_second", rec.get("tokens_per_sec", None))
    num_params = rec.get("num_params", None)

    # Some logs might have 'fitness' = -val_loss; we prefer explicit val_loss.
    if val_loss is None and "fitness" in rec:
        val_loss = -float(rec["fitness"])

    return {
        "config": cfg,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "tokens_per_second": float(tps) if tps is not None else None,
        "num_params": int(num_params) if num_params is not None else None,
        "source": rec.get("source", "unknown"),
    }


# ---------- Main analysis ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea_file",
        type=str,
        default="ea_candidates.jsonl",
        help="JSONL file for EA candidates."
    )
    parser.add_argument(
        "--random_file",
        type=str,
        default="random_candidates.jsonl",
        help="JSONL file for random-search candidates."
    )
    parser.add_argument(
        "--bo_file",
        type=str,
        default="bayes_candidates.jsonl",
        help="JSONL file for Bayesian optimization candidates (optional)."
    )
    parser.add_argument(
        "--loss_thresholds",
        type=str,
        default="6.5,6.3,6.1,6.0,5.9",
        help="Comma-separated val_loss thresholds for sublevel-set analysis."
    )
    args = parser.parse_args()

    # Load records
    raw_records = []
    raw_records += load_jsonl(args.ea_file, "ea")
    raw_records += load_jsonl(args.random_file, "random")
    raw_records += load_jsonl(args.bo_file, "bo")

    if not raw_records:
        print("No records loaded. Check file paths.")
        return

    # Normalize
    candidates = []
    for r in raw_records:
        cand = extract_candidate(r)
        if cand["val_loss"] is None:
            continue
        candidates.append(cand)

    print(f"Loaded {len(candidates)} total candidates.")

    # Encode configs
    encoded = [encode_config(c["config"]) for c in candidates]
    adj = build_neighbor_graph(encoded)

    # Basic stats by source
    by_source = defaultdict(list)
    for i, c in enumerate(candidates):
        by_source[c["source"]].append(i)

    print("\n=== Basic per-source stats ===")
    for src, idxs in by_source.items():
        best_idx = min(idxs, key=lambda i: candidates[i]["val_loss"])
        best = candidates[best_idx]
        print(
            f"{src}: n={len(idxs)}, "
            f"best_loss={best['val_loss']:.4f}, "
            f"best_ppl={pow(2.0, best['val_loss']) if False else None}, "
            f"best_tps={best['tokens_per_second']}"
        )

    # Sublevel-set topology: connected components vs threshold
    thresholds = [float(x) for x in args.loss_thresholds.split(",")]

    print("\n=== Sublevel set connectivity (by val_loss threshold) ===")
    for tau in thresholds:
        idxs = [i for i, c in enumerate(candidates) if c["val_loss"] <= tau]
        if not idxs:
            print(f"tau={tau:.3f}: no candidates")
            continue
        comps = connected_components(idxs, adj)
        print(f"\ntau={tau:.3f}: |S_tau|={len(idxs)}, components={len(comps)}")

        # For each component, count how many from each source
        for j, comp in enumerate(comps):
            counts = defaultdict(int)
            for i in comp:
                counts[candidates[i]["source"]] += 1
            summary = ", ".join(f"{src}:{cnt}" for src, cnt in counts.items())
            best_idx = min(comp, key=lambda i: candidates[i]["val_loss"])
            print(
                f"  comp {j}: size={len(comp)}, "
                f"best_loss={candidates[best_idx]['val_loss']:.4f}, "
                f"sources=[{summary}]"
            )

    # Local neighborhood quality (average neighbor loss)
    print("\n=== Neighborhood-quality stats ===")
    for src, idxs in by_source.items():
        neighbor_losses = []
        for i in idxs:
            neigh = adj[i]
            if not neigh:
                continue
            avg_loss = sum(candidates[j]["val_loss"] for j in neigh) / len(neigh)
            neighbor_losses.append(avg_loss)
        if neighbor_losses:
            mean_nloss = sum(neighbor_losses) / len(neighbor_losses)
            print(f"{src}: avg neighbor val_loss = {mean_nloss:.4f} over {len(neighbor_losses)} nodes")
        else:
            print(f"{src}: no neighbors in graph.")

    print("\nDone.")


if __name__ == 