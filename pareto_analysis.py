import json
import os

def load_records(path):
    if not os.path.exists(path):
        print(f"[warn] {path} not found")
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def is_dominated(a, b):
    # a is dominated by b if b is <= on both (loss, -tokens/s) and strictly better on at least one
    # We minimize loss, maximize tokens/s
    return (b["val_loss"] <= a["val_loss"] and b["tokens_per_second"] >= a["tokens_per_second"]) and \
           (b["val_loss"] < a["val_loss"] or b["tokens_per_second"] > a["tokens_per_second"])

def pareto_front(records):
    front = []
    for i, r in enumerate(records):
        dominated = False
        for j, s in enumerate(records):
            if i != j and is_dominated(r, s):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front

def summarize(label, records):
    print(f"\n=== {label} candidates: {len(records)} ===")
    if not records:
        print("  (no records found)")
        return

    best = min(records, key=lambda r: r["val_loss"])
    print(f"Best by loss: loss={best['val_loss']:.4f}, ppl={best['perplexity']:.2f}, "
          f"tps={best['tokens_per_second']:.1f}, params={best['num_params']}, "
          f"config={best['config']}")

    front = pareto_front(records)
    print(f"\nPareto front ({len(front)} configs, loss vs tokens/s):")
    for r in sorted(front, key=lambda x: x["val_loss"]):
        print(f"  loss={r['val_loss']:.4f}, ppl={r['perplexity']:.2f}, "
              f"tps={r['tokens_per_second']:.1f}, params={r['num_params']}, "
              f"config={r['config']}")

def main():
    ea_records = load_records("ea_candidates.jsonl")
    rand_records = load_records("random_candidates.jsonl")

    summarize("EA", ea_records)
    summarize("Random", rand_records)

if __name__ == "__main__":
    main()
