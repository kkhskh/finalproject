# eval_mutated_attention.py
#
# Usage:
#   1) Create mutated_attention.py with a new EvoMultiheadSelfAttention class
#      (for example, using the prompt from mutate_attention.py and ChatGPT).
#   2) Run:
#        python eval_mutated_attention.py
#
#   It will:
#     - Dynamically load 1.py as a module
#     - Load EvoMultiheadSelfAttention from mutated_attention.py
#     - Monkey-patch the project to use the mutated attention
#     - Run a 100-step short training/eval on WikiText-2
#     - Print and save metrics to mutated_attention_result.json

import importlib.util
import json
from pathlib import Path


def load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    # 1) Load the main project (1.py) as a module named 'proj'
    proj = load_module_from_path("proj", "1.py")

    # 2) Load the mutated attention module
    mutated = load_module_from_path("mutated_attention", "mutated_attention.py")

    # 3) Monkey-patch EvoMultiheadSelfAttention in the project
    if not hasattr(mutated, "EvoMultiheadSelfAttention"):
        raise RuntimeError(
            "mutated_attention.py must define EvoMultiheadSelfAttention"
        )
    proj.EvoMultiheadSelfAttention = mutated.EvoMultiheadSelfAttention

    # 4) Build a test config (you can tweak this if you want)
    cfg = proj.EvoConfig(
        d_model=256,
        n_heads=2,
        n_layers=4,
        d_ff=384,
        dropout=0.1,
        attention_type="full",  # external API; internal behavior defined by mutated class
        chunk_size=32,
        batch_size=16,
    )

    # 5) Load dataset + tokenizer exactly like in main()
    base_dir = "data/wikitext2"
    print(f"Loading WikiText-2 from {base_dir} ...")
    raw_datasets = proj.load_wikitext2(base_dir)

    print("Loading tokenizer (GPT-2)...")
    tokenizer = proj.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print("Tokenizing + grouping dataset...")
    lm_datasets = proj.tokenize_wikitext2(raw_datasets, tokenizer, block_size=128)

    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # 6) Run the existing short-training evaluation
    #    NOTE: this assumes your train_and_eval_short now returns:
    #      fitness, val_loss, perplexity, train_time, tokens_per_second, num_params
    (
        fitness,
        val_loss,
        ppl,
        train_time,
        tps,
        num_params,
    ) = proj.train_and_eval_short(
        cfg,
        lm_datasets,
        tokenizer,
        vocab_size,
        block_size=128,
        max_train_steps=100,
        max_eval_batches=40,
    )

    result = {
        "fitness": float(fitness),
        "val_loss": float(val_loss),
        "perplexity": float(ppl),
        "train_time": float(train_time),
        "tokens_per_second": float(tps),
        "num_params": int(num_params),
    }

    print("\nMutated attention short-run results:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    Path("mutated_attention_result.json").write_text(
        json.dumps(result, indent=2)
    )
    print("\nSaved mutated attention results to mutated_attention_result.json")


if __name__ == "__main__":
    main()

