# bayes_search.py

import argparse
import importlib.util
import json
import os
from pathlib import Path

import optuna


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="wt2",
        choices=["wt2", "wt103"],
        help="wt2 = WikiText-2, wt103 = WikiText-103",
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument(
        "--n_trials",
        type=int,
        default=12,  # match EA budget: ea_pop_size * ea_generations
        help="Number of BO trials (candidates).",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="Short-run train steps, should match EA/random.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=40,
        help="Short-run eval batches, should match EA/random.",
    )
    parser.add_argument(
        "--log_jsonl",
        type=str,
        default="bayes_candidates.jsonl",
        help="Where to log per-trial results.",
    )
    args = parser.parse_args()

    # Load your main project module (1.py)
    proj = load_module_from_path("proj", "1.py")

    # ----------------------------
    # Prepare dataset + tokenizer
    # ----------------------------
    if args.dataset == "wt2":
        base_dir = "data/wikitext2"
        print(f"Loading WikiText-2 from {base_dir} ...")
        raw_datasets = proj.load_wikitext2(base_dir)
    else:
        base_dir = "data/wikitext103"
        print(f"Loading WikiText-103 from {base_dir} ...")
        raw_datasets = proj.load_wikitext103(base_dir)

    print("Loading tokenizer (GPT-2)...")
    tokenizer = proj.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print("Tokenizing + grouping dataset...")
    lm_datasets = proj.tokenize_wikitext2(
        raw_datasets, tokenizer, block_size=args.block_size
    )

    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # ----------------------------
    # Logging setup
    # ----------------------------
    log_path = Path(args.log_jsonl)
    # Make sure we don't append to old runs by accident
    if log_path.exists():
        print(f"Removing existing log file: {log_path}")
        log_path.unlink()

    def objective(trial: optuna.Trial):
        # ----------------------------
        # Define search space
        # ----------------------------
        d_model = trial.suggest_categorical("d_model", [128, 192, 256])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        # sanity: d_model must be divisible by n_heads
        if d_model % n_heads != 0:
            # Penalize impossible configs with huge loss
            return 1e9

        n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
        d_ff = trial.suggest_categorical("d_ff", [256, 384, 512])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])

        attention_type = trial.suggest_categorical(
            "attention_type", ["full", "chunked", "hybrid"]
        )
        chunk_size = trial.suggest_categorical("chunk_size", [16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        cfg = proj.EvoConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_type=attention_type,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )

        print(f"[BO] Evaluating trial {trial.number} with config: {cfg}")

        # Short-run training, same as EA/random
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
            block_size=args.block_size,
            max_train_steps=args.max_train_steps,
            max_eval_batches=args.max_eval_batches,
        )

        record = {
            "search_type": "bayes",
            "trial": trial.number,
            "dataset": args.dataset,
            "config": cfg.__dict__,
            "fitness": fitness,
            "val_loss": val_loss,
            "perplexity": ppl,
            "train_time": train_time,
            "tokens_per_second": tps,
            "num_params": num_params,
        }

        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        # We minimize val_loss (not the neg fitness) as the BO objective
        return val_loss

    # ----------------------------
    # Run BO
    # ----------------------------
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest BO trial:")
    print(study.best_trial)

    # Save best config in the same style as EA/random
    best_cfg_dict = study.best_trial.params

    # Map params -> EvoConfig-friendly dict
    best_config = {
        "d_model": best_cfg_dict["d_model"],
        "n_heads": best_cfg_dict["n_heads"],
        "n_layers": best_cfg_dict["n_layers"],
        "d_ff": best_cfg_dict["d_ff"],
        "dropout": best_cfg_dict["dropout"],
        "attention_type": best_cfg_dict["attention_type"],
        "chunk_size": best_cfg_dict["chunk_size"],
        "batch_size": best_cfg_dict["batch_size"],
    }

    summary = {
        "search_type": "bayes",
        "dataset": args.dataset,
        "best_trial": int(study.best_trial.number),
        "config": best_config,
        "val_loss": float(study.best_trial.value),
    }

    out_json = f"best_bayes_config_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved best Bayesian config for {args.dataset} to {out_json}")


if __name__ == "__main__":
    main()
