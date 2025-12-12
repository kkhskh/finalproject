# CS242 Final Project – Experiment Playbook

This README captures the commands needed to reproduce the experiments and analyses in this repo (EA, random search, Bayesian optimization, neighbor probes, full evals, topology analysis, and final training).

## Setup (Colab/local)
```bash
git clone https://github.com/kkhskh/finalproject.git
cd finalproject
pip install -r requirements.txt
```
Select GPU in Colab (Runtime → Change runtime type → GPU).

## Datasets
The scripts auto-download and cache datasets to `data/`:
- WikiText-2: `data/wikitext2`
- WikiText-103: `data/wikitext103`

## Search runs (short budget)
Short runs log candidates to JSONL with dataset suffixes.

### Evolutionary search (EA)
```bash
python 1.py \
  --mode ea \
  --dataset wt2 \         # or wt103
  --block_size 128 \      # 256 for wt103
  --ea_pop_size 4 \
  --ea_generations 3 \
  --ea_train_steps 100 \
  --ea_eval_batches 40
# outputs: ea_candidates_<dataset>.jsonl
```

### Random search
```bash
python 1.py \
  --mode random \
  --dataset wt2 \
  --block_size 128 \
  --ea_pop_size 4 \
  --ea_generations 3 \
  --ea_train_steps 100 \
  --ea_eval_batches 40
# outputs: random_candidates_<dataset>.jsonl
```

### Bayesian optimization (Optuna)
```bash
python bayes_search.py \
  --dataset wt2 \
  --block_size 128 \
  --n_trials 12 \
  --max_train_steps 100 \
  --max_eval_batches 40 \
  --log_jsonl bayes_candidates_wt2.jsonl
# outputs: bayes_candidates_<dataset>.jsonl and best_bayes_config_<dataset>.json
```

## Post-hoc evaluation and probes

### Full eval of logged candidates (for ρᴰ)
Runs full training on every config in a log and pairs short/full losses.
```bash
python 1.py \
  --mode full_eval \
  --dataset wt2 \
  --block_size 128 \
  --epochs 5 \                # full-run epochs per config
  --log_jsonl bayes_candidates_wt2.jsonl
# outputs: bayes_candidates_wt2_full_eval.jsonl
```

### Hamming-1 neighbor probing around best (Δ̂(x*))
Evaluates neighbors of the best config in a log with the short budget.
```bash
python 1.py \
  --mode neighbors \
  --dataset wt2 \
  --block_size 128 \
  --ea_train_steps 100 \
  --ea_eval_batches 40 \
  --neighbors_k 16 \
  --log_jsonl ea_candidates_wt2.jsonl
# outputs: ea_candidates_wt2_neighbors.jsonl, prints valley-depth proxy
```

### Repeat eval (noise)
Re-evaluate one arch multiple times with the short budget.
```bash
python 1.py \
  --mode repeat_eval \
  --dataset wt2 \
  --block_size 128 \
  --ea_train_steps 100 \
  --ea_eval_batches 40 \
  --repeat_runs 5 \
  --log_jsonl ea_candidates_wt2.jsonl
# outputs: ea_candidates_wt2_repeats.jsonl
```

## Final / longer training
Train a config to convergence and save a checkpoint.
```bash
python 1.py \
  --mode train_best \
  --dataset wt103 \
  --config_json best_wt103_config.json \
  --epochs 5 \
  --block_size 256
# outputs: evolved_best_model.pt
```

## Topology analysis
After generating EA/Random/BO logs, run:
```bash
python topology_analysis.py \
  --ea_file ea_candidates_wt2.jsonl \
  --random_file random_candidates_wt2.jsonl \
  --bo_file bayes_candidates_wt2.jsonl \
  --loss_thresholds 6.5,6.3,6.1,6.0,5.9
```

## Mutated attention workflow (LLM-assisted)
1. Generate prompt:
   ```bash
   python mutate_attention.py > attention_llm_prompt.txt
   ```
2. Paste prompt into ChatGPT, save returned class to `mutated_attention.py`.
3. Evaluate mutated attention:
   ```bash
   python eval_mutated_attention.py
   # outputs: mutated_attention_result.json
   ```

