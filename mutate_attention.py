# mutate_attention.py
#
# Usage:
#   python mutate_attention.py > attention_llm_prompt.txt
# Then copy the prompt from attention_llm_prompt.txt into ChatGPT,
# and paste the returned class code into mutated_attention.py.

import json
from pathlib import Path

MAIN_FILE = Path("1.py")
BEST_EA_JSON = Path("best_config.json")
BEST_RANDOM_JSON = Path("best_random_config.json")


def extract_attention_class(src: str) -> str:
    """
    Extract the EvoMultiheadSelfAttention class definition from 1.py
    by scanning from the 'class EvoMultiheadSelfAttention' line until
    the next top-level 'class ' or EOF.
    """
    lines = src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("class EvoMultiheadSelfAttention"):
            start = i
            break
    if start is None:
        raise RuntimeError("Could not find EvoMultiheadSelfAttention in 1.py")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        # top-level class begins in column 0
        if lines[j].startswith("class "):
            end = j
            break

    return "\n".join(lines[start:end])


def load_metrics():
    ea = {}
    rnd = {}
    if BEST_EA_JSON.exists():
        try:
            ea = json.loads(BEST_EA_JSON.read_text())
        except Exception:
            ea = {}
    if BEST_RANDOM_JSON.exists():
        try:
            rnd = json.loads(BEST_RANDOM_JSON.read_text())
        except Exception:
            rnd = {}
    return {"ea_best": ea, "random_best": rnd}


def build_prompt(attn_code: str, metrics: dict) -> str:
    return f"""
You are an expert in efficient Transformer attention for small language models.

Below is the current PyTorch implementation of the EvoMultiheadSelfAttention class
used in our WikiText-2 language model. It already supports three attention types:
'full', 'chunked', and 'hybrid' (where 'hybrid' uses a learned gate between full and
chunked attention).

```python
{attn_code}
```

Here is a JSON summary of the best configurations found by our evolutionary search
and random search under a short 100-step training budget:

{json.dumps(metrics, indent=2)}

Task:
Propose an improved version of the EvoMultiheadSelfAttention class that keeps the
same public API but changes the internal attention computation to better trade off
validation loss (perplexity) vs training throughput (tokens/second) on small language models.

Constraints:

- Keep the class name exactly: EvoMultiheadSelfAttention

- Keep the init signature and forward signature exactly the same.

- The module must remain causal and compatible with existing code that calls it
  (same shapes, same return type).

- You may change the internal implementation of 'full', 'chunked', and/or
  'hybrid' attention, or add additional internal tricks (e.g., different sparsity
  pattern, better chunking scheme, gating structure, etc.), as long as the external
  behavior is compatible.

- The code must be valid PyTorch code (Python 3) with no placeholder ellipses.

Return ONLY the complete updated class definition for EvoMultiheadSelfAttention,
from the 'class' line down to the end of the class body.
"""


def main():
    src = MAIN_FILE.read_text()
    attn_code = extract_attention_class(src)
    metrics = load_metrics()
    prompt = build_prompt(attn_code, metrics)
    print(prompt)


if __name__ == "__main__":
    main()

