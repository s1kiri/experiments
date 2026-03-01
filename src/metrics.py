"""Pure-Python BLEU-4 and ROUGE-L metrics.

Drop-in offline replacements for evaluate.load("bleu") / evaluate.load("rouge").
Return dicts with the same keys used in lightning_module logging:
    bleu4(preds, refs)   → {"bleu": float}
    rouge_l(preds, refs) → {"rougeL": float}

No external dependencies beyond the standard library.
"""

from collections import Counter
import math


# ---------------------------------------------------------------------------
# BLEU-4 (corpus-level)
# ---------------------------------------------------------------------------

def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def bleu4(predictions: list[str], references: list[str]) -> dict:
    """Corpus-level BLEU-4.

    Args:
        predictions: list of hypothesis strings (whitespace-tokenised)
        references:  list of reference strings (one per hypothesis)
    Returns:
        {"bleu": float}  in [0, 1]
    """
    clipped_counts = Counter()
    total_counts   = Counter()
    hyp_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        p_toks = pred.split()
        r_toks = ref.split()
        hyp_len += len(p_toks)
        ref_len  += len(r_toks)

        for n in range(1, 5):
            p_ng = _ngrams(p_toks, n)
            r_ng = _ngrams(r_toks, n)
            clipped = {k: min(v, r_ng[k]) for k, v in p_ng.items()}
            clipped_counts[n] += sum(clipped.values())
            total_counts[n]   += sum(p_ng.values())

    # Brevity penalty
    if hyp_len == 0:
        return {"bleu": 0.0}
    bp = 1.0 if hyp_len >= ref_len else math.exp(1.0 - ref_len / hyp_len)

    # Geometric mean of 1-4 gram precisions
    log_avg = 0.0
    for n in range(1, 5):
        if clipped_counts[n] == 0 or total_counts[n] == 0:
            return {"bleu": 0.0}
        log_avg += math.log(clipped_counts[n] / total_counts[n])

    return {"bleu": bp * math.exp(log_avg / 4)}


# ---------------------------------------------------------------------------
# ROUGE-L (sentence-level, averaged)
# ---------------------------------------------------------------------------

def _lcs_len(a: list, b: list) -> int:
    """Length of longest common subsequence (space-efficient, O(min(m,n)) memory)."""
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(predictions: list[str], references: list[str]) -> dict:
    """Sentence-level ROUGE-L F1, averaged over the list.

    Args:
        predictions: list of hypothesis strings
        references:  list of reference strings
    Returns:
        {"rougeL": float}  in [0, 1]
    """
    scores = []
    for pred, ref in zip(predictions, references):
        p_toks = pred.split()
        r_toks = ref.split()
        if not p_toks or not r_toks:
            scores.append(0.0)
            continue
        lcs = _lcs_len(p_toks, r_toks)
        precision = lcs / len(p_toks)
        recall    = lcs / len(r_toks)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        scores.append(f1)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {"rougeL": avg}
