#!/usr/bin/env python3
"""Inference script for the mapper-LLM pipeline.

Manual text mode:
    python infer.py \\
        --checkpoint runs/pooled_qwen_trainable_emb_ctx16/version_0/checkpoints/last.ckpt \\
        --config configs/qwen_mlp_config.yaml \\
        --task narrative \\
        --source "Paste your source text here"

    For QA:
    python infer.py --checkpoint ... --config ... \\
        --task qa \\
        --source "The capital of France is Paris." \\
        --question "What is the capital of France?"

File mode (CSV / JSONL / Parquet — must have 'source_text' and 'task' columns):
    python infer.py \\
        --checkpoint runs/.../last.ckpt \\
        --config configs/qwen_mlp_config.yaml \\
        --input_file data/my_examples.csv \\
        --output_file predictions.csv \\
        --batch_size 16 \\
        --device cuda:0
"""

import argparse
import sys

import pandas as pd

from src.pipeline import load_pipeline, predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    required = {"source_text", "task"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"Input file is missing required columns: {missing}")
    q_col = df["question"] if "question" in df.columns else pd.Series([""] * len(df))
    return [
        {
            "source_text": str(row["source_text"]),
            "task":        str(row["task"]),
            "question":    str(q_col.iloc[i] or ""),
        }
        for i, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a trained mapper-LLM checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .ckpt checkpoint (e.g. runs/.../last.ckpt)")
    parser.add_argument("--config", required=True,
                        help="YAML config used during training")

    # Common options
    parser.add_argument("--device", default="cpu",
                        help='Device: "cpu" | "cuda:0" | "cuda:1" | … (default: cpu)')
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate per example (default: 128)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--repetition_penalty", type=float, default=1.3)

    # Manual-text mode
    parser.add_argument("--source",
                        help="Source text to feed to the embedder (manual mode)")
    parser.add_argument("--task", choices=["narrative", "qa"], default="narrative",
                        help="Task type for manual mode (default: narrative)")
    parser.add_argument("--question", default="",
                        help="Question text for QA task (manual mode)")

    # File mode
    parser.add_argument("--input_file",
                        help="CSV / JSONL / Parquet file for batch inference")
    parser.add_argument("--output_file",
                        help="Output CSV path; omit to print results to stdout")

    args = parser.parse_args()

    if args.source is None and args.input_file is None:
        parser.error("Provide either --source (manual mode) or --input_file (file mode).")

    # ── Load pipeline ────────────────────────────────────────────────────────
    print(f"Loading pipeline from: {args.checkpoint}")
    module, emb_tok, llm_tok = load_pipeline(args.checkpoint, args.config, args.device)
    print(f"Pipeline ready. Device: {args.device}\n")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        repetition_penalty=args.repetition_penalty,
    )

    # ── Manual mode ─────────────────────────────────────────────────────────
    if args.source is not None:
        records = [{"source_text": args.source,
                    "task": args.task,
                    "question": args.question}]
        preds = predict(module, emb_tok, llm_tok, records, **gen_kwargs)
        print("=== Prediction ===")
        print(preds[0])
        return

    # ── File mode ────────────────────────────────────────────────────────────
    print(f"Reading input file: {args.input_file}")
    df      = _read_file(args.input_file)
    records = _df_to_records(df)
    print(f"Running inference on {len(records)} examples …")

    preds        = predict(module, emb_tok, llm_tok, records, **gen_kwargs)
    df           = df.copy()
    df["prediction"] = preds

    if args.output_file:
        df.to_csv(args.output_file, index=False)
        print(f"\nSaved {len(df)} predictions → {args.output_file}")
    else:
        cols = [c for c in ["source_text", "task", "question", "answer", "prediction"]
                if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
