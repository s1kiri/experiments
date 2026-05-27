#!/usr/bin/env python3
"""
baseline_infer.py — запускает базовую Qwen3-0.6B на val-сете и считает метрики.

Берёт CSV с предсказаниями дообученной модели (колонка 'prediction'),
добавляет колонку 'baseline_prediction' и выводит сравнение ROUGE-L / BLEU-4
по задачам.

Использование:
  python baseline_infer.py
  python baseline_infer.py --val_csv artefacts/predictions_val_set_2500.csv \\
                           --val_samples 1000 \\
                           --out artefacts/baseline_val_1000.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from src.metrics import bleu4, rouge_l

DEFAULT_MODEL    = "model_cache/Qwen--Qwen3-0.6B"
DEFAULT_VAL_CSV  = "artefacts/predictions_val_set_2500.csv"
DEFAULT_OUT      = "artefacts/baseline_val_1000.csv"
DEFAULT_N        = 1000
DEFAULT_CTX_TOK  = 128
DEFAULT_MAX_NEW  = 150


# ── helpers ───────────────────────────────────────────────────────────────────

def truncate_to_tokens(text: str, tokenizer, n: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:n]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prompt(row: dict, tokenizer, ctx_tokens: int) -> str:
    src = truncate_to_tokens(str(row["source_text"]), tokenizer, ctx_tokens)

    if str(row["task"]) == "narrative":
        # Задача: реконструировать оригинальный текст по его началу.
        # Системный промпт запрещает лишние слова — только текст.
        messages = [
            {"role": "system",
             "content": "Reproduce the text verbatim. Output only the text itself, no commentary. Don't use additional characters or symbols for any purpose. You must reproduce the given text exactly as it is."},
            {"role": "user",
             "content": f"Reproduce the following text verbatim:\n\n{src}"},
        ]
    else:
        q = str(row.get("question") or "")
        messages = [
            {"role": "system",
             "content": "Answer concisely using only the given context. Output only the answer, no explanation. Extract the most relevant chunck of the given text and use it as an answer. Use the exact words from the given text. No answers more than 15 words, be consice"},
            {"role": "user",
             "content": f"Context: {src}\n\nQuestion: {q}"},
        ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_one(prompt: str, model, tokenizer, max_new_tokens: int,
                 device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference(df: pd.DataFrame, model, tokenizer,
                  ctx_tokens: int, max_new_tokens: int, device: str) -> list[str]:
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="baseline inference"):
        prompt = build_prompt(row.to_dict(), tokenizer, ctx_tokens)
        preds.append(generate_one(prompt, model, tokenizer, max_new_tokens, device))
    return preds


def _trunc(text: str, n: int) -> str:
    """Обрезает до первых n пробельных токенов."""
    return " ".join(str(text).split()[:n])


def compute_metrics(df: pd.DataFrame, pred_col: str,
                    max_eval_words: int = 128) -> pd.DataFrame:
    """Считает ROUGE-L и BLEU-4 по задачам + итог.

    Оба текста (pred и ref) обрезаются до max_eval_words слов перед подсчётом —
    это убирает штраф за краткость (BP) в BLEU для narrative-задачи, где
    модель генерирует ~128 токенов, а референс — полный текст (300-500 слов).
    """
    rows = []
    for task in sorted(df["task"].dropna().unique()):
        sub = df[df["task"] == task]
        preds = [_trunc(p, max_eval_words) for p in sub[pred_col].fillna("")]
        refs  = [_trunc(r, max_eval_words) for r in sub["answer"].fillna("")]
        r = rouge_l(preds, refs)
        b = bleu4(preds, refs)
        rows.append({
            "task":    task,
            "n":       len(sub),
            "ROUGE-L": round(r["rougeL"], 4),
            "BLEU-4":  round(b["bleu"],   6),
        })
    # Overall
    preds_all = [_trunc(p, max_eval_words) for p in df[pred_col].fillna("")]
    refs_all  = [_trunc(r, max_eval_words) for r in df["answer"].fillna("")]
    r = rouge_l(preds_all, refs_all)
    b = bleu4(preds_all, refs_all)
    rows.append({
        "task":    "OVERALL",
        "n":       len(df),
        "ROUGE-L": round(r["rougeL"], 4),
        "BLEU-4":  round(b["bleu"],   6),
    })
    return pd.DataFrame(rows).set_index("task")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline eval с базовой Qwen3-0.6B")
    p.add_argument("--val_csv",        default=DEFAULT_VAL_CSV)
    p.add_argument("--val_samples",    type=int, default=DEFAULT_N)
    p.add_argument("--out",            default=DEFAULT_OUT)
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--ctx_tokens",     type=int, default=DEFAULT_CTX_TOK)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW)
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Загрузка и стратифицированный сэмпл ────────────────────────────
    print(f"▶ Загружаю {args.val_csv} ...")
    df_full = pd.read_csv(args.val_csv)

    n = args.val_samples
    df = (df_full
          .groupby("task", group_keys=False)
          .apply(lambda g: g.sample(
              n=round(n * len(g) / len(df_full)),
              random_state=42))
          .reset_index(drop=True))

    task_counts = df["task"].value_counts().to_dict()
    print(f"  Сэмпл: {len(df)} строк | {task_counts}")

    # ── 2. Метрики дообученной модели (уже есть в CSV) ────────────────────
    if "prediction" in df.columns:
        print("\n▶ Метрики дообученной модели (колонка 'prediction'):")
        ft_metrics = compute_metrics(df, "prediction")
        print(ft_metrics.to_string())
    else:
        print("  Колонка 'prediction' не найдена — пропускаю метрики FT-модели.")
        ft_metrics = None

    # ── 3. Загрузка базовой модели ────────────────────────────────────────
    print(f"\n▶ Загружаю базовую модель: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Параметров: {n_params:.0f}M | device: {args.device} | dtype: {dtype}")

    # ── 4. Инференс ───────────────────────────────────────────────────────
    print(f"\n▶ Инференс baseline на {len(df)} примерах ...")
    df["baseline_prediction"] = run_inference(
        df, model, tokenizer, args.ctx_tokens, args.max_new_tokens, args.device
    )

    # ── 5. Метрики baseline ───────────────────────────────────────────────
    print("\n▶ Метрики baseline (колонка 'baseline_prediction'):")
    bl_metrics = compute_metrics(df, "baseline_prediction")
    print(bl_metrics.to_string())

    # ── 6. Сравнение ──────────────────────────────────────────────────────
    if ft_metrics is not None:
        print("\n▶ Сравнение (finetuned vs baseline):")
        cmp = pd.DataFrame({
            "FT ROUGE-L":  ft_metrics["ROUGE-L"],
            "BL ROUGE-L":  bl_metrics["ROUGE-L"],
            "Δ ROUGE-L":   (ft_metrics["ROUGE-L"] - bl_metrics["ROUGE-L"]).round(4),
            "FT BLEU-4":   ft_metrics["BLEU-4"],
            "BL BLEU-4":   bl_metrics["BLEU-4"],
            "Δ BLEU-4":    (ft_metrics["BLEU-4"] - bl_metrics["BLEU-4"]).round(6),
        })
        print(cmp.to_string())

    # ── 7. Сохранение ─────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✓ Результаты сохранены → {out_path}")
    print(f"  Колонки: {list(df.columns)}")

    # ── 8. HTML-просмотрщики ──────────────────────────────────────────────
    viewer = Path(__file__).parent / "predict_viewer.py"
    if not viewer.exists():
        print("  predict_viewer.py не найден — пропускаю HTML.")
        return

    comparison_dir = Path("artefacts/comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem  # e.g. "baseline_val_1000"

    # viewer ожидает колонку "prediction" — делаем временные CSV с нужным именем
    for col, label in [("baseline_prediction", f"viewer_{stem}_baseline.html"),
                       ("prediction",           f"viewer_{stem}_finetuned.html")]:
        if col not in df.columns:
            continue
        html_path = comparison_dir / label
        tmp_csv   = out_path.with_suffix(f".{col}.tmp.csv")
        df_view   = df.copy()
        df_view["prediction"] = df_view[col]
        df_view.to_csv(tmp_csv, index=False)
        print(f"  Генерирую {html_path.name} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, str(viewer), str(tmp_csv),
             "-o", str(html_path), "--max-cutoff", "256"],
            capture_output=True, text=True,
        )
        tmp_csv.unlink()
        if result.returncode != 0:
            print(f"ОШИБКА\n{result.stderr}")
        else:
            print("готово")


if __name__ == "__main__":
    main()
