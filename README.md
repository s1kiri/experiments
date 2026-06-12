# GRAFT: Интеграция внешних компонентов в языковые модели через непрерывные представления

> **Передача внешнего контекста в языковую модель через k непрерывных векторов**  


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/pytorch-2.8-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-models%20%26%20dataset-yellow)](https://huggingface.co/supersoska)

---

## Идея

Стандартный RAG передаёт контекст в LLM как текстовые токены — это дорого при масштабировании.  
Данная работа проверяет альтернативу: **сжать контекст в k векторов через обучаемый модуль (маппер) и подать их напрямую во входной слой LLM**, минуя токенизацию.

```
source_text ──► Embedder ──► Mapper ──► k vectors ──► LLM ──► ответ
                 (frozen)   (learned)   (k=1..32)   (frozen / fine-tuned)
```

Система достигает **BLEU-4 = 0.781** при 15-кратном сжатии контекста (16 векторов вместо ~240 токенов) — выше baseline с полным текстом в промпте (0.622).

---

## Архитектура

Три компонента соединены последовательно:

| Компонент | Модель | Роль |
|---|---|---|
| **Embedder** | `Qwen/Qwen3-Embedding-0.6B` | Кодирует source_text в последовательность векторов |
| **Mapper** | Linear / MLP / Transformer | Агрегирует векторы энкодера в k токенов контекста |
| **LLM** | `Qwen/Qwen3-{0.6,4,8}B` | Генерирует ответ, получая k векторов как мягкий контекст |

### Архитектуры маппера

| Маппер | Механизм агрегации | Параметры |
|---|---|---|
| **LinearMapper** | Chunk-mean pooling + линейная проекция | ~1M |
| **MLPMapper** | Chunk-mean pooling + MLP с SiLU | ~3.1M |
| **TransformerMapper** | k обучаемых запросов + cross-attention | ~33.6M |

### Конфигурации обучения

| Конфигурация | Что обучается |
|---|---|
| `mapper_only` | только маппер; энкодер и LLM заморожены |
| `map+LoRA` | маппер + LoRA-адаптеры LLM (r=16) |
| `soft-prompt` | маппер + 16 обучаемых токенов-префиксов |
| `full_ft` | все три компонента совместно |

---

## Результаты

### Влияние конфигурации обучения (TransformerMapper, k=16)

| Конфигурация | Narrative BLEU@64 | QA ROUGE-L@64 |
|---|---|---|
| mapper_only | 0.171 | 0.112 |
| map+LoRA | 0.290 | 0.170 |
| soft-prompt | 0.163 | — |
| **full_ft** | **0.743** | **0.418** |

### Сравнение с baseline (full_ft, k=16 vs полный текст в промпте)

| Метрика | Система (k=16) | Baseline |
|---|---|---|
| Narrative BLEU-4 | **0.781** | 0.622 |
| Narrative ROUGE-L | **0.888** | 0.756 |
| QA BLEU-4 | **0.431** | 0.032 |
| QA ROUGE-L | **0.598** | 0.255 |

Система с 16 векторами превосходит необученную модель, которая видит весь текст в токенах.

---

## Данные и модели на HuggingFace

| Артефакт | Ссылка | Размер |
|---|---|---|
| Датасет (791k примеров) | [supersoska/mapper-llm-dataset](https://huggingface.co/datasets/supersoska/mapper-llm-dataset) | 3.3 GB |
| Чекпоинт · mapper_only | [supersoska/mapper-llm-perceiver-ctx16-mapper-only](https://huggingface.co/supersoska/mapper-llm-perceiver-ctx16-mapper-only) | 2.6 GB |
| Чекпоинт · full_ft | [supersoska/mapper-llm-perceiver-ctx16-full-ft](https://huggingface.co/supersoska/mapper-llm-perceiver-ctx16-full-ft) | 7.0 GB |

---

## Структура репозитория

```
experiments/
├── src/
│   ├── models/
│   │   ├── embedder.py          # HFEmbedder
│   │   ├── mapper.py            # LinearMapper / MLPMapper / TransformerMapper
│   │   ├── llm.py               # HFLLM
│   │   └── lightning_module.py  # MapperLLMModule (PyTorch Lightning)
│   ├── data/
│   │   ├── data_module.py       # TextDataModule
│   │   └── schema.py            # схема датасета, SPECIAL_TOKENS
│   ├── metrics.py
│   └── pipeline.py              # load_pipeline / predict (инференс)
├── configs/
│   ├── perceiver/
│   │   ├── mapper_only/         # MO_per_ctx{1,4,8,16,32}.yaml
│   │   └── llm/                 # конфиги LoRA и soft-prompt
│   ├── linear/                  # mapper_only и full_ft для Linear
│   └── mlp/                     # mapper_only и full_ft для MLP
├── scripts/
│   ├── download_models.py       # скачать веса Qwen3 в model_cache/
│   └── upload_to_hf.py          # загрузить чекпоинты и датасет на Hub
├── train.py                     # обучение
├── infer.py                     # инференс из checkpoint
├── predict_viewer.py            # HTML-просмотрщик предсказаний
├── requirements.txt             # зависимости (pip)
├── pyproject.toml               # зависимости (Poetry, для dev)
└── Dockerfile / Dockerfile.cpu
```

---

## Схема датасета

Датасет собран из Cosmopedia (нарратив) и SQuAD 1.1/2.0 (QA), 791k примеров.  
Формат HuggingFace `datasets`, сплиты `train` / `validation`.

| Колонка | Тип | Описание |
|---|---|---|
| `id` | str | уникальный идентификатор |
| `split` | str | `"train"` / `"validation"` |
| `task` | str | `"narrative"` / `"qa"` |
| `source_text` | str | текст, подаваемый в embedder |
| `question` | str | вопрос (только для `qa`; `""` для `narrative`) |
| `answer` | str | целевой ответ / воспроизводимый текст |

---

## Установка

### pip (GPU, CUDA 12.1)
```bash
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### pip (CPU / Mac)
```bash
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Docker (CPU, для тестирования)
```bash
docker build -f Dockerfile.cpu -t mapper-llm-cpu .
docker run mapper-llm-cpu --help
```

### Docker (GPU, продакшн)
```bash
docker build -t mapper-llm .
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/model_cache:/workspace/model_cache \
  -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  mapper-llm --config configs/perceiver/mapper_only/MO_per_ctx16.yaml
```

---

## Подготовка

### Веса базовых моделей
```bash
python scripts/download_models.py   # скачивает Qwen3-Embedding-0.6B и Qwen3-0.6B в model_cache/
```

### Датасет
Готовится через `prepare_datasets.ipynb` или загружается с Hub:
```python
from datasets import load_dataset
ds = load_dataset("supersoska/mapper-llm-dataset")
ds.save_to_disk("data/unified_dataset")
```

### Готовые чекпоинты
```bash
# full_ft (лучший результат)
huggingface-cli download supersoska/mapper-llm-perceiver-ctx16-full-ft \
    last.ckpt --local-dir checkpoints/perceiver_ctx16_full_ft

# mapper_only (только маппер, LLM заморожен)
huggingface-cli download supersoska/mapper-llm-perceiver-ctx16-mapper-only \
    last.ckpt --local-dir checkpoints/perceiver_ctx16_mapper_only
```

---

## Обучение

```bash
# TransformerMapper k=16, только маппер
python train.py --config configs/perceiver/mapper_only/MO_per_ctx16.yaml

# TransformerMapper k=16, полное дообучение
python train.py --config configs/full_ft_perceiver.yaml

# LinearMapper k=16, full_ft, другой датасет
python train.py --config configs/linear/full_ft_lin_ctx16.yaml \
                --data data/my_dataset
```

Логи и чекпоинты → `runs/<experiment_name>/version_N/`  
Мониторинг: `tensorboard --logdir runs/`

---

## Инференс

```bash
# Батчевый инференс на CSV
python infer.py \
  --checkpoint checkpoints/perceiver_ctx16_full_ft/last.ckpt \
  --config configs/full_ft_perceiver.yaml \
  --input_file artefacts/val_100_comparison_set.csv \
  --output_file artefacts/preds.csv \
  --max_new_tokens 512 --device cuda:0

# Разовый запрос
python infer.py \
  --checkpoint checkpoints/perceiver_ctx16_full_ft/last.ckpt \
  --config configs/full_ft_perceiver.yaml \
  --task narrative --source "Вставьте текст сюда"

# QA
python infer.py \
  --checkpoint checkpoints/perceiver_ctx16_full_ft/last.ckpt \
  --config configs/full_ft_perceiver.yaml \
  --task qa \
  --source "Paris is the capital of France." \
  --question "What is the capital of France?"
```

### Просмотр результатов
```bash
python predict_viewer.py artefacts/preds.csv \
    --cutoff 128 --max-cutoff 512 -o viewer.html
open viewer.html
```

Интерактивный HTML-viewer с пословным сравнением предсказаний и референса, слайдером длины и метриками.

### Кодовые ассистенты
Для написания части кода проекта использовался Claude Code с Sonnet 4.5.