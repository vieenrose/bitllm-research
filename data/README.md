# Data

Raw corpora and packed token shards are **git-ignored** — they live on the
training box, not in the repo.

## Pipeline

1. **Train the tokenizer** on a sample of raw zh + en text:
   ```bash
   python scripts/train_tokenizer.py \
       --zh_files data/raw/zh/*.txt --en_files data/raw/en/*.txt \
       --vocab_size 48000 --out tokenizer/bitllm-zhen.json
   ```
2. **Pack** the full weighted mixture into flat token shards:
   ```bash
   python scripts/prepare_data.py --config configs/small_60m.yaml
   ```
   → writes `data/packed/{lang}_{source}.bin` (+ `.meta`).

## Recommended zh/en corpora

| Lang | Dataset | Notes |
|------|---------|-------|
| en | `HuggingFaceFW/fineweb-edu` | high-quality filtered English web (`sample-10BT` for a start) |
| en | `HuggingFaceFW/fineweb` | broader English web if you need volume |
| zh | `BAAI/CCI3-HQ` | high-quality Chinese web (CCI 3.0 HQ) |
| zh | `BAAI/CCI3-Data` | larger Chinese web corpus |
| zh | `opencsg/chinese-fineweb-edu` | educational-filtered Chinese |
| zh | `Skywork/SkyPile-150B` | large Chinese web (SkyPile) |
| zh/en | `cerebras/SlimPajama-627B` | English-heavy but clean, for dedup baseline |

Verify each dataset's exact name/config/split on the Hub before use — Hub ids
drift. The `weight` field in the config sets the zh:en sampling ratio per batch;
start at 0.5/0.5 by token count and adjust from per-language eval perplexity.

## Balancing note

Chinese and English tokenize at different fertilities (tokens per character vs
per word). To balance *content* rather than *tokens*, measure tokens-per-source
after packing (printed by `prepare_data.py`) and set weights so each language
contributes the intended share of gradient steps.
