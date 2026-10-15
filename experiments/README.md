# experiments

Scratch work that preceded the Modal pipeline in `ml/`. Nothing here is used by the pipeline;
it is kept because the two TRL notebooks produced the models currently published on Hugging
Face. New work belongs in `ml/`.

| file | what it is | outcome |
|---|---|---|
| `hishab-sft-qwen35-trl.ipynb` | TRL LoRA SFT of `Qwen/Qwen3.5-0.8B` on `ryuma007/hishab-synthetic-v2` (two_phase format, 7 epochs, A100), merge, push, GGUF, LiteRT attempt | Produced `ryuma007/qwen3.5-0.8b-hishab-synthetic-v2` and its `-Q4_K_M-GGUF`. Its "42% semantic match" was measured on a validation split contained in train; see `ml/results/RESULTS.md` for honest numbers. |
| `hishab-sft-lfm2-5-trl.ipynb` | Same recipe with `LiquidAI/LFM2.5-350M` in 4-bit QLoRA, 5 epochs | Produced `ryuma007/lfm2.5-350M-hishab-synthetic-v2x`. |
| `Hishab_SFT_Qwen3_600M.ipynb` | Unsloth (MLX) Qwen3-0.6B SFT with Banglish token additions | Early attempt; superseded. |
| `unsloth_test_model.ipynb`, `unsloth_test.py` | Loading Gemma-4 E2B with Unsloth and counting token frequencies on the pretraining corpus | Exploration for the tokenizer-pruning idea in `docs/HANDOFF_2026-01_tokenizer_expansion.md`. |
| `qwen_sample_finetune.py` | Unsloth's Qwen3.5-0.8B vision Colab exported to a script | Reference only. |
| `synthetic_dataset_pipeline.py` | First Gemini synthetic-data generator (pydantic schema, validation) | Superseded by `ml/scripts/append_synthetic_dataset.py`. |
| `test2.py`, `rpm_test.py` | Gemini structured-output and rate-limit probes | Throwaway. |

Notebooks read the Hugging Face token from `HF_TOKEN` in the environment. Do not paste tokens
into cells; the repo is public.
