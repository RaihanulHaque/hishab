"""Shared Modal objects: app, volumes, images.

Run every entrypoint from the repo root with the ML conda env, e.g.
  conda run -n ML modal run ml/modal_app/train.py --models qwen3.5-0.8b --fmts json
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]
HISHAB_SRC = REPO / "ml" / "hishab"
MODAL_APP_SRC = REPO / "ml" / "modal_app"

APP_NAME = "hishab"
DATA_DIR = "/data"      # splits
RUNS_DIR = "/runs"      # checkpoints, merged models, ggufs, eval json
HF_CACHE = "/hf-cache"

# kiln pins this llama.cpp commit (third_party/llama.cpp); GGUFs are converted and evaluated with
# the same code the phone runs.
LLAMA_CPP_COMMIT = "781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4"

data_vol = modal.Volume.from_name("hishab-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("hishab-runs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hishab-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")  # exposes HF_TOKEN

VOLUMES = {DATA_DIR: data_vol, RUNS_DIR: runs_vol, HF_CACHE: hf_cache_vol}
HF_ENV = {"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "transformers==5.10.2",
        "trl==1.5.1",
        "peft==0.19.1",
        "datasets>=4.0",
        "accelerate>=1.10",
        "huggingface_hub[hf_transfer]>=0.35",
        "sentencepiece",
        "protobuf",
        "tiktoken",
    )
    .env(HF_ENV)
    .add_local_dir(HISHAB_SRC, remote_path="/root/hishab")
    .add_local_dir(MODAL_APP_SRC, remote_path="/root/modal_app")
)

# CPU image with llama.cpp built from kiln's commit: convert_hf_to_gguf, llama-quantize,
# llama-imatrix, llama-server.
gguf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev", "pkg-config")
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        f"cd /opt/llama.cpp && git checkout {LLAMA_CPP_COMMIT}",
        "cd /opt/llama.cpp && cmake -S . -B build -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON "
        "-DGGML_F16C=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "
        "-DCMAKE_BUILD_TYPE=Release",
        "cd /opt/llama.cpp && cmake --build build --config Release -j "
        "--target llama-quantize llama-imatrix llama-server llama-cli",
        "cd /opt/llama.cpp && pip install --extra-index-url https://download.pytorch.org/whl/cpu "
        "-r requirements/requirements-convert_hf_to_gguf.txt",
        # the requirements file pins transformers 4.x, which cannot read tokenizer_config.json
        # written by transformers 5 ("TokenizersBackend"); use the training version instead
        "pip install 'transformers==5.10.2' 'huggingface_hub>=0.35' requests",
    )
    .env(HF_ENV)
    .add_local_dir(HISHAB_SRC, remote_path="/root/hishab")
    .add_local_dir(MODAL_APP_SRC, remote_path="/root/modal_app")
)

app = modal.App(APP_NAME)
