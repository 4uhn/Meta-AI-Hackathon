"""Horizon: federated FFA-LoRA fine-tuning with t-of-N threshold CKKS aggregation."""

import os

# MPS: fail fast on out-of-memory instead of swapping the whole machine. Must be set before torch
# initialises the MPS allocator, hence here. The low ratio (default 1.4 on unified memory) must not
# exceed the high one, or MPS refuses to start.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

# CUDA: deterministic cuBLAS, so the plaintext and encrypted runs differ only by CKKS error. Read at init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
