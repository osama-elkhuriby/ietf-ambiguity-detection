# GPU and numerical-stability preflight

Status: **PASS** on 2026-10-20 (Africa/Cairo).

- Python: 3.11.16 at `C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe`
- GPU: NVIDIA RTX 5000 Ada Generation
- VRAM at check: 32,760 MiB total; 30,269 MiB free
- Driver: 596.59
- PyTorch: 2.12.0; CUDA runtime: 13.0; cuDNN: 92501
- Transformers 5.16.1; tokenizers 0.23.1; sentencepiece 0.2.1;
  safetensors 0.8.0; NumPy 2.4.6; pandas 3.0.5; scikit-learn 1.9.0
- Transformers model/tokenizer imports and training-core import: PASS
- Approved local model exists and hashes to
  `206ee7bb2c7805bcc8fd3bbbc18a09e185bafc19d4e112f31d4a088136717dd7`
- Model/tokenizer weights loaded during preflight: no
- Training, backward passes, optimizer steps, and downloads during preflight: no

Preserved safeguards include FP32 cross-entropy/log-softmax, binary-label and
batch alignment checks, finite logits/loss checks, diagnostic backward context
with epoch/batch/optimizer-step identifiers, non-finite gradient rejection,
gradient clipping at 1.0, finite post-update parameters and optimizer state,
offline local-only loading, and fail-fast failure manifests. FP32 is retained
because it is the validated baseline and there is no evidence-based reason to
change it; historical instability was not attributed solely to BF16.

Historical calibration runs on this GPU used about 1.7-4.0 GiB allocated VRAM
and took 56-336 seconds per cell (30-cell mean 196 seconds; total 98 minutes).
For 40 fresh cells, a planning estimate is roughly 2-4 hours sequentially,
excluding investigation time. Forty final checkpoints at the base weight size
imply about 13.8 GiB minimum weight storage, plus logs and metadata. These are
planning estimates, not guarantees for the revised cohort/environment.

