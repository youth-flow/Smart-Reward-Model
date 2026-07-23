# HPC4 image identity

This directory defines the only supported environment for the controlled HPC4
experiment. It contains no model or dataset weights.

The host gate was executed on HKUST HPC4 job `1640437` on `gpu-l20`. The
observed device was an NVIDIA L20 with 46,068 MiB, compute capability 8.9, and
driver `570.211.01` (`nvidia-smi` maximum CUDA 12.8). That evidence fixes the
candidate base to:

```text
docker.io/pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3
```

CUDA 12.6 is a stable PyTorch 2.7.1 build and is supported by the observed
driver. `transformers==4.52.3` is exact because the pinned Skywork configuration
identifies `Qwen3ForSequenceClassification` at that version. PEFT, Datasets,
Arrow, NumPy, tokenizer, Hub and all introduced transitive packages are also
exact-pinned in `requirements-hpc4.lock`. Torch is supplied by the digest-locked
base and is never re-resolved by pip.

## Build and publication

`.github/workflows/build-hpc4-image.yml` builds the SIF with Apptainer 1.5.2 on
an isolated GitHub runner, runs the definition test and `pip check`, records a
sorted `pip freeze`, and publishes the raw SIF to:

```text
oras://ghcr.io/youth-flow/smart-reward-model-hpc4:git-<40-hex-commit>
```

The workflow also proves that the resulting manifest is anonymously readable.
HPC4 cannot locally build this definition: SUID installation is disabled, no
subuid/subgid mappings exist, and user namespaces are disabled. Pulling the raw
SIF over ORAS therefore preserves the exact bytes built by the workflow.

On HPC4, fetch by the immutable Git build identity:

```bash
source .env.hpc4
bash scripts/hpc4/fetch_candidate_image.sh <40-hex-image-build-commit>
```

The fetcher resolves the tag to an OCI manifest digest, hashes the exact
manifest bytes, pulls by digest, and requires the downloaded SIF SHA256 to equal
the manifest's sole SIF-layer digest. It never replaces a different existing
image.

The downloaded image remains a **candidate**. It becomes
**HPC4-validated** only when `submit_gpu_smoke.sh gpu-l20` completes with exit
code zero and persists its report.
