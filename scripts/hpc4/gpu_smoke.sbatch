#!/usr/bin/env bash
#SBATCH --job-name=prorm-gpu-smoke
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00

set -euo pipefail
umask 027

resolve_compat_env() {
  local canonical="$1" legacy="$2" canonical_value="" legacy_value="" resolved=""
  local canonical_set=0 legacy_set=0
  if [[ -v "${canonical}" ]]; then canonical_set=1; canonical_value="${!canonical}"; fi
  if [[ -v "${legacy}" ]]; then legacy_set=1; legacy_value="${!legacy}"; fi
  if (( canonical_set && legacy_set )) && [[ "${canonical_value}" != "${legacy_value}" ]]; then
    echo "conflicting ${canonical} and legacy ${legacy}" >&2
    exit 2
  fi
  if (( canonical_set )); then resolved="${canonical_value}"; else resolved="${legacy_value}"; fi
  printf -v "${canonical}" '%s' "${resolved}"
  printf -v "${legacy}" '%s' "${resolved}"
  export "${canonical}" "${legacy}"
}

die() {
  echo "error: $*" >&2
  exit 2
}

reject_apptainer_control_environment() {
  local variable=""
  while IFS= read -r variable; do
    case "${variable}" in
      APPTAINER*|SINGULARITY*)
        die "unset exported ${variable}; GPU smoke forbids ambient container controls"
        ;;
    esac
  done < <(compgen -e)
}

normalize_absolute_root() {
  local canonical="$1" legacy="$2" raw="${!1}" resolved=""
  [[ -n "${raw}" ]] || die "${canonical} must be set"
  [[ "${raw}" = /* ]] || die "${canonical} must be an absolute path: ${raw}"
  if ! resolved="$(realpath -e -- "${raw}")"; then
    die "${canonical} does not exist or cannot be resolved: ${raw}"
  fi
  [[ -d "${resolved}" ]] || die "${canonical} is not a directory: ${resolved}"
  [[ "${resolved}" != "/" ]] || die "${canonical} may not be the filesystem root"
  printf -v "${canonical}" '%s' "${resolved}"
  printf -v "${legacy}" '%s' "${resolved}"
  export "${canonical}" "${legacy}"
}

roots_overlap() {
  local first="$1" second="$2"
  case "${first}" in
    "${second}"|"${second}"/*) return 0 ;;
  esac
  case "${second}" in
    "${first}"|"${first}"/*) return 0 ;;
  esac
  return 1
}

reject_bind_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || {
    die "${name} may not contain commas or newlines"
  }
  [[ "${value}" != *":"* ]] || die "${name} may not contain ':' in an Apptainer bind path"
}

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
resolve_compat_env PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
resolve_compat_env PRORM_IMAGE SRM_IMAGE
resolve_compat_env PRORM_IMAGE_SHA256 SRM_IMAGE_SHA256
resolve_compat_env PRORM_REPO_ROOT SRM_REPO_ROOT
: "${PRORM_PROJECT_ROOT:?absolute persistent project root is required}"
: "${PRORM_SCRATCH_ROOT:?absolute per-user scratch root is required}"
: "${PRORM_IMAGE:?set PRORM_IMAGE to the absolute path of the validated Apptainer image}"
: "${PRORM_IMAGE_SHA256:?set PRORM_IMAGE_SHA256 to the expected lowercase SHA256}"
: "${PRORM_REPO_ROOT:?set PRORM_REPO_ROOT to the absolute repository root}"
: "${SLURM_JOB_ID:?GPU smoke must execute inside a Slurm job}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || {
  die "SLURM_JOB_ID must be a positive integer"
}
reject_apptainer_control_environment

normalize_absolute_root PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
normalize_absolute_root PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "PRORM_PROJECT_ROOT and PRORM_SCRATCH_ROOT may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "PRORM_PROJECT_ROOT is not writable: ${PRORM_PROJECT_ROOT}"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "PRORM_SCRATCH_ROOT is not writable: ${PRORM_SCRATCH_ROOT}"
[[ "${PRORM_REPO_ROOT}" = /* ]] || die "PRORM_REPO_ROOT must be absolute"
[[ -d "${PRORM_REPO_ROOT}/.git" ]] || {
  die "PRORM_REPO_ROOT is not a Git worktree root: ${PRORM_REPO_ROOT}"
}
[[ "${PRORM_IMAGE}" = /* ]] || {
  die "submit script must resolve PRORM_IMAGE to an absolute path"
}
[[ -f "${PRORM_IMAGE}" ]] || die "PRORM_IMAGE is not a file: ${PRORM_IMAGE}"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  die "PRORM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters"
}
if ! printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status; then
  die "PRORM_IMAGE_SHA256 does not match PRORM_IMAGE: ${PRORM_IMAGE}"
fi

repo_root="${PRORM_REPO_ROOT}"
for bind_name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_REPO_ROOT; do
  reject_bind_delimiters "${bind_name}" "${!bind_name}"
done

container_visibility=()
for variable in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    container_visibility+=(--env "${variable}=${!variable}")
  fi
done

job_dir="${PRORM_SCRATCH_ROOT}/jobs/${SLURM_JOB_ID}"
mkdir -p "${job_dir}" "${PRORM_PROJECT_ROOT}/system-reports"

report="${job_dir}/gpu-smoke-${SLURM_JOB_ID}.txt"
persisted_report="${PRORM_PROJECT_ROOT}/system-reports/gpu-smoke-${SLURM_JOB_ID}.txt"
finalize_report() {
  local workload_exit=$? final_exit temporary=""
  final_exit="${workload_exit}"
  trap - EXIT
  set +e
  if [[ ! -f "${report}" ]]; then
    echo "GPU smoke report was not created: ${report}" >&2
    exit 1
  fi
  if [[ -e "${persisted_report}" || -L "${persisted_report}" ]]; then
    echo "refusing to overwrite existing GPU smoke report: ${persisted_report}" >&2
    exit 1
  fi
  temporary="$(mktemp \
    "${PRORM_PROJECT_ROOT}/system-reports/.gpu-smoke-${SLURM_JOB_ID}.XXXXXX")" \
    || exit 1
  if ! cp -p -- "${report}" "${temporary}"; then
    rm -f -- "${temporary}"
    exit 1
  fi
  if ! {
    printf 'workload_exit_code=%s\n' "${workload_exit}"
    if (( workload_exit == 0 )); then
      printf 'status=passed\n'
    else
      printf 'status=failed\n'
    fi
  } >> "${temporary}"; then
    rm -f -- "${temporary}"
    exit 1
  fi
  if ! mv -T -- "${temporary}" "${persisted_report}"; then
    rm -f -- "${temporary}"
    final_exit=1
  fi
  exit "${final_exit}"
}
trap finalize_report EXIT
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "git_commit=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "image_sha256=${PRORM_IMAGE_SHA256}"
  hostname
  nvidia-smi
  echo "== allowlisted Slurm identity =="
  for variable in \
    SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
    SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION \
    SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_CPUS_PER_TASK \
    SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_JOB_GPUS \
    SLURM_PROCID SLURM_LOCALID SLURM_NODEID; do
    if [[ -v "${variable}" ]]; then
      printf '%s=%s\n' "${variable}" "${!variable}"
    fi
  done
} | tee "${report}"

apptainer exec --cleanenv --nv \
  --bind "${repo_root}:${repo_root}" \
  --env "PYTHONPATH=${repo_root}/src" \
  "${container_visibility[@]}" \
  "${PRORM_IMAGE}" \
  python - <<'PY' | tee -a "${report}"
import sys
from importlib.metadata import version

import torch
import transformers
from datasets import DownloadConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import load, save
from transformers import (
    AutoModelForSequenceClassification,
    GenerationConfig,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForSequenceClassification,
)

print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("transformers", transformers.__version__)
for package in (
    "accelerate",
    "datasets",
    "huggingface-hub",
    "numpy",
    "peft",
    "pyarrow",
    "safetensors",
    "tokenizers",
):
    print(package, version(package))
expected = {
    "accelerate": "1.7.0",
    "datasets": "3.6.0",
    "huggingface-hub": "0.31.4",
    "numpy": "1.26.4",
    "peft": "0.15.2",
    "pyarrow": "17.0.0",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.1",
    "transformers": "4.52.3",
}
observed = {name: version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"GPU smoke failed: package lock mismatch: {observed!r}")
if version("torch").split("+", 1)[0] != "2.7.1":
    raise SystemExit(f"GPU smoke failed: expected torch 2.7.1, got {version('torch')}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"GPU smoke failed: expected Python 3.11, got {sys.version}")
if torch.version.cuda != "12.6":
    raise SystemExit(f"GPU smoke failed: expected compiled CUDA 12.6, got {torch.version.cuda!r}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("GPU smoke failed: CUDA GPU is not visible inside the image")
if torch.cuda.device_count() != 1:
    raise SystemExit("GPU smoke failed: exactly one allocated GPU must be visible")
gpu_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
architectures = torch.cuda.get_arch_list()
print("gpu_name", gpu_name)
print("compute_capability", capability)
print("compiled_architectures", architectures)
if "L20" not in gpu_name:
    raise SystemExit(f"GPU smoke failed: expected NVIDIA L20, got {gpu_name!r}")
if capability != (8, 9):
    raise SystemExit(f"GPU smoke failed: expected compute capability (8, 9), got {capability!r}")
if AutoModelForSequenceClassification._model_mapping[Qwen3Config] is not Qwen3ForSequenceClassification:
    raise SystemExit("GPU smoke failed: Qwen3 sequence-classification auto mapping is wrong")
required_symbols = (
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForSequenceClassification,
    DownloadConfig,
    LoraConfig,
    get_peft_model,
)
if any(symbol is None for symbol in required_symbols):
    raise SystemExit("GPU smoke failed: a required Qwen/PEFT/Datasets symbol is unavailable")
generation = GenerationConfig()
for field in (
    "stop_strings",
    "dola_layers",
    "min_p",
    "token_healing",
    "watermarking_config",
):
    if not hasattr(generation, field):
        raise SystemExit(f"GPU smoke failed: GenerationConfig lacks {field}")
roundtrip = load(save({"x": torch.arange(8, dtype=torch.float32)}))
if not torch.equal(roundtrip["x"], torch.arange(8, dtype=torch.float32)):
    raise SystemExit("GPU smoke failed: safetensors round trip changed data")
a = torch.randn(256, 256, device="cuda", dtype=torch.float32, requires_grad=True)
b = torch.randn(256, 256, device="cuda", dtype=torch.float32, requires_grad=True)
loss = (a @ b).square().mean()
loss.backward()
torch.cuda.synchronize()
if not torch.isfinite(loss) or a.grad is None or b.grad is None:
    raise SystemExit("GPU smoke failed: CUDA FP32 forward/backward is invalid")
print("cuda_fp32_loss", loss.item())
PY

apptainer exec --cleanenv "${PRORM_IMAGE}" git --version | tee -a "${report}"

apptainer exec --cleanenv "${PRORM_IMAGE}" python -m pip check | tee -a "${report}"

apptainer exec --cleanenv "${PRORM_IMAGE}" python -m pip freeze \
  | sort | tee -a "${report}"
