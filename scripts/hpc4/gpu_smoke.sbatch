#!/usr/bin/env bash
#SBATCH --job-name=prorm-gpu-smoke
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00

set -euo pipefail

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

resolve_compat_env PRORM_IMAGE SRM_IMAGE
resolve_compat_env PRORM_IMAGE_SHA256 SRM_IMAGE_SHA256
resolve_compat_env PRORM_REPO_ROOT SRM_REPO_ROOT
: "${PRORM_IMAGE:?set PRORM_IMAGE to the absolute path of the validated Apptainer image}"
: "${PRORM_IMAGE_SHA256:?set PRORM_IMAGE_SHA256 to the expected lowercase SHA256}"
: "${PRORM_REPO_ROOT:?set PRORM_REPO_ROOT to the absolute repository root}"
[[ "${PRORM_REPO_ROOT}" = /* ]] || { echo "PRORM_REPO_ROOT must be absolute" >&2; exit 2; }
test -d "${PRORM_REPO_ROOT}/.git"
test -f "${PRORM_IMAGE}"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" | sha256sum --check --status

repo_root="${PRORM_REPO_ROOT}"

container_visibility=()
for variable in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    container_visibility+=(--env "${variable}=${!variable}")
  fi
done

scratch_base="/scratch/${USER}/smart-reward-model"
project_base="/project/sigroup/smart-reward-model"
job_dir="${scratch_base}/jobs/${SLURM_JOB_ID}"
mkdir -p "${job_dir}" "${project_base}/system-reports"

report="${job_dir}/gpu-smoke-${SLURM_JOB_ID}.txt"
sync_report() {
  if [[ -f "${report}" ]]; then
    cp -f "${report}" "${project_base}/system-reports/"
  fi
}
trap sync_report EXIT
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "submit_dir=${SLURM_SUBMIT_DIR}"
  echo "git_commit=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "image=${PRORM_IMAGE}"
  echo "image_sha256=${PRORM_IMAGE_SHA256}"
  hostname
  nvidia-smi
  env | grep '^SLURM_' | sort
} | tee "${report}"

apptainer exec --cleanenv --nv \
  --bind "${repo_root}:${repo_root}" \
  --env "PYTHONPATH=${repo_root}/src" \
  "${container_visibility[@]}" \
  "${PRORM_IMAGE}" \
  python - <<'PY' | tee -a "${report}"
import torch
import transformers
from importlib.metadata import version
from packaging.version import Version

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("transformers", transformers.__version__)
for package in ("accelerate", "datasets", "peft", "safetensors"):
    print(package, version(package))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("GPU smoke failed: CUDA GPU is not visible inside the image")
if torch.cuda.device_count() != 1:
    raise SystemExit("GPU smoke failed: exactly one allocated GPU must be visible")
if not hasattr(transformers, "Qwen3ForSequenceClassification"):
    raise SystemExit("GPU smoke failed: Transformers lacks the pinned Qwen3 oracle class")
if not Version("4.52.3") <= Version(transformers.__version__) < Version("5"):
    raise SystemExit("GPU smoke failed: Transformers version is outside [4.52.3, 5)")
x = torch.arange(16, device="cuda", dtype=torch.float32)
print("cuda_sum", x.sum().item())
PY

apptainer exec --cleanenv "${PRORM_IMAGE}" git --version | tee -a "${report}"

apptainer exec --cleanenv "${PRORM_IMAGE}" python -m pip check | tee -a "${report}"

apptainer exec --cleanenv "${PRORM_IMAGE}" python -m pip freeze \
  | sort | tee -a "${report}"
