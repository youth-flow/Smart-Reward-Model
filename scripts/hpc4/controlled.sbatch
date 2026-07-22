#!/usr/bin/env bash
#SBATCH --job-name=prorm-controlled
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1

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
resolve_compat_env PRORM_CONFIG SRM_CONFIG
resolve_compat_env PRORM_HF_CACHE SRM_HF_CACHE
resolve_compat_env PRORM_REPO_ROOT SRM_REPO_ROOT
: "${PRORM_IMAGE:?absolute validated Apptainer image is required}"
: "${PRORM_IMAGE_SHA256:?validated image SHA256 is required}"
: "${PRORM_CONFIG:?absolute experiment config path is required}"
: "${PRORM_HF_CACHE:?absolute pre-staged Hugging Face cache is required}"
: "${PRORM_REPO_ROOT:?absolute Git repository root is required}"

for path in "${PRORM_IMAGE}" "${PRORM_CONFIG}"; do
  [[ "${path}" = /* ]] || { echo "path must be absolute: ${path}" >&2; exit 2; }
  test -f "${path}"
done
[[ "${PRORM_HF_CACHE}" = /* ]] || { echo "PRORM_HF_CACHE must be absolute" >&2; exit 2; }
test -d "${PRORM_HF_CACHE}"
[[ "${PRORM_REPO_ROOT}" = /* ]] || { echo "PRORM_REPO_ROOT must be absolute" >&2; exit 2; }
test -d "${PRORM_REPO_ROOT}/.git"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" | sha256sum --check --status

repo_root="${PRORM_REPO_ROOT}"
config_path="$(realpath "${PRORM_CONFIG}")"
case "${config_path}" in
  "${repo_root}"/*) ;;
  *) echo "formal configs must be tracked files inside PRORM_REPO_ROOT" >&2; exit 2 ;;
esac
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
if [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]]; then
  echo "formal controlled runs require a clean Git worktree" >&2
  exit 2
fi
config_relative="$(realpath --relative-to="${repo_root}" "${config_path}")"
git -C "${repo_root}" ls-files --error-unmatch -- "${config_relative}" >/dev/null
scratch_base="/scratch/${USER}/smart-reward-model"
project_base="/project/sigroup/smart-reward-model"
job_dir="${scratch_base}/jobs/${SLURM_JOB_ID}"
mkdir -p "${job_dir}" "${project_base}/runs" "${project_base}/artifacts"

container=(
  apptainer exec --cleanenv --nv
  --bind "${repo_root}:${repo_root},${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}"
  --env "PYTHONPATH=${repo_root}/src"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1"
)

# --cleanenv is deliberate; pass only the reproducibility allowlist consumed
# by RunManifest, never the caller's complete environment.
for variable in \
  SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
  SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST SLURM_NNODES \
  SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_GPUS SLURM_GPUS_ON_NODE \
  SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_SUBMIT_DIR \
  CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    container+=(--env "${variable}=${!variable}")
  fi
done
container+=(
  --env "PRORM_IMAGE=${PRORM_IMAGE}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_GIT_COMMIT=${git_commit}"
  --env "SRM_IMAGE=${SRM_IMAGE}"
  --env "SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256}"
  --env "SRM_GIT_COMMIT=${git_commit}"
  "${PRORM_IMAGE}"
)

mapfile -t run_info < <(
  "${container[@]}" python - "${PRORM_CONFIG}" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import sys
from smart_reward.config import config_hash, load_config

config = load_config(sys.argv[1])
index = int(sys.argv[2])
run = config["run"]
seeds = [run["seed"]] if "seed" in run else run["seeds"]
if not 0 <= index < len(seeds):
    raise SystemExit(f"array index {index} is outside configured seed count {len(seeds)}")
print(run["name"])
print(seeds[index])
print(config_hash(config))
PY
)
[[ "${#run_info[@]}" -eq 3 ]] || { echo "failed to resolve run identity" >&2; exit 2; }
run_name="${run_info[0]}"
seed="${run_info[1]}"
config_sha="${run_info[2]}"

artifact_dir="${job_dir}/artifact"
project_artifact="${project_base}/artifacts/${config_sha}/${PRORM_IMAGE_SHA256}/${git_commit}/seed-${seed}"
project_run="${project_base}/runs/${run_name}/${config_sha}/seed-${seed}/job-${SLURM_JOB_ID}"
mkdir -p "${project_run}"

sync_run() {
  for filename in \
    run-manifest.json comparison.json rollout.json updated_rollouts.jsonl \
    memory-materialize.json memory-compare.json memory-rollout.json; do
    if [[ -f "${job_dir}/${filename}" ]]; then
      cp -f "${job_dir}/${filename}" "${project_run}/${filename}"
    fi
  done
}
trap sync_run EXIT

"${container[@]}" python - <<'PY'
import subprocess
import torch

subprocess.run(["git", "--version"], check=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("formal controlled run requires exactly one visible allocated GPU")
print("formal_gpu", torch.cuda.get_device_name(0))
PY

"${container[@]}" python -m smart_reward.cli env-report \
  "${PRORM_CONFIG}" --seed "${seed}" --repo-root "${repo_root}" \
  --output "${job_dir}/run-manifest.json"

if [[ -f "${project_artifact}/metadata.json" ]]; then
  cp -a "${project_artifact}" "${artifact_dir}"
else
  "${container[@]}" env \
    "PRORM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
    "SRM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
    python -m smart_reward.cli controlled-materialize \
    "${PRORM_CONFIG}" "${artifact_dir}" --seed "${seed}" --device cuda
  mkdir -p "$(dirname "${project_artifact}")"
  artifact_staging="${project_artifact}.tmp-${SLURM_JOB_ID}"
  cp -a "${artifact_dir}" "${artifact_staging}"
  mv "${artifact_staging}" "${project_artifact}"
fi

"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-compare.json" \
  "SRM_MEMORY_REPORT=${job_dir}/memory-compare.json" \
  python -m smart_reward.cli controlled-compare \
  "${PRORM_CONFIG}" "${artifact_dir}" "${job_dir}/comparison.json" \
  --seed "${seed}" --device cuda --run-manifest "${job_dir}/run-manifest.json"

"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-rollout.json" \
  "SRM_MEMORY_REPORT=${job_dir}/memory-rollout.json" \
  python -m smart_reward.cli controlled-rollout \
  "${PRORM_CONFIG}" "${artifact_dir}" "${job_dir}/comparison.json" \
  "${job_dir}/rollout.json" --seed "${seed}" --device cuda

echo "controlled run complete: ${project_run}"
