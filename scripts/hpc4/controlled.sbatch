#!/usr/bin/env bash
#SBATCH --job-name=srm-controlled
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1

set -euo pipefail

: "${SRM_IMAGE:?absolute validated Apptainer image is required}"
: "${SRM_IMAGE_SHA256:?validated image SHA256 is required}"
: "${SRM_CONFIG:?absolute experiment config path is required}"
: "${SRM_HF_CACHE:?absolute pre-staged Hugging Face cache is required}"
: "${SRM_REPO_ROOT:?absolute Git repository root is required}"

for path in "${SRM_IMAGE}" "${SRM_CONFIG}"; do
  [[ "${path}" = /* ]] || { echo "path must be absolute: ${path}" >&2; exit 2; }
  test -f "${path}"
done
[[ "${SRM_HF_CACHE}" = /* ]] || { echo "SRM_HF_CACHE must be absolute" >&2; exit 2; }
test -d "${SRM_HF_CACHE}"
[[ "${SRM_REPO_ROOT}" = /* ]] || { echo "SRM_REPO_ROOT must be absolute" >&2; exit 2; }
test -d "${SRM_REPO_ROOT}/.git"
printf '%s  %s\n' "${SRM_IMAGE_SHA256}" "${SRM_IMAGE}" | sha256sum --check --status

repo_root="${SRM_REPO_ROOT}"
config_path="$(realpath "${SRM_CONFIG}")"
case "${config_path}" in
  "${repo_root}"/*) ;;
  *) echo "formal configs must be tracked files inside SRM_REPO_ROOT" >&2; exit 2 ;;
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
  --bind "${repo_root}:${repo_root},${job_dir}:${job_dir},${SRM_HF_CACHE}:${SRM_HF_CACHE}"
  --env "PYTHONPATH=${repo_root}/src"
  --env "HF_HOME=${SRM_HF_CACHE}"
  --env "HF_HUB_CACHE=${SRM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${SRM_HF_CACHE}/datasets"
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
  --env "SRM_IMAGE=${SRM_IMAGE}"
  --env "SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256}"
  --env "SRM_GIT_COMMIT=${git_commit}"
  "${SRM_IMAGE}"
)

mapfile -t run_info < <(
  "${container[@]}" python - "${SRM_CONFIG}" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
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
project_artifact="${project_base}/artifacts/${config_sha}/${SRM_IMAGE_SHA256}/${git_commit}/seed-${seed}"
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
  "${SRM_CONFIG}" --seed "${seed}" --repo-root "${repo_root}" \
  --output "${job_dir}/run-manifest.json"

if [[ -f "${project_artifact}/metadata.json" ]]; then
  cp -a "${project_artifact}" "${artifact_dir}"
else
  "${container[@]}" env "SRM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
    python -m smart_reward.cli controlled-materialize \
    "${SRM_CONFIG}" "${artifact_dir}" --seed "${seed}" --device cuda
  mkdir -p "$(dirname "${project_artifact}")"
  artifact_staging="${project_artifact}.tmp-${SLURM_JOB_ID}"
  cp -a "${artifact_dir}" "${artifact_staging}"
  mv "${artifact_staging}" "${project_artifact}"
fi

"${container[@]}" env "SRM_MEMORY_REPORT=${job_dir}/memory-compare.json" \
  python -m smart_reward.cli controlled-compare \
  "${SRM_CONFIG}" "${artifact_dir}" "${job_dir}/comparison.json" \
  --seed "${seed}" --device cuda --run-manifest "${job_dir}/run-manifest.json"

"${container[@]}" env "SRM_MEMORY_REPORT=${job_dir}/memory-rollout.json" \
  python -m smart_reward.cli controlled-rollout \
  "${SRM_CONFIG}" "${artifact_dir}" "${job_dir}/comparison.json" \
  "${job_dir}/rollout.json" --seed "${seed}" --device cuda

echo "controlled run complete: ${project_run}"
