#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <config.yaml> <gpu-partition> <walltime>" >&2
  exit 2
fi

: "${SRM_IMAGE:?set SRM_IMAGE to an absolute .sif path}"
: "${SRM_IMAGE_SHA256:?set SRM_IMAGE_SHA256 to that image SHA256}"
: "${SRM_HF_CACHE:?set SRM_HF_CACHE to the pre-staged offline cache root}"

config="$(realpath "$1")"
partition="$2"
walltime="$3"
case "${partition}" in
  gpu-a30|gpu-l20|gpu-rtx5880|gpu-rtx4090d) ;;
  *) echo "unsupported HPC4 GPU partition: ${partition}" >&2; exit 2 ;;
esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
  echo "walltime must be HH:MM:SS or D-HH:MM:SS" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test -f "${config}"
test -f "${SRM_IMAGE}"
test -d "${SRM_HF_CACHE}"
printf '%s  %s\n' "${SRM_IMAGE_SHA256}" "${SRM_IMAGE}" | sha256sum --check --status

seed_count="$({
  apptainer exec --cleanenv \
    --bind "${repo_root}:${repo_root},${config}:${config}" \
    --env "PYTHONPATH=${repo_root}/src" \
    "${SRM_IMAGE}" python - "${config}" <<'PY'
import sys
from smart_reward.config import load_config

run = load_config(sys.argv[1])["run"]
print(1 if "seed" in run else len(run["seeds"]))
PY
} | tail -n 1)"
[[ "${seed_count}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid configured seed count" >&2; exit 2; }
concurrency="${SRM_ARRAY_CONCURRENCY:-1}"
[[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid array concurrency" >&2; exit 2; }

mkdir -p "${repo_root}/logs"
sbatch \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --time="${walltime}" \
  --array="0-$((seed_count - 1))%${concurrency}" \
  --output="${repo_root}/logs/%x-%A_%a.out" \
  --export="ALL,SRM_IMAGE=${SRM_IMAGE},SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256},SRM_CONFIG=${config},SRM_HF_CACHE=${SRM_HF_CACHE},SRM_REPO_ROOT=${repo_root}" \
  "${repo_root}/scripts/hpc4/controlled.sbatch"
