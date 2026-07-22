#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <config.yaml> <gpu-partition> <walltime>" >&2
  exit 2
fi

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
resolve_compat_env PRORM_HF_CACHE SRM_HF_CACHE
: "${PRORM_IMAGE:?set PRORM_IMAGE to an absolute .sif path}"
: "${PRORM_IMAGE_SHA256:?set PRORM_IMAGE_SHA256 to that image SHA256}"
: "${PRORM_HF_CACHE:?set PRORM_HF_CACHE to the pre-staged offline cache root}"

case "${PRORM_IMAGE}" in
  /*) ;;
  *) echo "PRORM_IMAGE must be an absolute path" >&2; exit 2 ;;
esac
case "${PRORM_HF_CACHE}" in
  /*) ;;
  *) echo "PRORM_HF_CACHE must be an absolute path" >&2; exit 2 ;;
esac
if [[ ! "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "PRORM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters" >&2
  exit 2
fi

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
test -f "${PRORM_IMAGE}"
test -d "${PRORM_HF_CACHE}"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" | sha256sum --check --status

seed_count="$({
  apptainer exec --cleanenv \
    --bind "${repo_root}:${repo_root},${config}:${config}" \
    --env "PYTHONPATH=${repo_root}/src" \
    "${PRORM_IMAGE}" python - "${config}" <<'PY'
import sys
from smart_reward.config import load_config

run = load_config(sys.argv[1])["run"]
print(1 if "seed" in run else len(run["seeds"]))
PY
} | tail -n 1)"
[[ "${seed_count}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid configured seed count" >&2; exit 2; }
if [[ -v PRORM_ARRAY_CONCURRENCY && -v SRM_ARRAY_CONCURRENCY ]]; then
  if [[ "${PRORM_ARRAY_CONCURRENCY}" != "${SRM_ARRAY_CONCURRENCY}" ]]; then
    echo "conflicting PRORM_ARRAY_CONCURRENCY and legacy SRM_ARRAY_CONCURRENCY" >&2
    exit 2
  fi
fi
if [[ -v PRORM_ARRAY_CONCURRENCY ]]; then
  concurrency="${PRORM_ARRAY_CONCURRENCY}"
elif [[ -v SRM_ARRAY_CONCURRENCY ]]; then
  concurrency="${SRM_ARRAY_CONCURRENCY}"
else
  concurrency=1
fi
[[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid array concurrency" >&2; exit 2; }

mkdir -p "${repo_root}/logs"
sbatch \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --time="${walltime}" \
  --array="0-$((seed_count - 1))%${concurrency}" \
  --output="${repo_root}/logs/%x-%A_%a.out" \
  --export="ALL,PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_CONFIG=${config},PRORM_HF_CACHE=${PRORM_HF_CACHE},PRORM_REPO_ROOT=${repo_root},SRM_IMAGE=${SRM_IMAGE},SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256},SRM_CONFIG=${config},SRM_HF_CACHE=${SRM_HF_CACHE},SRM_REPO_ROOT=${repo_root}" \
  "${repo_root}/scripts/hpc4/controlled.sbatch"
