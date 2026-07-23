#!/usr/bin/env bash
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

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <gpu-partition>" >&2
  exit 2
fi

partition="$1"
case "${partition}" in
  gpu-a30|gpu-l20|gpu-rtx5880|gpu-rtx4090d) ;;
  *)
    echo "unsupported HPC4 GPU partition: ${partition}" >&2
    exit 2
    ;;
esac

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
: "${PRORM_PROJECT_ROOT:?set PRORM_PROJECT_ROOT to an absolute persistent project root}"
case "${PRORM_PROJECT_ROOT}" in
  /*) ;;
  *)
    echo "PRORM_PROJECT_ROOT must be an absolute path" >&2
    exit 2
    ;;
esac
[[ "${PRORM_PROJECT_ROOT}" != *","* && "${PRORM_PROJECT_ROOT}" != *$'\n'* \
  && "${PRORM_PROJECT_ROOT}" != *$'\r'* ]] || {
  echo "PRORM_PROJECT_ROOT may not contain commas or newlines" >&2
  exit 2
}
test -d "${PRORM_PROJECT_ROOT}"
test -w "${PRORM_PROJECT_ROOT}"

project_root="$(realpath "${PRORM_PROJECT_ROOT}")"
PRORM_PROJECT_ROOT="${project_root}"
SRM_PROJECT_ROOT="${project_root}"
export PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
[[ "${PRORM_PROJECT_ROOT}" != *","* && "${PRORM_PROJECT_ROOT}" != *$'\n'* \
  && "${PRORM_PROJECT_ROOT}" != *$'\r'* ]] || {
  echo "canonical PRORM_PROJECT_ROOT may not contain commas or newlines" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
test -f "${repo_root}/scripts/hpc4/host_gpu_probe.sbatch"
command -v sbatch >/dev/null
slurm_log_dir="${PRORM_PROJECT_ROOT}/slurm-logs"
mkdir -p "${slurm_log_dir}"

sbatch \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --output="${slurm_log_dir}/%x-%j.out" \
  --export="ALL,PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},SRM_PROJECT_ROOT=${SRM_PROJECT_ROOT}" \
  "${repo_root}/scripts/hpc4/host_gpu_probe.sbatch"
