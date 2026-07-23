#!/usr/bin/env bash
set -euo pipefail

account="sigroup"

resolve_compat_env() {
  local canonical="$1" legacy="$2" canonical_value="" legacy_value="" resolved=""
  local canonical_set=0 legacy_set=0
  if [[ -v "${canonical}" ]]; then canonical_set=1; canonical_value="${!canonical}"; fi
  if [[ -v "${legacy}" ]]; then legacy_set=1; legacy_value="${!legacy}"; fi
  if (( canonical_set && legacy_set )) && [[ "${canonical_value}" != "${legacy_value}" ]]; then
    echo "error: conflicting ${canonical} and legacy ${legacy}" >&2
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

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
resolve_compat_env PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
if [[ -z "${PRORM_PROJECT_ROOT}" ]]; then
  die "set PRORM_PROJECT_ROOT (recommended: /project/sigroup/smart-reward-model)"
fi
if [[ -z "${PRORM_SCRATCH_ROOT}" ]]; then
  die 'set PRORM_SCRATCH_ROOT (recommended: /scratch/$USER/smart-reward-model)'
fi
normalize_absolute_root PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
normalize_absolute_root PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "PRORM_PROJECT_ROOT and PRORM_SCRATCH_ROOT may not be equal or nested"

echo "== identity =="
id

echo "== quota =="
squota
squota -A "${account}"

echo "== scheduler =="
savail
squeue -u "${USER}"
scontrol show partition

echo "== storage =="
[[ -w "${PRORM_PROJECT_ROOT}" ]] || {
  die "PRORM_PROJECT_ROOT is not writable: ${PRORM_PROJECT_ROOT}"
}
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || {
  die "PRORM_SCRATCH_ROOT is not writable: ${PRORM_SCRATCH_ROOT}"
}
printf 'project_root=%s\n' "${PRORM_PROJECT_ROOT}"
printf 'scratch_root=%s\n' "${PRORM_SCRATCH_ROOT}"
df -h "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}"

echo "== software =="
module avail 2>&1 | head -n 200 || true
command -v apptainer
apptainer --version

echo "Preflight passed. Next, probe the host GPU/driver before validating an image:"
echo "  bash scripts/hpc4/submit_host_gpu_probe.sh <gpu-partition>"
