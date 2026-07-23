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

reject_export_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || {
    die "${name} may not contain commas or newlines (unsafe for sbatch --export)"
  }
}

reject_bind_delimiters() {
  local name="$1" value="$2"
  reject_export_delimiters "${name}" "${value}"
  [[ "${value}" != *":"* ]] || {
    die "${name} may not contain ':' (unsafe for an Apptainer bind path)"
  }
}

resolve_project_file() {
  local canonical="$1" legacy="$2" raw="${!1}" candidate="" resolved=""
  [[ -n "${raw}" ]] || die "${canonical} must be set"
  if [[ "${raw}" = /* ]]; then
    candidate="${raw}"
  else
    candidate="${PRORM_PROJECT_ROOT}/${raw}"
  fi
  if ! resolved="$(realpath -e -- "${candidate}")"; then
    die "${canonical} does not exist or cannot be resolved: ${candidate}"
  fi
  if [[ "${raw}" != /* ]]; then
    case "${resolved}" in
      "${PRORM_PROJECT_ROOT}"|"${PRORM_PROJECT_ROOT}"/*) ;;
      *) die "${canonical} relative path escapes PRORM_PROJECT_ROOT: ${raw}" ;;
    esac
  fi
  [[ -f "${resolved}" ]] || die "${canonical} is not a file: ${resolved}"
  printf -v "${canonical}" '%s' "${resolved}"
  printf -v "${legacy}" '%s' "${resolved}"
  export "${canonical}" "${legacy}"
}

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
resolve_compat_env PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
resolve_compat_env PRORM_IMAGE SRM_IMAGE
resolve_compat_env PRORM_IMAGE_SHA256 SRM_IMAGE_SHA256
: "${PRORM_PROJECT_ROOT:?set PRORM_PROJECT_ROOT, for example /project/sigroup/smart-reward-model}"
: "${PRORM_SCRATCH_ROOT:?set PRORM_SCRATCH_ROOT, for example /scratch/\$USER/smart-reward-model}"
: "${PRORM_IMAGE:?set PRORM_IMAGE to a .sif path, absolute or relative to PRORM_PROJECT_ROOT}"
: "${PRORM_IMAGE_SHA256:?set PRORM_IMAGE_SHA256 to the lowercase image SHA256}"
reject_apptainer_control_environment

normalize_absolute_root PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
normalize_absolute_root PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "PRORM_PROJECT_ROOT and PRORM_SCRATCH_ROOT may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "PRORM_PROJECT_ROOT is not writable: ${PRORM_PROJECT_ROOT}"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "PRORM_SCRATCH_ROOT is not writable: ${PRORM_SCRATCH_ROOT}"
resolve_project_file PRORM_IMAGE SRM_IMAGE
if [[ ! "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  die "PRORM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters"
fi
if ! printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status; then
  die "PRORM_IMAGE_SHA256 does not match PRORM_IMAGE: ${PRORM_IMAGE}"
fi

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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
for export_name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_REPO_ROOT SRM_PROJECT_ROOT SRM_SCRATCH_ROOT SRM_IMAGE \
  SRM_IMAGE_SHA256 SRM_REPO_ROOT; do
  case "${export_name}" in
    PRORM_REPO_ROOT|SRM_REPO_ROOT) export_value="${repo_root}" ;;
    *) export_value="${!export_name}" ;;
  esac
  reject_export_delimiters "${export_name}" "${export_value}"
done
for bind_name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_REPO_ROOT; do
  case "${bind_name}" in
    PRORM_REPO_ROOT) bind_value="${repo_root}" ;;
    *) bind_value="${!bind_name}" ;;
  esac
  reject_bind_delimiters "${bind_name}" "${bind_value}"
done

slurm_log_dir="${PRORM_PROJECT_ROOT}/slurm-logs"
mkdir -p "${slurm_log_dir}"

sbatch \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --output="${slurm_log_dir}/%x-%j.out" \
  --export="ALL,PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},PRORM_SCRATCH_ROOT=${PRORM_SCRATCH_ROOT},PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_REPO_ROOT=${repo_root},SRM_PROJECT_ROOT=${SRM_PROJECT_ROOT},SRM_SCRATCH_ROOT=${SRM_SCRATCH_ROOT},SRM_IMAGE=${SRM_IMAGE},SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256},SRM_REPO_ROOT=${repo_root}" \
  "${repo_root}/scripts/hpc4/gpu_smoke.sbatch"
