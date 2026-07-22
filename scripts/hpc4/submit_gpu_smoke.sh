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

resolve_compat_env PRORM_IMAGE SRM_IMAGE
resolve_compat_env PRORM_IMAGE_SHA256 SRM_IMAGE_SHA256
: "${PRORM_IMAGE:?set PRORM_IMAGE to an absolute .sif path}"
: "${PRORM_IMAGE_SHA256:?set PRORM_IMAGE_SHA256 to the lowercase image SHA256}"
case "${PRORM_IMAGE}" in
  /*) ;;
  *) echo "PRORM_IMAGE must be an absolute path" >&2; exit 2 ;;
esac
test -f "${PRORM_IMAGE}"
if [[ ! "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "PRORM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters" >&2
  exit 2
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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${repo_root}/logs"

sbatch \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --output="${repo_root}/logs/%x-%j.out" \
  --export="ALL,PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_REPO_ROOT=${repo_root},SRM_IMAGE=${SRM_IMAGE},SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256},SRM_REPO_ROOT=${repo_root}" \
  "${repo_root}/scripts/hpc4/gpu_smoke.sbatch"
