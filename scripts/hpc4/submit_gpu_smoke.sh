#!/usr/bin/env bash
set -euo pipefail

: "${SRM_IMAGE:?set SRM_IMAGE to an absolute .sif path}"
: "${SRM_IMAGE_SHA256:?set SRM_IMAGE_SHA256 to that image's lowercase SHA256}"
case "${SRM_IMAGE}" in
  /*) ;;
  *) echo "SRM_IMAGE must be an absolute path" >&2; exit 2 ;;
esac
test -f "${SRM_IMAGE}"
if [[ ! "${SRM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "SRM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters" >&2
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
  --export="ALL,SRM_IMAGE=${SRM_IMAGE},SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256},SRM_REPO_ROOT=${repo_root}" \
  "${repo_root}/scripts/hpc4/gpu_smoke.sbatch"
