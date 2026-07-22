#!/usr/bin/env bash
set -euo pipefail

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
  --partition="${partition}" \
  --output="${repo_root}/logs/%x-%j.out" \
  "${repo_root}/scripts/hpc4/gpu_smoke.sbatch"
