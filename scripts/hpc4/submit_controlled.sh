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

die() {
  echo "error: $*" >&2
  exit 2
}

reject_apptainer_control_environment() {
  local variable=""
  while IFS= read -r variable; do
    case "${variable}" in
      APPTAINER*|SINGULARITY*)
        die "unset exported ${variable}; formal submission forbids ambient container controls"
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

resolve_project_path() {
  local canonical="$1" legacy="$2" kind="$3" raw="${!1}" candidate="" resolved=""
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
  case "${kind}" in
    file) [[ -f "${resolved}" ]] || die "${canonical} is not a file: ${resolved}" ;;
    directory) [[ -d "${resolved}" ]] || die "${canonical} is not a directory: ${resolved}" ;;
    *) die "internal path kind is invalid: ${kind}" ;;
  esac
  printf -v "${canonical}" '%s' "${resolved}"
  printf -v "${legacy}" '%s' "${resolved}"
  export "${canonical}" "${legacy}"
}

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
resolve_compat_env PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
resolve_compat_env PRORM_IMAGE SRM_IMAGE
resolve_compat_env PRORM_IMAGE_SHA256 SRM_IMAGE_SHA256
resolve_compat_env PRORM_HF_CACHE SRM_HF_CACHE
: "${PRORM_PROJECT_ROOT:?set PRORM_PROJECT_ROOT, for example /project/sigroup/smart-reward-model}"
: "${PRORM_SCRATCH_ROOT:?set PRORM_SCRATCH_ROOT, for example /scratch/\$USER/smart-reward-model}"
: "${PRORM_IMAGE:?set PRORM_IMAGE to a .sif path, absolute or relative to PRORM_PROJECT_ROOT}"
: "${PRORM_IMAGE_SHA256:?set PRORM_IMAGE_SHA256 to that image SHA256}"
: "${PRORM_HF_CACHE:?set PRORM_HF_CACHE, absolute or relative to PRORM_PROJECT_ROOT}"
reject_apptainer_control_environment

normalize_absolute_root PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
normalize_absolute_root PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "PRORM_PROJECT_ROOT and PRORM_SCRATCH_ROOT may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "PRORM_PROJECT_ROOT is not writable: ${PRORM_PROJECT_ROOT}"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "PRORM_SCRATCH_ROOT is not writable: ${PRORM_SCRATCH_ROOT}"
resolve_project_path PRORM_IMAGE SRM_IMAGE file
resolve_project_path PRORM_HF_CACHE SRM_HF_CACHE directory
if [[ ! "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  die "PRORM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters"
fi

if ! config="$(realpath -e -- "$1")"; then
  die "configuration file does not exist or cannot be resolved: $1"
fi
partition="$2"
walltime="$3"
case "${partition}" in
  gpu-a30|gpu-l20|gpu-rtx5880|gpu-rtx4090d) ;;
  *) die "unsupported HPC4 GPU partition: ${partition}" ;;
esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
  die "walltime must be HH:MM:SS or D-HH:MM:SS"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ -f "${config}" ]] || die "configuration path is not a file: ${config}"
case "${config}" in
  "${repo_root}"/*) ;;
  *) die "formal configs must be tracked files inside the repository: ${config}" ;;
esac
config_relative="$(realpath --relative-to="${repo_root}" "${config}")"
git -C "${repo_root}" ls-files --error-unmatch -- "${config_relative}" >/dev/null \
  || die "formal config is not tracked by Git: ${config_relative}"
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "could not resolve a full Git HEAD"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "formal controlled submission requires a clean Git worktree"
reject_bind_delimiters PRORM_PROJECT_ROOT "${PRORM_PROJECT_ROOT}"
reject_bind_delimiters PRORM_SCRATCH_ROOT "${PRORM_SCRATCH_ROOT}"
reject_bind_delimiters PRORM_IMAGE "${PRORM_IMAGE}"
reject_bind_delimiters PRORM_CONFIG "${config}"
reject_bind_delimiters PRORM_HF_CACHE "${PRORM_HF_CACHE}"
reject_bind_delimiters PRORM_REPO_ROOT "${repo_root}"
if ! printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status; then
  die "PRORM_IMAGE_SHA256 does not match PRORM_IMAGE: ${PRORM_IMAGE}"
fi

mapfile -t config_info < <({
  apptainer exec --cleanenv \
    --bind "${repo_root}:${repo_root}" \
    --env "PYTHONPATH=${repo_root}/src" \
    "${PRORM_IMAGE}" python - "${config}" <<'PY'
import sys
from smart_reward.config import config_hash, load_config

config = load_config(sys.argv[1])
run = config["run"]
print(1 if "seed" in run else len(run["seeds"]))
print(config_hash(config))
PY
})
[[ "${#config_info[@]}" -eq 2 ]] || die "failed to resolve config identity in the container"
seed_count="${config_info[0]}"
config_sha="${config_info[1]}"
[[ "${seed_count}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid configured seed count" >&2; exit 2; }
[[ "${config_sha}" =~ ^[0-9a-f]{64}$ ]] || die "invalid configured config hash"

inventory_expected="${PRORM_HF_CACHE}/inventories/${config_sha}.json"
if ! PRORM_HF_INVENTORY="$(realpath -e -- "${inventory_expected}")"; then
  die "missing required staged asset inventory: ${inventory_expected}"
fi
[[ -f "${PRORM_HF_INVENTORY}" ]] || die "inventory is not a regular file: ${PRORM_HF_INVENTORY}"
case "${PRORM_HF_INVENTORY}" in
  "${PRORM_HF_CACHE}/inventories/"*) ;;
  *) die "inventory resolves outside PRORM_HF_CACHE/inventories: ${PRORM_HF_INVENTORY}" ;;
esac
PRORM_HF_INVENTORY_SHA256="$(sha256sum -- "${PRORM_HF_INVENTORY}" | awk '{print $1}')"
[[ "${PRORM_HF_INVENTORY_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to hash asset inventory: ${PRORM_HF_INVENTORY}"
SRM_HF_INVENTORY="${PRORM_HF_INVENTORY}"
SRM_HF_INVENTORY_SHA256="${PRORM_HF_INVENTORY_SHA256}"
PRORM_GIT_COMMIT="${git_commit}"
SRM_GIT_COMMIT="${git_commit}"
export \
  PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 PRORM_GIT_COMMIT \
  SRM_HF_INVENTORY SRM_HF_INVENTORY_SHA256 SRM_GIT_COMMIT

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

for export_name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_CONFIG PRORM_HF_CACHE PRORM_REPO_ROOT PRORM_GIT_COMMIT \
  PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  SRM_PROJECT_ROOT SRM_SCRATCH_ROOT SRM_IMAGE SRM_IMAGE_SHA256 \
  SRM_CONFIG SRM_HF_CACHE SRM_REPO_ROOT SRM_GIT_COMMIT \
  SRM_HF_INVENTORY SRM_HF_INVENTORY_SHA256; do
  case "${export_name}" in
    PRORM_CONFIG|SRM_CONFIG)
      export_value="${config}"
      ;;
    PRORM_REPO_ROOT|SRM_REPO_ROOT)
      export_value="${repo_root}"
      ;;
    *)
      export_value="${!export_name}"
      ;;
  esac
  reject_export_delimiters "${export_name}" "${export_value}"
done
for bind_name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_CONFIG \
  PRORM_HF_CACHE PRORM_REPO_ROOT PRORM_HF_INVENTORY; do
  case "${bind_name}" in
    PRORM_CONFIG) bind_value="${config}" ;;
    PRORM_REPO_ROOT) bind_value="${repo_root}" ;;
    *) bind_value="${!bind_name}" ;;
  esac
  reject_bind_delimiters "${bind_name}" "${bind_value}"
done

# Close the submit/check race as far as the submit host can: the compute job
# independently requires this exact clean commit again.
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" == "${git_commit}" ]] \
  || die "Git HEAD changed while preparing the submission"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "Git worktree changed while preparing the submission"

slurm_log_dir="${PRORM_PROJECT_ROOT}/slurm-logs"
mkdir -p "${slurm_log_dir}"
sbatch \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --time="${walltime}" \
  --array="0-$((seed_count - 1))%${concurrency}" \
  --output="${slurm_log_dir}/%x-%A_%a.out" \
  --export="ALL,PRORM_PROJECT_ROOT=${PRORM_PROJECT_ROOT},PRORM_SCRATCH_ROOT=${PRORM_SCRATCH_ROOT},PRORM_IMAGE=${PRORM_IMAGE},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_CONFIG=${config},PRORM_HF_CACHE=${PRORM_HF_CACHE},PRORM_REPO_ROOT=${repo_root},PRORM_GIT_COMMIT=${git_commit},PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY},PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256},SRM_PROJECT_ROOT=${SRM_PROJECT_ROOT},SRM_SCRATCH_ROOT=${SRM_SCRATCH_ROOT},SRM_IMAGE=${SRM_IMAGE},SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256},SRM_CONFIG=${config},SRM_HF_CACHE=${SRM_HF_CACHE},SRM_REPO_ROOT=${repo_root},SRM_GIT_COMMIT=${git_commit},SRM_HF_INVENTORY=${SRM_HF_INVENTORY},SRM_HF_INVENTORY_SHA256=${SRM_HF_INVENTORY_SHA256}" \
  "${repo_root}/scripts/hpc4/controlled.sbatch"
