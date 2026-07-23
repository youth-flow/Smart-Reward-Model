#!/usr/bin/env bash
#SBATCH --job-name=prorm-hf-stage
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

reject_apptainer_control_environment() {
  local variable=""
  while IFS= read -r variable; do
    case "${variable}" in
      APPTAINER*|SINGULARITY*)
        die "unset exported ${variable}; staging forbids ambient container controls"
        ;;
    esac
  done < <(compgen -e)
}

require_canonical_directory() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" \
    || die "${name} does not exist or cannot be resolved: ${raw}"
  [[ -d "${resolved}" ]] || die "${name} is not a directory: ${resolved}"
  [[ "${resolved}" = "${raw}" ]] || die "${name} must be canonical: ${raw}"
  [[ "${resolved}" != "/" ]] || die "${name} may not be the filesystem root"
}

require_canonical_file() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" \
    || die "${name} does not exist or cannot be resolved: ${raw}"
  [[ -f "${resolved}" ]] || die "${name} is not a regular file: ${resolved}"
  [[ "${resolved}" = "${raw}" ]] || die "${name} must be canonical: ${raw}"
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

require_contained() {
  local name="$1" child="$2" parent="$3"
  case "${child}" in
    "${parent}"/*) ;;
    *) die "${name} escaped its required root: ${child}" ;;
  esac
}

reject_path_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *":"* && "${value}" != *","* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} contains an unsafe bind/export delimiter"
}

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE PRORM_REPO_ROOT PRORM_CONFIG PRORM_CONFIG_HASH \
  PRORM_CONFIG_FILE_SHA256 PRORM_GIT_COMMIT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
: "${SLURM_JOB_ID:?HF staging must execute inside a Slurm job}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_JOB_ID must be positive"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] \
  || die "HF staging requires Slurm account sigroup"
case "${SLURM_JOB_PARTITION:-}" in
  amd|intel) ;;
  *) die "HF staging requires partition amd or intel" ;;
esac
reject_apptainer_control_environment

for command_name in apptainer flock git python3 realpath sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

require_canonical_directory PRORM_PROJECT_ROOT
require_canonical_directory PRORM_SCRATCH_ROOT
require_canonical_directory PRORM_HF_CACHE
require_canonical_directory PRORM_REPO_ROOT
require_canonical_file PRORM_IMAGE
require_canonical_file PRORM_CONFIG
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "project and scratch roots may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "project root is not writable"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "scratch root is not writable"
[[ -w "${PRORM_HF_CACHE}" ]] || die "Hugging Face cache is not writable"
require_contained PRORM_IMAGE "${PRORM_IMAGE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_HF_CACHE "${PRORM_HF_CACHE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_CONFIG "${PRORM_CONFIG}" "${PRORM_REPO_ROOT}"

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_HF_CACHE \
  PRORM_REPO_ROOT PRORM_CONFIG; do
  reject_path_delimiters "${name}" "${!name}"
done
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
[[ "${PRORM_CONFIG_HASH}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_CONFIG_HASH must be lowercase SHA256"
[[ "${PRORM_CONFIG_FILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_CONFIG_FILE_SHA256 must be lowercase SHA256"
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] \
  || die "PRORM_GIT_COMMIT must be a full object ID"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status \
  || die "image SHA256 mismatch"
printf '%s  %s\n' "${PRORM_CONFIG_FILE_SHA256}" "${PRORM_CONFIG}" \
  | sha256sum --check --status \
  || die "worktree config changed after submission"

repository_top="$(git -C "${PRORM_REPO_ROOT}" rev-parse --show-toplevel)"
[[ "${repository_top}" = "${PRORM_REPO_ROOT}" ]] || die "invalid repository root"
repo_head="$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${repo_head}" = "${PRORM_GIT_COMMIT}" ]] \
  || die "repository HEAD changed after submission"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "staging requires a clean repository"
config_relative="$(
  realpath --relative-to="${PRORM_REPO_ROOT}" "${PRORM_CONFIG}"
)"
case "${config_relative}" in
  configs/*.yaml) ;;
  *) die "staging config must be a tracked configs/*.yaml file" ;;
esac
git -C "${PRORM_REPO_ROOT}" cat-file blob \
  "${PRORM_GIT_COMMIT}:${config_relative}" \
  | sha256sum \
  | awk '{print $1}' \
  | grep -Fx "${PRORM_CONFIG_FILE_SHA256}" >/dev/null \
  || die "committed config bytes do not match submitted config identity"

job_dir="${PRORM_SCRATCH_ROOT}/jobs/${SLURM_JOB_ID}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "refusing to reuse scratch job directory: ${job_dir}"
mkdir -p "${PRORM_SCRATCH_ROOT}/jobs" "${PRORM_PROJECT_ROOT}/system-reports"
mkdir "${job_dir}"

report="${job_dir}/hf-stage-${SLURM_JOB_ID}.log"
persisted_report="${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${SLURM_JOB_ID}.log"
: > "${report}"
finalize_report() {
  local workload_exit=$? temporary=""
  trap - EXIT
  set +e
  if [[ ! -f "${report}" ]]; then
    printf 'staging report was not created\n' >&2
    exit 1
  fi
  if [[ -e "${persisted_report}" || -L "${persisted_report}" ]]; then
    printf 'refusing to overwrite staging report: %s\n' "${persisted_report}" >&2
    exit 1
  fi
  temporary="$(mktemp \
    "${PRORM_PROJECT_ROOT}/system-reports/.hf-stage-${SLURM_JOB_ID}.XXXXXX")" \
    || exit 1
  cp -p -- "${report}" "${temporary}" || {
    rm -f -- "${temporary}"
    exit 1
  }
  {
    printf 'workload_exit_code=%s\n' "${workload_exit}"
    if (( workload_exit == 0 )); then
      printf 'status=passed\n'
    else
      printf 'status=failed\n'
    fi
  } >> "${temporary}" || {
    rm -f -- "${temporary}"
    exit 1
  }
  chmod 0440 "${temporary}" || {
    rm -f -- "${temporary}"
    exit 1
  }
  ln -- "${temporary}" "${persisted_report}" || {
    rm -f -- "${temporary}"
    exit 1
  }
  rm -f -- "${temporary}" || exit 1
  exit "${workload_exit}"
}
trap finalize_report EXIT

{
  printf 'schema_version=prorm-hf-stage-report/v1\n'
  printf 'job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'account=%s\n' "${SLURM_JOB_ACCOUNT}"
  printf 'partition=%s\n' "${SLURM_JOB_PARTITION}"
  printf 'git_commit=%s\n' "${PRORM_GIT_COMMIT}"
  printf 'image_sha256=%s\n' "${PRORM_IMAGE_SHA256}"
  printf 'config=%s\n' "${config_relative}"
  printf 'config_hash=%s\n' "${PRORM_CONFIG_HASH}"
  printf 'config_file_sha256=%s\n' "${PRORM_CONFIG_FILE_SHA256}"
  printf 'node=%s\n' "$(hostname)"
} | tee -a "${report}"

# Freeze the submitted source before any networked staging. A mutable login
# checkout can no longer change the script or config while the job runs.
execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
[[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" ]] \
  || die "detached staging source has the wrong commit"
[[ -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
  || die "detached staging source is dirty"
execution_config="${execution_repo}/${config_relative}"
[[ -f "${execution_config}" ]] || die "config is missing from detached execution source"
printf '%s  %s\n' "${PRORM_CONFIG_FILE_SHA256}" "${execution_config}" \
  | sha256sum --check --status \
  || die "detached execution config does not match submitted identity"

command -v flock >/dev/null 2>&1 || die "flock is required for cache serialization"
exec {cache_lock_fd}> "${PRORM_HF_CACHE}/.stage.lock"
flock -n "${cache_lock_fd}" \
  || die "another Hugging Face staging job is writing the shared cache"

inventory_relative="inventories/${PRORM_CONFIG_HASH}.json"
inventory_expected="${PRORM_HF_CACHE}/${inventory_relative}"
stage_mode="networked-stage"
stage_arguments=("${execution_config}" "${PRORM_HF_CACHE}")
expected_verify_only="false"
if [[ -e "${inventory_expected}" || -L "${inventory_expected}" ]]; then
  [[ -f "${inventory_expected}" ]] \
    || die "existing inventory path is not a regular file: ${inventory_expected}"
  stage_mode="offline-reverify"
  expected_verify_only="true"
  stage_arguments+=(--verify-only --inventory "${inventory_expected}")
fi
printf 'stage_mode=%s\n' "${stage_mode}" | tee -a "${report}"

stage_output="${job_dir}/hf-stage-command.log"
apptainer exec --cleanenv \
  --no-mount home,cwd \
  --pwd "${execution_repo}" \
  --bind "${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}" \
  --env "PYTHONPATH=${execution_repo}/src" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  --env "HF_HOME=${PRORM_HF_CACHE}" \
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub" \
  --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets" \
  --env "HF_HUB_DISABLE_TELEMETRY=1" \
  --env "TOKENIZERS_PARALLELISM=false" \
  "${PRORM_IMAGE}" \
  python "${execution_repo}/scripts/hpc4/stage_hf_assets.py" \
  "${stage_arguments[@]}" \
  2>&1 | tee -a "${report}" "${stage_output}"

mapfile -t stage_result < <(
  python3 -I -S - \
    "${stage_output}" "${PRORM_CONFIG_HASH}" "${inventory_relative}" \
    "${expected_verify_only}" <<'PY'
import json
import re
import sys
from pathlib import Path


path, expected_hash, expected_inventory, expected_verify_raw = sys.argv[1:]
expected_verify = expected_verify_raw == "true"
result = None
for line in reversed(Path(path).read_text(encoding="utf-8").splitlines()):
    try:
        candidate = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and candidate.get("status") == "ok":
        result = candidate
        break
if result is None:
    raise SystemExit("stage command did not emit its success JSON")
if set(result) != {
    "config_hash",
    "inventory",
    "inventory_sha256",
    "status",
    "verify_only",
}:
    raise SystemExit("stage success JSON has an unexpected schema")
if result["config_hash"] != expected_hash:
    raise SystemExit("stage result config_hash mismatch")
if result["inventory"] != expected_inventory:
    raise SystemExit("stage result inventory path mismatch")
if result["verify_only"] is not expected_verify:
    raise SystemExit("stage result verify_only mode mismatch")
inventory_sha256 = result["inventory_sha256"]
if not isinstance(inventory_sha256, str) or re.fullmatch(
    r"[0-9a-f]{64}", inventory_sha256
) is None:
    raise SystemExit("stage result inventory_sha256 is invalid")
print(result["inventory"])
print(inventory_sha256)
PY
)
[[ "${#stage_result[@]}" -eq 2 ]] || die "could not validate stage command result"
[[ "${stage_result[0]}" = "${inventory_relative}" ]] \
  || die "validated inventory path changed unexpectedly"
inventory_sha256="${stage_result[1]}"

inventory_canonical="$(realpath -e -- "${inventory_expected}")" \
  || die "staged inventory does not exist"
[[ "${inventory_canonical}" = "${inventory_expected}" ]] \
  || die "staged inventory path is not canonical"
printf '%s  %s\n' "${inventory_sha256}" "${inventory_canonical}" \
  | sha256sum --check --status \
  || die "staged inventory SHA256 does not match stage result"
python3 -I -S - \
  "${inventory_canonical}" "${PRORM_CONFIG_HASH}" <<'PY'
import json
import sys
from pathlib import Path


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate inventory key: {key}")
        result[key] = value
    return result


payload = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")
    ),
)
if payload.get("schema_version") != "prorm-hf-assets/v1":
    raise SystemExit("staged inventory schema mismatch")
if payload.get("config_hash") != sys.argv[2]:
    raise SystemExit("staged inventory config_hash mismatch")
PY

{
  printf 'inventory=%s\n' "${inventory_relative}"
  printf 'inventory_sha256=%s\n' "${inventory_sha256}"
  printf 'offline_verification=passed\n'
} | tee -a "${report}"
