#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ $# -lt 4 ]]; then
  die "usage: $0 <config.yaml> <cpu-partition> <walltime> <seed=controlled_job_id>..."
fi

config_input="$1"
partition="$2"
walltime="$3"
shift 3
seed_job_pairs=("$@")

case "${partition}" in
  amd|intel) ;;
  *) die "aggregation partition must be amd or intel" ;;
esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*)
      die "unset exported ${variable}; aggregation submission forbids ambient container controls"
      ;;
    SBATCH_*)
      die "unset exported ${variable}; formal submission forbids ambient sbatch option overrides"
      ;;
  esac
done < <(compgen -e)

for command_name in git python3 realpath sbatch sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "formal aggregation submission requires a clean repository"
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid Git HEAD"

config="$(realpath -e -- "${config_input}")" \
  || die "config does not exist or cannot be resolved: ${config_input}"
[[ -f "${config}" && ! -L "${config}" ]] || die "config must be a regular non-symlink file"
case "${config}" in
  "${repo_root}"/configs/*.yaml) ;;
  *) die "config must be a tracked configs/*.yaml file" ;;
esac
config_relative="$(realpath --relative-to="${repo_root}" "${config}")"
[[ "${config_relative}" =~ ^configs/[A-Za-z0-9._-]+\.yaml$ ]] \
  || die "config has an unsafe repository-relative path: ${config_relative}"
git -C "${repo_root}" ls-files --error-unmatch -- "${config_relative}" >/dev/null \
  || die "formal config is not tracked by Git: ${config_relative}"

# Resolve the experiment identity exclusively from blobs at the submitted
# commit. The login-node Python process never imports the project or executes
# the research image.
identity_relative="configs/identities.json"
config_worktree_sha256="$(sha256sum -- "${config}" | awk '{print $1}')"
identity_output="$(
  python3 -I -S - \
    "${repo_root}" "${git_commit}" "${identity_relative}" \
    "${config_relative}" "${config_worktree_sha256}" <<'PY'
import hashlib
import json
import re
import subprocess
import sys


repo_root, commit, identity_relative, config_relative, worktree_sha256 = sys.argv[1:]


def committed_blob(relative):
    result = subprocess.run(
        ["git", "-C", repo_root, "cat-file", "blob", f"{commit}:{relative}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"cannot read committed blob {relative}: {message}")
    return result.stdout


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


config_bytes = committed_blob(config_relative)
config_file_sha256 = hashlib.sha256(config_bytes).hexdigest()
if config_file_sha256 != worktree_sha256:
    raise SystemExit("worktree config bytes do not match submitted Git commit")
payload = json.loads(
    committed_blob(identity_relative).decode("utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")
    ),
)
if payload.get("schema_version") != "prorm-config-identities/v1":
    raise SystemExit("unsupported config identity schema")
configs = payload.get("configs")
if not isinstance(configs, dict) or config_relative not in configs:
    raise SystemExit(f"config is absent from identity file: {config_relative}")
entry = configs[config_relative]
if not isinstance(entry, dict) or set(entry) != {
    "config_hash",
    "file_sha256",
    "seed_count",
}:
    raise SystemExit("invalid config identity entry")
if entry["file_sha256"] != config_file_sha256:
    raise SystemExit("committed config bytes do not match the committed identity")
if not isinstance(entry["seed_count"], int) or entry["seed_count"] <= 0:
    raise SystemExit("invalid identity seed_count")
if not isinstance(entry["config_hash"], str) or not re.fullmatch(
    r"[0-9a-f]{64}", entry["config_hash"]
):
    raise SystemExit("invalid semantic config hash")
print(entry["seed_count"])
print(entry["config_hash"])
print(config_file_sha256)
PY
)" || die "failed to resolve committed config identity"
mapfile -t config_info <<< "${identity_output}"
[[ "${#config_info[@]}" -eq 3 ]] || die "failed to resolve committed config identity"
config_seed_count="${config_info[0]}"
config_hash="${config_info[1]}"
config_file_sha256="${config_info[2]}"
[[ "${config_seed_count}" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid committed config seed count"
[[ "${config_hash}" =~ ^[0-9a-f]{64}$ ]] || die "invalid committed config hash"
[[ "${config_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid committed config file SHA256"

declare -A seen_seeds=()
declare -A seen_jobs=()
for pair in "${seed_job_pairs[@]}"; do
  [[ "${pair}" =~ ^(0|-?[1-9][0-9]*)=([1-9][0-9]*)$ ]] \
    || die "seed/job mapping must be canonical seed=positive_job_id: ${pair}"
  seed="${BASH_REMATCH[1]}"
  controlled_job="${BASH_REMATCH[2]}"
  [[ ! -v "seen_seeds[${seed}]" ]] || die "duplicate seed mapping: ${seed}"
  [[ ! -v "seen_jobs[${controlled_job}]" ]] \
    || die "controlled job ID is mapped more than once: ${controlled_job}"
  seen_seeds["${seed}"]=1
  seen_jobs["${controlled_job}"]=1
done
[[ "${#seed_job_pairs[@]}" -eq "${config_seed_count}" ]] \
  || die "expected ${config_seed_count} explicit seed=job mappings"

project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")" \
  || die "PRORM_PROJECT_ROOT does not exist"
scratch_root="$(realpath -e -- "${PRORM_SCRATCH_ROOT}")" \
  || die "PRORM_SCRATCH_ROOT does not exist"
[[ -d "${project_root}" && -d "${scratch_root}" ]] \
  || die "project and scratch roots must be directories"
[[ "${project_root}" != "/" && "${scratch_root}" != "/" ]] \
  || die "project and scratch roots may not be /"
case "${project_root}" in
  "${scratch_root}"|"${scratch_root}"/*) die "project and scratch roots overlap" ;;
esac
case "${scratch_root}" in
  "${project_root}"|"${project_root}"/*) die "project and scratch roots overlap" ;;
esac
[[ -w "${project_root}" ]] || die "project root is not writable: ${project_root}"
[[ -w "${scratch_root}" ]] || die "scratch root is not writable: ${scratch_root}"

resolve_project_path() {
  local raw="$1" kind="$2" candidate="" resolved=""
  if [[ "${raw}" = /* ]]; then
    candidate="${raw}"
  else
    candidate="${project_root}/${raw}"
  fi
  resolved="$(realpath -e -- "${candidate}")" \
    || die "path does not exist or cannot be resolved: ${candidate}"
  case "${resolved}" in
    "${project_root}"/*) ;;
    *) die "project path escaped PRORM_PROJECT_ROOT: ${raw}" ;;
  esac
  case "${kind}" in
    file) [[ -f "${resolved}" ]] || die "not a file: ${resolved}" ;;
    directory) [[ -d "${resolved}" ]] || die "not a directory: ${resolved}" ;;
    *) die "internal path kind is invalid: ${kind}" ;;
  esac
  printf '%s\n' "${resolved}"
}

image="$(resolve_project_path "${PRORM_IMAGE}" file)"
hf_cache="$(resolve_project_path "${PRORM_HF_CACHE}" directory)"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status \
  || die "image SHA256 mismatch"

inventory_expected="${hf_cache}/inventories/${config_hash}.json"
inventory="$(realpath -e -- "${inventory_expected}")" \
  || die "missing config-specific HF inventory: ${inventory_expected}"
[[ "${inventory}" = "${inventory_expected}" ]] \
  || die "HF inventory path must be canonical and may not traverse symlinks"
[[ -f "${inventory}" && ! -L "${inventory}" ]] \
  || die "HF inventory must be a regular non-symlink file"
inventory_sha256="$(sha256sum -- "${inventory}" | awk '{print $1}')"
[[ "${inventory_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to hash HF inventory"

for value in \
  "${project_root}" "${scratch_root}" "${image}" "${hf_cache}" "${inventory}" \
  "${repo_root}" "${config_relative}"; do
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "unsafe sbatch export delimiter in path"
  [[ "${value}" != *":"* ]] || die "unsafe Apptainer bind delimiter in path"
done

# Close the validation/submission race on the mutable login checkout. The
# compute job performs the same identity checks again before making a detached
# exact-commit checkout.
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${git_commit}" ]] \
  || die "Git HEAD changed while preparing aggregation"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "Git worktree changed while preparing aggregation"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status \
  || die "image changed while preparing aggregation"
printf '%s  %s\n' "${inventory_sha256}" "${inventory}" \
  | sha256sum --check --status \
  || die "HF inventory changed while preparing aggregation"

slurm_log_dir="${project_root}/slurm-logs"
mkdir -p "${slurm_log_dir}" "${scratch_root}/jobs"
sbatch \
  --parsable \
  --account=sigroup \
  --chdir="${repo_root}" \
  --job-name=prorm-aggregate \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=8G \
  --partition="${partition}" \
  --time="${walltime}" \
  --output="${slurm_log_dir}/%x-%j.out" \
  --export="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_REPO_ROOT=${repo_root},PRORM_CONFIG_REL=${config_relative},PRORM_CONFIG_HASH=${config_hash},PRORM_CONFIG_FILE_SHA256=${config_file_sha256},PRORM_CONFIG_SEED_COUNT=${config_seed_count},PRORM_GIT_COMMIT=${git_commit}" \
  "${repo_root}/scripts/hpc4/aggregate.sbatch" \
  "${seed_job_pairs[@]}"
