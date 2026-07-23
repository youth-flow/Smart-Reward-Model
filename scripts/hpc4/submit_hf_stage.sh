#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ $# -ne 3 ]]; then
  die "usage: $0 <config.yaml> <cpu-partition> <walltime>"
fi

config_input="$1"
partition="$2"
walltime="$3"
case "${partition}" in
  amd|intel) ;;
  *) die "HF staging partition must be amd or intel" ;;
esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for variable in $(compgen -e); do
  case "${variable}" in
    APPTAINER*|SINGULARITY*)
      die "unset exported ${variable}; staging submission forbids ambient container controls"
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "staging submission requires a clean repository"
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid Git HEAD"

config="$(realpath -e -- "${config_input}")"
case "${config}" in
  "${repo_root}"/configs/*.yaml) ;;
  *) die "config must be a tracked configs/*.yaml file" ;;
esac
config_relative="$(realpath --relative-to="${repo_root}" "${config}")"
git -C "${repo_root}" ls-files --error-unmatch -- "${config_relative}" >/dev/null

identity_relative="configs/identities.json"
command -v python3 >/dev/null 2>&1 || die "python3 is required to read config identities"
config_worktree_sha256="$(sha256sum -- "${config}" | awk '{print $1}')"
mapfile -t config_info < <(
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
)
[[ "${#config_info[@]}" -eq 3 ]] || die "failed to resolve committed config identity"
config_seed_count="${config_info[0]}"
config_hash="${config_info[1]}"
config_file_sha256="${config_info[2]}"
[[ "${config_seed_count}" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid committed config seed count"
[[ "${config_hash}" =~ ^[0-9a-f]{64}$ ]] || die "invalid committed config hash"
[[ "${config_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid committed config file SHA256"

project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_SCRATCH_ROOT}")"
[[ "${project_root}" != "/" && "${scratch_root}" != "/" ]] \
  || die "project and scratch roots may not be /"
case "${project_root}" in
  "${scratch_root}"|"${scratch_root}"/*) die "project and scratch roots overlap" ;;
esac
case "${scratch_root}" in
  "${project_root}"|"${project_root}"/*) die "project and scratch roots overlap" ;;
esac

resolve_project_path() {
  local raw="$1" kind="$2" candidate="" resolved=""
  if [[ "${raw}" = /* ]]; then candidate="${raw}"; else candidate="${project_root}/${raw}"; fi
  resolved="$(realpath -e -- "${candidate}")" || die "path cannot be resolved: ${candidate}"
  case "${resolved}" in
    "${project_root}"|"${project_root}"/*) ;;
    *) die "project-relative path escaped project root: ${raw}" ;;
  esac
  case "${kind}" in
    file) [[ -f "${resolved}" ]] || die "not a file: ${resolved}" ;;
    directory) [[ -d "${resolved}" ]] || die "not a directory: ${resolved}" ;;
  esac
  printf '%s\n' "${resolved}"
}

image="$(resolve_project_path "${PRORM_IMAGE}" file)"
cache_candidate="$(
  if [[ "${PRORM_HF_CACHE}" = /* ]]; then
    realpath -m -- "${PRORM_HF_CACHE}"
  else
    realpath -m -- "${project_root}/${PRORM_HF_CACHE}"
  fi
)"
case "${cache_candidate}" in
  "${project_root}"|"${project_root}"/*) ;;
  *) die "PRORM_HF_CACHE escapes PRORM_PROJECT_ROOT" ;;
esac
mkdir -p -- "${cache_candidate}"
hf_cache="$(resolve_project_path "${cache_candidate}" directory)"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status \
  || die "image SHA256 mismatch"

for value in \
  "${project_root}" "${scratch_root}" "${image}" "${hf_cache}" \
  "${repo_root}" "${config}"; do
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "unsafe sbatch export delimiter in path"
  [[ "${value}" != *":"* ]] || die "unsafe Apptainer bind delimiter in path"
done

slurm_log_dir="${project_root}/slurm-logs"
mkdir -p "${slurm_log_dir}" "${project_root}/system-reports" "${scratch_root}/jobs"
sbatch \
  --parsable \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --time="${walltime}" \
  --output="${slurm_log_dir}/%x-%j.out" \
  --export="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_REPO_ROOT=${repo_root},PRORM_CONFIG=${config},PRORM_CONFIG_HASH=${config_hash},PRORM_CONFIG_FILE_SHA256=${config_file_sha256},PRORM_GIT_COMMIT=${git_commit}" \
  "${repo_root}/scripts/hpc4/hf_stage.sbatch"
