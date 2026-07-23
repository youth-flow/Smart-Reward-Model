#!/usr/bin/env bash
#SBATCH --job-name=prorm-aggregate
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

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
        die "unset exported ${variable}; formal aggregation forbids ambient container controls"
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
  [[ -f "${resolved}" && ! -L "${raw}" ]] \
    || die "${name} is not a regular non-symlink file: ${resolved}"
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
  PRORM_HF_CACHE PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256 \
  PRORM_REPO_ROOT PRORM_CONFIG_REL PRORM_CONFIG_HASH \
  PRORM_CONFIG_FILE_SHA256 PRORM_CONFIG_SEED_COUNT PRORM_GIT_COMMIT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
[[ "$#" -gt 0 ]] || die "at least one explicit seed=controlled_job_id mapping is required"
: "${SLURM_JOB_ID:?aggregation must execute inside a Slurm job}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_JOB_ID must be positive"
[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]] \
  || die "formal aggregation requires Slurm account sigroup"
case "${SLURM_JOB_PARTITION:-}" in
  amd|intel) ;;
  *) die "formal aggregation requires CPU partition amd or intel" ;;
esac
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
  || die "formal aggregation must be a single non-array CPU job"
case "${SLURM_GPUS:-0}" in
  ""|0) ;;
  *) die "formal aggregation must not request GPUs" ;;
esac
case "${SLURM_GPUS_ON_NODE:-0}" in
  ""|0) ;;
  *) die "formal aggregation must not request GPUs" ;;
esac
for gpu_variable in \
  SLURM_JOB_GPUS SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES \
  NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  [[ -z "${!gpu_variable:-}" ]] \
    || die "formal aggregation received a GPU visibility/allocation variable: ${gpu_variable}"
done
for gres_variable in \
  SLURM_JOB_GRES SLURM_STEP_GRES SLURM_TRES_PER_JOB \
  SLURM_TRES_PER_NODE SLURM_TRES_PER_TASK; do
  gres_value="${!gres_variable:-}"
  [[ "${gres_value,,}" != *gpu* ]] \
    || die "formal aggregation received GPU GRES/TRES: ${gres_variable}"
done
reject_apptainer_control_environment

for command_name in apptainer cmp flock git mktemp mv python3 readlink realpath sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

require_canonical_directory PRORM_PROJECT_ROOT
require_canonical_directory PRORM_SCRATCH_ROOT
require_canonical_directory PRORM_HF_CACHE
require_canonical_directory PRORM_REPO_ROOT
require_canonical_file PRORM_IMAGE
require_canonical_file PRORM_HF_INVENTORY
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "project and scratch roots may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "project root is not writable"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "scratch root is not writable"
require_contained PRORM_IMAGE "${PRORM_IMAGE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_HF_CACHE "${PRORM_HF_CACHE}" "${PRORM_PROJECT_ROOT}"
require_contained PRORM_HF_INVENTORY "${PRORM_HF_INVENTORY}" "${PRORM_HF_CACHE}"
[[ "${PRORM_HF_INVENTORY}" = \
  "${PRORM_HF_CACHE}/inventories/${PRORM_CONFIG_HASH}.json" ]] \
  || die "HF inventory does not match the submitted config identity"

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_HF_CACHE \
  PRORM_HF_INVENTORY PRORM_REPO_ROOT PRORM_CONFIG_REL; do
  reject_path_delimiters "${name}" "${!name}"
done
[[ "${PRORM_CONFIG_REL}" =~ ^configs/[A-Za-z0-9._-]+\.yaml$ ]] \
  || die "PRORM_CONFIG_REL is not a safe tracked config path"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
[[ "${PRORM_HF_INVENTORY_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_HF_INVENTORY_SHA256 must be lowercase SHA256"
[[ "${PRORM_CONFIG_HASH}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_CONFIG_HASH must be lowercase SHA256"
[[ "${PRORM_CONFIG_FILE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_CONFIG_FILE_SHA256 must be lowercase SHA256"
[[ "${PRORM_CONFIG_SEED_COUNT}" =~ ^[1-9][0-9]*$ ]] \
  || die "PRORM_CONFIG_SEED_COUNT must be positive"
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] \
  || die "PRORM_GIT_COMMIT must be a full lowercase object ID"
[[ "$#" -eq "${PRORM_CONFIG_SEED_COUNT}" ]] \
  || die "explicit seed/job pair count differs from the submitted config identity"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status \
  || die "image SHA256 mismatch"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status \
  || die "HF inventory SHA256 mismatch"

declare -A job_by_seed=()
declare -A seen_jobs=()
submitted_pairs=("$@")
for pair in "${submitted_pairs[@]}"; do
  [[ "${pair}" =~ ^(0|-?[1-9][0-9]*)=([1-9][0-9]*)$ ]] \
    || die "seed/job mapping must be canonical seed=positive_job_id: ${pair}"
  seed="${BASH_REMATCH[1]}"
  controlled_job="${BASH_REMATCH[2]}"
  [[ ! -v "job_by_seed[${seed}]" ]] || die "duplicate seed mapping: ${seed}"
  [[ ! -v "seen_jobs[${controlled_job}]" ]] \
    || die "controlled job ID is mapped more than once: ${controlled_job}"
  job_by_seed["${seed}"]="${controlled_job}"
  seen_jobs["${controlled_job}"]=1
done

repository_top="$(git -C "${PRORM_REPO_ROOT}" rev-parse --show-toplevel)"
[[ "${repository_top}" = "${PRORM_REPO_ROOT}" ]] || die "invalid repository root"
repo_head="$(git -C "${PRORM_REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${repo_head}" = "${PRORM_GIT_COMMIT}" ]] \
  || die "repository HEAD changed after submission"
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || die "aggregation requires the submitted repository to remain clean"
git -C "${PRORM_REPO_ROOT}" cat-file -e \
  "${PRORM_GIT_COMMIT}^{commit}" 2>/dev/null \
  || die "submitted Git commit is unavailable"
git -C "${PRORM_REPO_ROOT}" cat-file blob \
  "${PRORM_GIT_COMMIT}:${PRORM_CONFIG_REL}" \
  | sha256sum \
  | awk '{print $1}' \
  | grep -Fx "${PRORM_CONFIG_FILE_SHA256}" >/dev/null \
  || die "committed config blob does not match submitted identity"

job_dir="${PRORM_SCRATCH_ROOT}/jobs/${SLURM_JOB_ID}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "refusing to reuse scratch job directory: ${job_dir}"
mkdir -p "${PRORM_SCRATCH_ROOT}/jobs"
mkdir "${job_dir}"
[[ -d "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "scratch job path is not a real directory: ${job_dir}"
[[ "$(realpath -e -- "${job_dir}")" = "${job_dir}" ]] \
  || die "scratch job path traverses a symlink"
staging_dir=""

cleanup_failed_staging() {
  local workload_exit=$? failure_copy=""
  trap - EXIT
  set +e
  if (( workload_exit != 0 )) && [[ -n "${staging_dir}" ]] \
    && [[ -e "${staging_dir}" || -L "${staging_dir}" ]]; then
    case "${staging_dir}" in
      "${PRORM_PROJECT_ROOT}"/runs/*/.aggregate.publish-"${SLURM_JOB_ID}".*) ;;
      *)
        printf 'refusing to clean an unexpected aggregation staging path: %s\n' \
          "${staging_dir}" >&2
        exit 1
        ;;
    esac
    failure_copy="${job_dir}/failed-aggregate-staging"
    if [[ ! -e "${failure_copy}" && ! -L "${failure_copy}" ]]; then
      cp -a -- "${staging_dir}" "${failure_copy}" 2>/dev/null || true
    fi
    rm -rf --one-file-system -- "${staging_dir}"
  fi
  exit "${workload_exit}"
}
trap cleanup_failed_staging EXIT

# Freeze aggregation code and config at the submitted commit. The mutable
# login checkout is never mounted into the container.
execution_repo="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${PRORM_REPO_ROOT}" "${execution_repo}"
git -C "${execution_repo}" checkout --quiet --detach "${PRORM_GIT_COMMIT}"
execution_config="${execution_repo}/${PRORM_CONFIG_REL}"
[[ -f "${execution_config}" && ! -L "${execution_config}" ]] \
  || die "config is missing from the detached execution source"

assert_execution_checkout() {
  [[ "$(git -C "${execution_repo}" rev-parse --verify HEAD)" = "${PRORM_GIT_COMMIT}" ]] \
    || die "detached aggregation source changed from submitted commit"
  [[ -z "$(git -C "${execution_repo}" status --porcelain --untracked-files=normal)" ]] \
    || die "detached aggregation source became dirty"
  printf '%s  %s\n' "${PRORM_CONFIG_FILE_SHA256}" "${execution_config}" \
    | sha256sum --check --status \
    || die "detached config bytes do not match submitted identity"
}
assert_execution_checkout

identity_check="${job_dir}/detached-config-identity.txt"
python3 -I -S - \
  "${execution_repo}" "${PRORM_CONFIG_REL}" "${PRORM_CONFIG_HASH}" \
  "${PRORM_CONFIG_FILE_SHA256}" "${PRORM_CONFIG_SEED_COUNT}" \
  > "${identity_check}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


root = Path(sys.argv[1])
relative, expected_hash, expected_file_sha, expected_count_raw = sys.argv[2:]
expected_count = int(expected_count_raw)


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


identity = json.loads(
    (root / "configs" / "identities.json").read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")
    ),
)
if identity.get("schema_version") != "prorm-config-identities/v1":
    raise SystemExit("detached config identity schema is invalid")
entry = identity.get("configs", {}).get(relative)
expected = {
    "config_hash": expected_hash,
    "file_sha256": expected_file_sha,
    "seed_count": expected_count,
}
if entry != expected:
    raise SystemExit("detached committed config identity differs from submission")
actual_file_sha = hashlib.sha256((root / relative).read_bytes()).hexdigest()
if actual_file_sha != expected_file_sha:
    raise SystemExit("detached config file SHA256 differs from submission")
if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
    raise SystemExit("invalid expected config hash")
print("detached_config_identity=passed")
PY

container=(
  apptainer exec --cleanenv
  --no-mount home,cwd
  --pwd "${execution_repo}"
  --bind "${job_dir}:${job_dir},${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}"
  --env "PYTHONPATH=${execution_repo}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  "${PRORM_IMAGE}"
)

config_info_path="${job_dir}/config-info.txt"
"${container[@]}" python - "${execution_config}" > "${config_info_path}" <<'PY'
import sys

from smart_reward.config import config_hash, load_config


config = load_config(sys.argv[1])
run = config["run"]
seeds = run.get("seeds")
if not isinstance(seeds, list):
    raise SystemExit("formal aggregation requires config run.seeds")
print(run["name"])
print(config_hash(config))
print(len(seeds))
for seed in seeds:
    print(seed)
PY
mapfile -t config_info < "${config_info_path}"
[[ "${#config_info[@]}" -ge 4 ]] || die "failed to parse aggregation config"
run_name="${config_info[0]}"
computed_config_hash="${config_info[1]}"
computed_seed_count="${config_info[2]}"
configured_seeds=("${config_info[@]:3}")
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "run.name is unsafe for a project path: ${run_name}"
[[ "${computed_config_hash}" = "${PRORM_CONFIG_HASH}" ]] \
  || die "container-computed config hash differs from submitted identity"
[[ "${computed_seed_count}" = "${PRORM_CONFIG_SEED_COUNT}" ]] \
  || die "container-computed seed count differs from submitted identity"
[[ "${#configured_seeds[@]}" -eq "${PRORM_CONFIG_SEED_COUNT}" ]] \
  || die "container did not emit every configured seed"

ordered_pairs=()
comparisons=()
rollouts=()
for seed in "${configured_seeds[@]}"; do
  [[ "${seed}" =~ ^(0|-?[1-9][0-9]*)$ ]] \
    || die "config contains a non-canonical integer seed: ${seed}"
  [[ -v "job_by_seed[${seed}]" ]] \
    || die "explicit mappings do not exactly match config run.seeds; missing ${seed}"
  controlled_job="${job_by_seed[${seed}]}"
  ordered_pairs+=("${seed}=${controlled_job}")
  run_directory="${PRORM_PROJECT_ROOT}/runs/${run_name}/${PRORM_CONFIG_HASH}/seed-${seed}/job-${controlled_job}"
  comparisons+=("${run_directory}/comparison.json")
  rollouts+=("${run_directory}/rollout.json")
done
[[ "${#job_by_seed[@]}" -eq "${#configured_seeds[@]}" ]] \
  || die "explicit mappings contain a seed not declared by config run.seeds"

campaign_root="${PRORM_PROJECT_ROOT}/runs/${run_name}/${PRORM_CONFIG_HASH}"
[[ -d "${campaign_root}" && ! -L "${campaign_root}" ]] \
  || die "campaign root is not a real directory: ${campaign_root}"
[[ "$(realpath -e -- "${campaign_root}")" = "${campaign_root}" ]] \
  || die "campaign root traverses a symlink"
aggregate_final="${campaign_root}/aggregate"
aggregate_lock="${campaign_root}/.aggregate.lock"
if [[ -e "${aggregate_lock}" || -L "${aggregate_lock}" ]]; then
  [[ -f "${aggregate_lock}" && ! -L "${aggregate_lock}" ]] \
    || die "aggregation lock path is not a regular non-symlink file"
fi
exec {aggregate_lock_fd}> "${aggregate_lock}"
flock -n "${aggregate_lock_fd}" \
  || die "another formal aggregation is active for this campaign"
[[ ! -e "${aggregate_final}" && ! -L "${aggregate_final}" ]] \
  || die "refusing to overwrite an existing formal aggregate"

validate_sources() {
  local output="$1"
  "${container[@]}" python - \
    "${output}" "${execution_config}" "${PRORM_CONFIG_HASH}" "${run_name}" \
    "${PRORM_PROJECT_ROOT}" "${PRORM_IMAGE_SHA256}" \
    "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_GIT_COMMIT}" \
    "${ordered_pairs[@]}" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from smart_reward.config import config_hash, load_config


(
    output_raw,
    config_raw,
    expected_config_hash,
    expected_run_name,
    project_raw,
    expected_image_sha,
    expected_inventory_sha,
    expected_git_commit,
    *pair_args,
) = sys.argv[1:]
output = Path(output_raw)
config_path = Path(config_raw)
project_root = Path(project_raw)


def fail(message):
    raise SystemExit(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path, max_bytes=64 * 1024 * 1024):
    if path.stat().st_size > max_bytes:
        fail(f"JSON input exceeds safety limit: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        fail(f"invalid strict JSON {path}: {error}")


def require_real_directory(path, label):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        fail(f"missing {label}: {path}")
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        fail(f"{label} must be a real directory: {path}")
    if path.resolve(strict=True) != path:
        fail(f"{label} traverses a symlink: {path}")


def require_regular(path, label):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        fail(f"missing {label}: {path}")
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        fail(f"{label} must be a regular non-symlink file: {path}")


def parse_success(path, expected_job):
    require_regular(path, "SUCCESS marker")
    if path.stat().st_size > 16 * 1024:
        fail(f"SUCCESS marker is too large: {path}")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            fail(f"malformed SUCCESS marker line: {path}")
        key, value = line.split("=", 1)
        if not key or key in values:
            fail(f"duplicate or empty SUCCESS marker key: {path}")
        values[key] = value
    if set(values) != {
        "schema_version",
        "status",
        "workload_exit_code",
        "final_exit_code",
        "job_id",
        "created_at_utc",
    }:
        fail(f"unexpected SUCCESS marker schema: {path}")
    expected_values = {
        "schema_version": "prorm-run-status/v1",
        "status": "SUCCESS",
        "workload_exit_code": "0",
        "final_exit_code": "0",
        "job_id": str(expected_job),
    }
    if any(values.get(key) != value for key, value in expected_values.items()):
        fail(f"SUCCESS marker does not certify a successful controlled job: {path}")
    if not values["created_at_utc"]:
        fail(f"SUCCESS marker does not certify a successful controlled job: {path}")


def compatibility_value(mapping, canonical, legacy):
    canonical_value = mapping.get(canonical)
    legacy_value = mapping.get(legacy)
    if canonical_value is not None and legacy_value is not None:
        if canonical_value != legacy_value:
            fail(f"conflicting {canonical}/{legacy} identity in run manifest")
    return canonical_value if canonical_value is not None else legacy_value


def run_environment(manifest, *, seed, expected_job):
    if manifest.get("schema_version") != "smart-reward-run/v1":
        fail("run-manifest.json has the wrong schema")
    if manifest.get("config_hash") != expected_config_hash:
        fail("run-manifest.json config hash mismatch")
    if manifest.get("selected_seed") != seed:
        fail("run-manifest.json selected seed mismatch")
    git_state = manifest.get("git")
    slurm = manifest.get("slurm")
    torch_state = manifest.get("torch")
    if not all(isinstance(value, dict) for value in (git_state, slurm, torch_state)):
        fail("run-manifest.json is missing Git/Slurm/Torch identity")
    if git_state != {"commit": expected_git_commit, "dirty": False}:
        fail("run-manifest.json Git identity mismatch")
    image = compatibility_value(slurm, "PRORM_IMAGE_SHA256", "SRM_IMAGE_SHA256")
    inventory = compatibility_value(
        slurm,
        "PRORM_HF_INVENTORY_SHA256",
        "SRM_HF_INVENTORY_SHA256",
    )
    commit = compatibility_value(slurm, "PRORM_GIT_COMMIT", "SRM_GIT_COMMIT")
    gpus = torch_state.get("gpus")
    gpu_names = (
        [value.get("name") for value in gpus if isinstance(value, dict)]
        if isinstance(gpus, list)
        else []
    )
    if (
        image != expected_image_sha
        or inventory != expected_inventory_sha
        or commit != expected_git_commit
        or slurm.get("SLURM_JOB_ID") != str(expected_job)
        or slurm.get("SLURM_JOB_ACCOUNT") != "sigroup"
        or slurm.get("SLURM_JOB_PARTITION")
        not in {"gpu-a30", "gpu-l20", "gpu-rtx5880", "gpu-rtx4090d"}
        or torch_state.get("cuda_available") is not True
        or torch_state.get("gpu_count") != 1
        or len(gpu_names) != 1
        or not isinstance(gpu_names[0], str)
        or not gpu_names[0]
    ):
        fail("run-manifest.json lacks the required formal producer identity")
    return {
        "formal": True,
        "git_commit": expected_git_commit,
        "image_sha256": expected_image_sha,
        "hf_inventory_sha256": expected_inventory_sha,
        "account": "sigroup",
        "partition": slurm["SLURM_JOB_PARTITION"],
        "gpu_models": gpu_names,
    }


config = load_config(config_path)
run = config["run"]
configured_seeds = run.get("seeds")
if run.get("name") != expected_run_name or not isinstance(configured_seeds, list):
    fail("config run identity differs during source validation")
if config_hash(config) != expected_config_hash:
    fail("semantic config identity differs during source validation")

pairs = []
for pair in pair_args:
    seed_raw, job_raw = pair.split("=", 1)
    pairs.append((int(seed_raw), int(job_raw)))
if [seed for seed, _ in pairs] != configured_seeds:
    fail("ordered explicit seed mappings differ from config run.seeds")

records = []
for seed, controlled_job in pairs:
    run_dir = (
        project_root
        / "runs"
        / expected_run_name
        / expected_config_hash
        / f"seed-{seed}"
        / f"job-{controlled_job}"
    )
    require_real_directory(run_dir, "controlled run directory")
    success_path = run_dir / "SUCCESS"
    parse_success(success_path, controlled_job)
    if os.path.lexists(run_dir / "FAILED"):
        fail(f"successful controlled run also contains FAILED: {run_dir}")

    source_names = (
        "comparison.json",
        "rollout.json",
        "run-manifest.json",
        "updated_rollouts.jsonl",
    )
    for name in source_names:
        require_regular(run_dir / name, name)

    expected_artifact = (
        project_root
        / "artifacts"
        / expected_config_hash
        / expected_image_sha
        / expected_inventory_sha
        / expected_git_commit
        / f"seed-{seed}"
    )
    require_real_directory(expected_artifact, "content-addressed artifact")
    metadata_path = expected_artifact / "metadata.json"
    require_regular(metadata_path, "artifact metadata")
    artifact_link = run_dir / "artifact"
    try:
        link_mode = artifact_link.lstat().st_mode
    except FileNotFoundError:
        fail(f"missing artifact symlink: {artifact_link}")
    if not stat.S_ISLNK(link_mode):
        fail(f"run artifact must be a relative symlink: {artifact_link}")
    link_target = os.readlink(artifact_link)
    if Path(link_target).is_absolute() or "\n" in link_target or "\r" in link_target:
        fail(f"run artifact symlink target is not safe and relative: {artifact_link}")
    if artifact_link.resolve(strict=True) != expected_artifact:
        fail(f"run artifact symlink resolves to the wrong content address: {artifact_link}")

    comparison_path = run_dir / "comparison.json"
    rollout_path = run_dir / "rollout.json"
    manifest_path = run_dir / "run-manifest.json"
    updated_path = run_dir / "updated_rollouts.jsonl"
    comparison = read_json(comparison_path)
    rollout = read_json(rollout_path)
    manifest = read_json(manifest_path)
    artifact_metadata = read_json(metadata_path, max_bytes=16 * 1024 * 1024)
    if not all(
        isinstance(value, dict)
        for value in (comparison, rollout, manifest, artifact_metadata)
    ):
        fail(f"controlled source JSON root is not an object: {run_dir}")

    environment = run_environment(
        manifest,
        seed=seed,
        expected_job=controlled_job,
    )
    success_sha = sha256_file(success_path)
    manifest_sha = sha256_file(manifest_path)
    comparison_sha = sha256_file(comparison_path)
    artifact_metadata_sha = sha256_file(metadata_path)
    updated_sha = sha256_file(updated_path)
    if (
        comparison.get("config_hash") != expected_config_hash
        or comparison.get("seed") != seed
        or comparison.get("artifact_dir") != "artifact"
        or comparison.get("artifact_metadata_sha256") != artifact_metadata_sha
        or comparison.get("run_manifest") != "run-manifest.json"
        or comparison.get("run_manifest_sha256") != manifest_sha
        or comparison.get("environment_identity") != environment
    ):
        fail(f"comparison identity/artifact binding is invalid: {comparison_path}")
    if (
        artifact_metadata.get("schema") != "controlled-feature-artifact/v1"
        or artifact_metadata.get("config_hash") != expected_config_hash
        or artifact_metadata.get("seed") != seed
    ):
        fail(f"artifact metadata identity is invalid: {metadata_path}")
    evidence = artifact_metadata.get("evidence")
    producer = evidence.get("producer") if isinstance(evidence, dict) else None
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != "phase1-materialization/v1"
        or evidence.get("config_sha256") != expected_config_hash
        or producer
        != {
            "git_commit": expected_git_commit,
            "image_sha256": expected_image_sha,
            "hf_inventory_sha256": expected_inventory_sha,
        }
    ):
        fail(f"artifact producer identity is invalid: {metadata_path}")
    if (
        rollout.get("config_hash") != expected_config_hash
        or rollout.get("seed") != seed
        or rollout.get("artifact_dir") != "artifact"
        or rollout.get("comparison_json") != "comparison.json"
        or rollout.get("updated_rollouts_jsonl") != "updated_rollouts.jsonl"
        or rollout.get("artifact_metadata_sha256") != artifact_metadata_sha
        or rollout.get("comparison_sha256") != comparison_sha
        or rollout.get("run_manifest_sha256") != manifest_sha
        or rollout.get("updated_rollouts_sha256") != updated_sha
        or rollout.get("environment_identity") != environment
    ):
        fail(f"rollout identity/source binding is invalid: {rollout_path}")

    records.append(
        {
            "seed": seed,
            "controlled_job_id": controlled_job,
            "run_directory": run_dir.relative_to(project_root).as_posix(),
            "artifact_directory": expected_artifact.relative_to(project_root).as_posix(),
            "artifact_symlink_target": link_target,
            "artifact_metadata_sha256": artifact_metadata_sha,
            "comparison_sha256": comparison_sha,
            "rollout_sha256": sha256_file(rollout_path),
            "run_manifest_sha256": manifest_sha,
            "success_sha256": success_sha,
            "updated_rollouts_sha256": updated_sha,
            "environment_identity": environment,
        }
    )

payload = {
    "schema_version": "prorm-aggregation-sources/v1",
    "config_hash": expected_config_hash,
    "git_commit": expected_git_commit,
    "image_sha256": expected_image_sha,
    "hf_inventory_sha256": expected_inventory_sha,
    "sources": records,
}
serialized = json.dumps(
    payload,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
) + "\n"
with output.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(serialized)
PY
}

source_records_before="${job_dir}/aggregation-sources-before.json"
source_records_after="${job_dir}/aggregation-sources-after.json"
source_records_final="${job_dir}/aggregation-sources-final.json"
validate_sources "${source_records_before}"

staging_dir="$(
  mktemp -d "${campaign_root}/.aggregate.publish-${SLURM_JOB_ID}.XXXXXX"
)"
[[ "$(dirname "${staging_dir}")" = "${campaign_root}" ]] \
  || die "aggregation staging directory is not a sibling of the final directory"
[[ -d "${staging_dir}" && ! -L "${staging_dir}" ]] \
  || die "aggregation staging path is not a real directory"
[[ ! -e "${aggregate_final}" && ! -L "${aggregate_final}" ]] \
  || die "aggregate destination appeared before CLI execution"

assert_execution_checkout
aggregate_cli_log="${job_dir}/aggregate-cli.log"
"${container[@]}" python -m smart_reward.cli aggregate-results \
  "${execution_config}" "${staging_dir}/aggregate.json" \
  "${comparisons[@]}" \
  --repo-root "${execution_repo}" \
  --rollouts "${rollouts[@]}" \
  2>&1 | tee "${aggregate_cli_log}"
assert_execution_checkout
[[ -f "${staging_dir}/aggregate.json" && ! -L "${staging_dir}/aggregate.json" ]] \
  || die "aggregate-results did not create a regular aggregate.json"

# Re-run the complete source preflight after the semantic CLI audit. Identical
# records prove that every source file and artifact metadata file remained
# byte-identical across aggregation.
validate_sources "${source_records_after}"
cmp -s -- "${source_records_before}" "${source_records_after}" \
  || die "controlled sources changed while aggregate-results was running"

aggregate_verification="${job_dir}/aggregate-verification.json"
"${container[@]}" python - \
  "${staging_dir}/aggregate.json" "${source_records_after}" \
  "${aggregate_verification}" "${PRORM_CONFIG_REL}" <<'PY'
import json
import sys
from pathlib import Path


aggregate_path = Path(sys.argv[1])
records_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
config_relative = sys.argv[4]


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_object(path):
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value


aggregate = read_object(aggregate_path)
records = read_object(records_path)
sources = records["sources"]
if aggregate.get("schema_version") != "paired-seed-aggregate/v2":
    raise SystemExit("aggregate has the wrong schema")
if aggregate.get("config_hash") != records["config_hash"]:
    raise SystemExit("aggregate config identity mismatch")
if aggregate.get("num_seeds") != len(sources):
    raise SystemExit("aggregate seed count mismatch")
environments = [record["environment_identity"] for record in sources]
if not environments or any(value != environments[0] for value in environments[1:]):
    raise SystemExit("preflight producer environments are not identical")
if aggregate.get("environment_identity") != environments[0]:
    raise SystemExit("aggregate producer environment mismatch")
if aggregate.get("aggregation_source") != {
    "git_commit": records["git_commit"],
    "git_dirty": False,
    "config_path": config_relative,
}:
    raise SystemExit("aggregate source checkout identity mismatch")

expected_sources = []
for record in sorted(sources, key=lambda value: value["seed"]):
    prefix = (
        f"../seed-{record['seed']}/"
        f"job-{record['controlled_job_id']}"
    )
    expected_sources.append(
        {
            "seed": record["seed"],
            "comparison_path": f"{prefix}/comparison.json",
            "comparison_sha256": record["comparison_sha256"],
            "artifact_metadata_sha256": record["artifact_metadata_sha256"],
            "run_manifest_path": f"{prefix}/run-manifest.json",
            "run_manifest_sha256": record["run_manifest_sha256"],
            "environment_identity": record["environment_identity"],
            "rollout_path": f"{prefix}/rollout.json",
            "rollout_sha256": record["rollout_sha256"],
            "updated_rollouts_path": f"{prefix}/updated_rollouts.jsonl",
            "updated_rollouts_sha256": record["updated_rollouts_sha256"],
        }
    )
if aggregate.get("sources") != expected_sources:
    raise SystemExit("aggregate source paths or SHA256 identities are not exact")
evidence = aggregate.get("pre_registered_evidence")
if not isinstance(evidence, dict) or evidence.get("status") not in {
    "passed",
    "not_passed",
}:
    raise SystemExit("aggregate has an invalid preregistered evidence status")
if evidence.get("supports_pre_registered_claim") is not (
    evidence["status"] == "passed"
):
    raise SystemExit("aggregate preregistered evidence boolean is inconsistent")

verification = {
    "schema_version": "prorm-aggregate-verification/v1",
    "status": "passed",
    "pre_registered_evidence_status": evidence["status"],
    "producer_environment": environments[0],
}
with Path(output_path).open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(
        verification,
        stream,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream.write("\n")
PY

aggregate_sha256="$(
  sha256sum -- "${staging_dir}/aggregate.json" | awk '{print $1}'
)"
source_validation_sha256="$(
  sha256sum -- "${source_records_after}" | awk '{print $1}'
)"
cli_log_sha256="$(sha256sum -- "${aggregate_cli_log}" | awk '{print $1}')"
[[ "${aggregate_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to hash aggregate.json"

sidecar_temporary="$(mktemp "${staging_dir}/.aggregate.json.sha256.XXXXXX")"
printf '%s  aggregate.json\n' "${aggregate_sha256}" > "${sidecar_temporary}"
mv -T -- "${sidecar_temporary}" "${staging_dir}/aggregate.json.sha256"

manifest_path="${staging_dir}/aggregation-manifest.json"
python3 -I -S - \
  "${staging_dir}/aggregate.json" "${source_records_after}" \
  "${aggregate_verification}" "${manifest_path}" \
  "${SLURM_JOB_ID}" "${SLURM_JOB_ACCOUNT}" "${SLURM_JOB_PARTITION}" \
  "$(hostname)" "${PRORM_CONFIG_REL}" "${PRORM_CONFIG_HASH}" \
  "${PRORM_CONFIG_FILE_SHA256}" "${PRORM_GIT_COMMIT}" \
  "${PRORM_IMAGE_SHA256}" "${PRORM_HF_INVENTORY_SHA256}" \
  "${aggregate_sha256}" "${source_validation_sha256}" "${cli_log_sha256}" <<'PY'
import datetime
import json
import sys
from pathlib import Path


(
    aggregate_raw,
    records_raw,
    verification_raw,
    output_raw,
    job_raw,
    account,
    partition,
    node,
    config_path,
    config_hash,
    config_file_sha,
    git_commit,
    image_sha,
    inventory_sha,
    aggregate_sha,
    source_validation_sha,
    cli_log_sha,
) = sys.argv[1:]


def read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return value


aggregate = read_object(aggregate_raw)
records = read_object(records_raw)
verification = read_object(verification_raw)
evidence = aggregate["pre_registered_evidence"]
manifest = {
    "schema_version": "prorm-aggregation-manifest/v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z"),
    "aggregation_job": {
        "slurm_job_id": int(job_raw),
        "account": account,
        "partition": partition,
        "node": node,
        "gpu_requested": False,
    },
    "config": {
        "path": config_path,
        "config_hash": config_hash,
        "file_sha256": config_file_sha,
    },
    "git_commit": git_commit,
    "image_sha256": image_sha,
    "hf_inventory_sha256": inventory_sha,
    "seed_to_controlled_job": [
        {
            "seed": record["seed"],
            "controlled_job_id": record["controlled_job_id"],
        }
        for record in records["sources"]
    ],
    "sources": records["sources"],
    "source_validation_sha256": source_validation_sha,
    "aggregate": {
        "path": "aggregate.json",
        "sha256": aggregate_sha,
        "cli_log_sha256": cli_log_sha,
    },
    "producer_environment": verification["producer_environment"],
    "pre_registered_evidence": evidence,
}
serialized = json.dumps(
    manifest,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
) + "\n"
with Path(output_raw).open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(serialized)
PY

manifest_sha256="$(sha256sum -- "${manifest_path}" | awk '{print $1}')"
success_temporary="$(mktemp "${staging_dir}/.SUCCESS.XXXXXX")"
{
  printf 'schema_version=prorm-aggregation-status/v1\n'
  printf 'status=SUCCESS\n'
  printf 'job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'aggregate_sha256=%s\n' "${aggregate_sha256}"
  printf 'manifest_sha256=%s\n' "${manifest_sha256}"
  python3 -I -S - "${aggregate_verification}" <<'PY'
import json
import sys
from pathlib import Path


value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"pre_registered_evidence_status={value['pre_registered_evidence_status']}")
PY
  printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${success_temporary}"
mv -T -- "${success_temporary}" "${staging_dir}/SUCCESS"

python3 -I -S - \
  "${staging_dir}" "${aggregate_sha256}" "${manifest_sha256}" \
  "${SLURM_JOB_ID}" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path


root = Path(sys.argv[1])
expected_aggregate_sha, expected_manifest_sha = sys.argv[2:4]
expected_job_id = sys.argv[4]
expected_names = {
    "SUCCESS",
    "aggregate.json",
    "aggregate.json.sha256",
    "aggregation-manifest.json",
}
if {path.name for path in root.iterdir()} != expected_names:
    raise SystemExit("aggregation staging directory has unexpected contents")
for path in root.iterdir():
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SystemExit(f"aggregation output is not a regular non-symlink file: {path}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(root / "aggregate.json") != expected_aggregate_sha:
    raise SystemExit("aggregate.json changed before publication")
if sha256(root / "aggregation-manifest.json") != expected_manifest_sha:
    raise SystemExit("aggregation-manifest.json changed before publication")
if (root / "aggregate.json.sha256").read_text(encoding="utf-8") != (
    f"{expected_aggregate_sha}  aggregate.json\n"
):
    raise SystemExit("aggregate.json.sha256 sidecar is invalid")
status = {}
for line in (root / "SUCCESS").read_text(encoding="utf-8").splitlines():
    key, value = line.split("=", 1)
    if key in status:
        raise SystemExit("SUCCESS contains a duplicate key")
    status[key] = value
if (
    set(status)
    != {
        "schema_version",
        "status",
        "job_id",
        "aggregate_sha256",
        "manifest_sha256",
        "pre_registered_evidence_status",
        "created_at_utc",
    }
    or status.get("schema_version") != "prorm-aggregation-status/v1"
    or status.get("status") != "SUCCESS"
    or status.get("job_id") != expected_job_id
    or status.get("aggregate_sha256") != expected_aggregate_sha
    or status.get("manifest_sha256") != expected_manifest_sha
    or status.get("pre_registered_evidence_status") not in {"passed", "not_passed"}
):
    raise SystemExit("aggregation SUCCESS marker is invalid")
PY

validate_sources "${source_records_final}"
cmp -s -- "${source_records_before}" "${source_records_final}" \
  || die "controlled sources changed before atomic aggregation publication"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status \
  || die "image changed before atomic aggregation publication"
printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status \
  || die "HF inventory changed before atomic aggregation publication"

assert_execution_checkout
[[ ! -e "${aggregate_final}" && ! -L "${aggregate_final}" ]] \
  || die "aggregate destination appeared before atomic publication"
if ! mv -T --no-clobber -- "${staging_dir}" "${aggregate_final}"; then
  die "atomic no-overwrite aggregation publication failed"
fi
[[ ! -e "${staging_dir}" && ! -L "${staging_dir}" ]] \
  || die "aggregate destination won a publication race; staging was not installed"
[[ -d "${aggregate_final}" && ! -L "${aggregate_final}" ]] \
  || die "atomic publication did not install a real aggregate directory"
staging_dir=""
trap - EXIT
flock -u "${aggregate_lock_fd}" || true
printf 'formal aggregate published: %s\n' "${aggregate_final}" || true
exit 0
