#!/usr/bin/env bash
#SBATCH --job-name=prorm-controlled
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1

set -euo pipefail
umask 027

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
        die "unset exported ${variable}; formal execution forbids ambient container controls"
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
    die "${name} may not contain commas or newlines"
  }
}

reject_bind_delimiters() {
  local name="$1" value="$2"
  reject_export_delimiters "${name}" "${value}"
  [[ "${value}" != *":"* ]] || die "${name} may not contain ':' in an Apptainer bind path"
}

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
resolve_compat_env PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
resolve_compat_env PRORM_IMAGE SRM_IMAGE
resolve_compat_env PRORM_IMAGE_SHA256 SRM_IMAGE_SHA256
resolve_compat_env PRORM_CONFIG SRM_CONFIG
resolve_compat_env PRORM_HF_CACHE SRM_HF_CACHE
resolve_compat_env PRORM_REPO_ROOT SRM_REPO_ROOT
resolve_compat_env PRORM_GIT_COMMIT SRM_GIT_COMMIT
resolve_compat_env PRORM_HF_INVENTORY SRM_HF_INVENTORY
resolve_compat_env PRORM_HF_INVENTORY_SHA256 SRM_HF_INVENTORY_SHA256
: "${PRORM_PROJECT_ROOT:?absolute persistent project root is required}"
: "${PRORM_SCRATCH_ROOT:?absolute per-user scratch root is required}"
: "${PRORM_IMAGE:?absolute validated Apptainer image is required}"
: "${PRORM_IMAGE_SHA256:?validated image SHA256 is required}"
: "${PRORM_CONFIG:?absolute experiment config path is required}"
: "${PRORM_HF_CACHE:?absolute pre-staged Hugging Face cache is required}"
: "${PRORM_REPO_ROOT:?absolute Git repository root is required}"
: "${PRORM_GIT_COMMIT:?submit-time clean Git HEAD is required}"
: "${PRORM_HF_INVENTORY:?absolute staged asset inventory is required}"
: "${PRORM_HF_INVENTORY_SHA256:?staged asset inventory SHA256 is required}"
: "${SLURM_JOB_ID:?controlled run must execute inside a Slurm job}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || {
  die "SLURM_JOB_ID must be a positive integer"
}
[[ "${SLURM_JOB_ACCOUNT:-}" == "sigroup" ]] || {
  die "formal controlled run requires Slurm account sigroup"
}
case "${SLURM_JOB_PARTITION:-}" in
  gpu-a30|gpu-l20|gpu-rtx5880|gpu-rtx4090d) ;;
  *) die "formal controlled run is on an unsupported GPU partition" ;;
esac
reject_apptainer_control_environment

normalize_absolute_root PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
normalize_absolute_root PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "PRORM_PROJECT_ROOT and PRORM_SCRATCH_ROOT may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "PRORM_PROJECT_ROOT is not writable: ${PRORM_PROJECT_ROOT}"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "PRORM_SCRATCH_ROOT is not writable: ${PRORM_SCRATCH_ROOT}"
for path in "${PRORM_IMAGE}" "${PRORM_CONFIG}" "${PRORM_HF_INVENTORY}"; do
  [[ "${path}" = /* ]] || die "submit script must resolve file path to absolute: ${path}"
  [[ -f "${path}" ]] || die "required file does not exist: ${path}"
done
[[ "${PRORM_HF_CACHE}" = /* ]] || {
  die "submit script must resolve PRORM_HF_CACHE to an absolute path"
}
[[ -d "${PRORM_HF_CACHE}" ]] || die "PRORM_HF_CACHE is not a directory: ${PRORM_HF_CACHE}"
[[ -r "${PRORM_HF_CACHE}" ]] || die "PRORM_HF_CACHE is not readable: ${PRORM_HF_CACHE}"
image_canonical="$(realpath -e -- "${PRORM_IMAGE}")" \
  || die "PRORM_IMAGE cannot be resolved"
cache_canonical="$(realpath -e -- "${PRORM_HF_CACHE}")" \
  || die "PRORM_HF_CACHE cannot be resolved"
[[ "${image_canonical}" == "${PRORM_IMAGE}" ]] \
  || die "PRORM_IMAGE must be canonical"
[[ "${cache_canonical}" == "${PRORM_HF_CACHE}" ]] \
  || die "PRORM_HF_CACHE must be canonical"
case "${PRORM_IMAGE}" in
  "${PRORM_PROJECT_ROOT}"/*) ;;
  *) die "PRORM_IMAGE escaped PRORM_PROJECT_ROOT" ;;
esac
case "${PRORM_HF_CACHE}" in
  "${PRORM_PROJECT_ROOT}"/*) ;;
  *) die "PRORM_HF_CACHE escaped PRORM_PROJECT_ROOT" ;;
esac
[[ "${PRORM_REPO_ROOT}" = /* ]] || die "PRORM_REPO_ROOT must be absolute"
[[ -d "${PRORM_REPO_ROOT}/.git" ]] || {
  die "PRORM_REPO_ROOT is not a Git worktree root: ${PRORM_REPO_ROOT}"
}
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  die "PRORM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters"
}
[[ "${PRORM_HF_INVENTORY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  die "PRORM_HF_INVENTORY_SHA256 must be 64 lowercase hexadecimal characters"
}
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] || {
  die "PRORM_GIT_COMMIT must be a full lowercase Git object ID"
}
if ! printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status; then
  die "PRORM_IMAGE_SHA256 does not match PRORM_IMAGE: ${PRORM_IMAGE}"
fi
if ! printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${PRORM_HF_INVENTORY}" \
  | sha256sum --check --status; then
  die "PRORM_HF_INVENTORY_SHA256 does not match PRORM_HF_INVENTORY"
fi

if ! submission_repo_root="$(realpath -e -- "${PRORM_REPO_ROOT}")"; then
  die "PRORM_REPO_ROOT cannot be resolved: ${PRORM_REPO_ROOT}"
fi
[[ "${submission_repo_root}" == "${PRORM_REPO_ROOT}" ]] || {
  die "PRORM_REPO_ROOT must be canonical: ${PRORM_REPO_ROOT}"
}
if ! submitted_config_path="$(realpath -e -- "${PRORM_CONFIG}")"; then
  die "PRORM_CONFIG cannot be resolved: ${PRORM_CONFIG}"
fi
case "${submitted_config_path}" in
  "${submission_repo_root}"/*) ;;
  *) echo "formal configs must be tracked files inside PRORM_REPO_ROOT" >&2; exit 2 ;;
esac
git_commit="$(git -C "${submission_repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" == "${PRORM_GIT_COMMIT}" ]] || {
  die "compute Git HEAD ${git_commit} does not match submitted clean HEAD ${PRORM_GIT_COMMIT}"
}
if [[ -n "$(git -C "${submission_repo_root}" status --porcelain --untracked-files=normal)" ]]; then
  echo "formal controlled runs require a clean Git worktree" >&2
  exit 2
fi
config_relative="$(
  realpath --relative-to="${submission_repo_root}" "${submitted_config_path}"
)"
git -C "${submission_repo_root}" ls-files --error-unmatch -- "${config_relative}" >/dev/null

for export_name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_CONFIG PRORM_HF_CACHE PRORM_REPO_ROOT PRORM_GIT_COMMIT \
  PRORM_HF_INVENTORY PRORM_HF_INVENTORY_SHA256; do
  reject_export_delimiters "${export_name}" "${!export_name}"
done
for bind_name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_CONFIG \
  PRORM_HF_CACHE PRORM_REPO_ROOT PRORM_HF_INVENTORY; do
  reject_bind_delimiters "${bind_name}" "${!bind_name}"
done

job_dir="${PRORM_SCRATCH_ROOT}/jobs/${SLURM_JOB_ID}"
[[ ! -e "${job_dir}" && ! -L "${job_dir}" ]] || {
  die "refusing to reuse or overwrite scratch job directory: ${job_dir}"
}
mkdir -p "$(dirname "${job_dir}")" "${PRORM_PROJECT_ROOT}/runs" \
  "${PRORM_PROJECT_ROOT}/artifacts"
mkdir "${job_dir}"

# Execute every phase from a private, detached checkout of the submitted
# commit. The user's mutable login checkout is only the clone source and
# cannot change code or config after this point.
repo_root="${job_dir}/source"
git clone --quiet --no-hardlinks --no-checkout \
  "${submission_repo_root}" "${repo_root}"
git -C "${repo_root}" checkout --quiet --detach "${git_commit}"
config_path="${repo_root}/${config_relative}"
[[ -f "${config_path}" ]] || {
  die "submitted config is absent from the detached execution checkout: ${config_relative}"
}

assert_execution_checkout() {
  [[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" == "${git_commit}" ]] || {
    die "detached execution checkout changed from submitted commit ${git_commit}"
  }
  [[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] || {
    die "detached execution checkout became dirty"
  }
}
assert_execution_checkout

container=(
  apptainer exec --cleanenv --nv
  --no-mount home,cwd
  --pwd "${repo_root}"
  --bind "${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}"
  --env "PYTHONPATH=${repo_root}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1"
  --env "HF_HUB_DISABLE_TELEMETRY=1"
  --env "TOKENIZERS_PARALLELISM=false"
)

# --cleanenv is deliberate; pass only the reproducibility allowlist consumed
# by RunManifest, never the caller's complete environment.
for variable in \
  SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
  SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION SLURM_JOB_NODELIST SLURM_NNODES \
  SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_GPUS SLURM_GPUS_ON_NODE \
  SLURM_PROCID SLURM_LOCALID SLURM_NODEID \
  CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    container+=(--env "${variable}=${!variable}")
  fi
done
container+=(
  --env "PRORM_IMAGE=${PRORM_IMAGE}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_GIT_COMMIT=${git_commit}"
  --env "PRORM_HF_INVENTORY=${PRORM_HF_INVENTORY}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
  --env "SRM_IMAGE=${SRM_IMAGE}"
  --env "SRM_IMAGE_SHA256=${SRM_IMAGE_SHA256}"
  --env "SRM_GIT_COMMIT=${git_commit}"
  --env "SRM_HF_INVENTORY=${SRM_HF_INVENTORY}"
  --env "SRM_HF_INVENTORY_SHA256=${SRM_HF_INVENTORY_SHA256}"
  "${PRORM_IMAGE}"
)

mapfile -t run_info < <(
  "${container[@]}" python - "${config_path}" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import sys
from smart_reward.config import config_hash, load_config

config = load_config(sys.argv[1])
index = int(sys.argv[2])
run = config["run"]
seeds = [run["seed"]] if "seed" in run else run["seeds"]
if not 0 <= index < len(seeds):
    raise SystemExit(f"array index {index} is outside configured seed count {len(seeds)}")
print(run["name"])
print(seeds[index])
print(config_hash(config))
PY
)
[[ "${#run_info[@]}" -eq 3 ]] || { echo "failed to resolve run identity" >&2; exit 2; }
run_name="${run_info[0]}"
seed="${run_info[1]}"
config_sha="${run_info[2]}"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  die "run.name is unsafe for a project path: ${run_name}"
}
[[ "${seed}" =~ ^-?[0-9]+$ ]] || die "resolved run seed is not an integer: ${seed}"
[[ "${config_sha}" =~ ^[0-9a-f]{64}$ ]] || die "resolved config hash is invalid"

artifact_dir="${job_dir}/artifact"
project_artifact="${PRORM_PROJECT_ROOT}/artifacts/${config_sha}/${PRORM_IMAGE_SHA256}/${PRORM_HF_INVENTORY_SHA256}/${git_commit}/seed-${seed}"
project_run="${PRORM_PROJECT_ROOT}/runs/${run_name}/${config_sha}/seed-${seed}/job-${SLURM_JOB_ID}"
materialized_this_job=0
mkdir -p "$(dirname "${project_run}")"
[[ ! -e "${project_run}" && ! -L "${project_run}" ]] || {
  die "refusing to reuse or overwrite project run directory: ${project_run}"
}
mkdir "${project_run}"

atomic_sync_file() {
  local source="$1" filename="$2" destination="${project_run}/${filename}" temporary=""
  [[ -f "${source}" ]] || return 0
  [[ ! -e "${destination}" && ! -L "${destination}" ]] || {
    echo "refusing to overwrite synchronized run file: ${destination}" >&2
    return 1
  }
  temporary="$(mktemp "${project_run}/.${filename}.tmp.XXXXXX")" || return 1
  if ! cp -p -- "${source}" "${temporary}"; then
    rm -f -- "${temporary}"
    return 1
  fi
  if ! mv -T -- "${temporary}" "${destination}"; then
    rm -f -- "${temporary}"
    return 1
  fi
}

atomic_write_marker() {
  local marker="$1" workload_exit="$2" final_exit="$3"
  local destination="${project_run}/${marker}" temporary=""
  [[ ! -e "${project_run}/SUCCESS" && ! -L "${project_run}/SUCCESS" ]] || return 1
  [[ ! -e "${project_run}/FAILED" && ! -L "${project_run}/FAILED" ]] || return 1
  temporary="$(mktemp "${project_run}/.${marker}.tmp.XXXXXX")" || return 1
  if ! {
    printf 'schema_version=prorm-run-status/v1\n'
    printf 'status=%s\n' "${marker}"
    printf 'workload_exit_code=%s\n' "${workload_exit}"
    printf 'final_exit_code=%s\n' "${final_exit}"
    printf 'job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${temporary}"; then
    rm -f -- "${temporary}"
    return 1
  fi
  if ! mv -T -- "${temporary}" "${destination}"; then
    rm -f -- "${temporary}"
    return 1
  fi
}

finalize_run() {
  local workload_exit=$? sync_exit=0 final_exit marker filename
  trap - EXIT
  set +e
  for filename in \
    run-manifest.json hf-inventory-verification.json artifact-materialization.json \
    comparison.json rollout.json updated_rollouts.jsonl memory-materialize.json \
    memory-compare.json memory-rollout.json; do
    atomic_sync_file "${job_dir}/${filename}" "${filename}" || sync_exit=1
  done
  if (( workload_exit == 0 )); then
    for filename in \
      run-manifest.json hf-inventory-verification.json artifact-materialization.json \
      comparison.json rollout.json updated_rollouts.jsonl memory-compare.json \
      memory-rollout.json; do
      [[ -f "${project_run}/${filename}" ]] || {
        echo "successful workload is missing synchronized core file: ${filename}" >&2
        sync_exit=1
      }
    done
    if (( materialized_this_job == 1 )) \
      && [[ ! -f "${project_run}/memory-materialize.json" ]]; then
      echo "artifact-producing workload is missing memory-materialize.json" >&2
      sync_exit=1
    fi
  fi
  final_exit="${workload_exit}"
  if (( sync_exit != 0 && final_exit == 0 )); then
    final_exit=1
  fi
  if (( final_exit == 0 )); then marker="SUCCESS"; else marker="FAILED"; fi
  if ! atomic_write_marker "${marker}" "${workload_exit}" "${final_exit}"; then
    echo "failed to install atomic ${marker} marker in ${project_run}" >&2
    final_exit=1
  fi
  exit "${final_exit}"
}
trap finalize_run EXIT

inventory_expected="${PRORM_HF_CACHE}/inventories/${config_sha}.json"
if ! inventory_expected="$(realpath -e -- "${inventory_expected}")"; then
  die "missing inventory for compute-resolved config hash ${config_sha}"
fi
inventory_submitted="$(realpath -e -- "${PRORM_HF_INVENTORY}")"
[[ "${inventory_expected}" == "${inventory_submitted}" ]] || {
  die "submitted inventory does not match compute-resolved config hash: ${PRORM_HF_INVENTORY}"
}
if ! printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${inventory_submitted}" \
  | sha256sum --check --status; then
  die "asset inventory changed after submission: ${inventory_submitted}"
fi

verification_result="${job_dir}/hf-inventory-verification.json"
[[ ! -e "${verification_result}" && ! -L "${verification_result}" ]] || {
  die "refusing to overwrite scratch inventory verification result: ${verification_result}"
}
assert_execution_checkout
"${container[@]}" python "${repo_root}/scripts/hpc4/stage_hf_assets.py" \
  "${config_path}" "${PRORM_HF_CACHE}" --verify-only --inventory "${inventory_submitted}" \
  | tee "${verification_result}"
if ! printf '%s  %s\n' "${PRORM_HF_INVENTORY_SHA256}" "${inventory_submitted}" \
  | sha256sum --check --status; then
  die "asset inventory changed during offline verification: ${inventory_submitted}"
fi
verification_sha="$("${container[@]}" python - \
  "${verification_result}" "${config_sha}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("status") != "ok" or result.get("verify_only") is not True:
    raise SystemExit("stage_hf_assets verify-only did not report a successful verification")
if result.get("config_hash") != sys.argv[2]:
    raise SystemExit("stage_hf_assets verification config_hash does not match the run")
inventory_sha256 = result.get("inventory_sha256")
if not isinstance(inventory_sha256, str):
    raise SystemExit("stage_hf_assets verification omitted inventory_sha256")
print(inventory_sha256)
PY
)"
[[ "${verification_sha}" == "${PRORM_HF_INVENTORY_SHA256}" ]] || {
  die "offline verification SHA does not match submitted inventory SHA"
}

assert_execution_checkout

"${container[@]}" python - <<'PY'
import subprocess
import torch

subprocess.run(["git", "--version"], check=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("formal controlled run requires exactly one visible allocated GPU")
print("formal_gpu", torch.cuda.get_device_name(0))
PY

assert_execution_checkout
"${container[@]}" python -m smart_reward.cli env-report \
  "${config_path}" --seed "${seed}" --repo-root "${repo_root}" \
  --output "${job_dir}/run-manifest.json"

artifact_parent="$(dirname "${project_artifact}")"
mkdir -p "${artifact_parent}"
if [[ -e "${project_artifact}" || -L "${project_artifact}" ]]; then
  [[ -d "${project_artifact}" && -f "${project_artifact}/metadata.json" ]] || {
    die "content-addressed artifact exists but is incomplete: ${project_artifact}"
  }
  cp -a "${project_artifact}" "${artifact_dir}"
else
  command -v flock >/dev/null || die "flock is required for race-free artifact publication"
  artifact_lock="${project_artifact}.publish.lock"
  exec {artifact_lock_fd}> "${artifact_lock}"
  flock -n "${artifact_lock_fd}" || {
    die "another job is publishing the same content-addressed artifact: ${project_artifact}"
  }
  [[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] || {
    die "artifact appeared after the initial existence check; refusing publication race"
  }
  "${container[@]}" env \
    "PRORM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
    "SRM_MEMORY_REPORT=${job_dir}/memory-materialize.json" \
    python -m smart_reward.cli controlled-materialize \
    "${config_path}" "${artifact_dir}" --seed "${seed}" --device cuda
  materialized_this_job=1
  assert_execution_checkout
  [[ -f "${artifact_dir}/metadata.json" ]] || {
    die "materialized artifact is missing metadata.json: ${artifact_dir}"
  }
  artifact_staging="$(
    mktemp -d "${artifact_parent}/.seed-${seed}.publish-${SLURM_JOB_ID}.XXXXXX"
  )"
  cp -a "${artifact_dir}/." "${artifact_staging}/"
  [[ ! -e "${project_artifact}" && ! -L "${project_artifact}" ]] || {
    die "artifact destination appeared before atomic publish; refusing to overwrite it"
  }
  if ! mv -T --no-clobber -- "${artifact_staging}" "${project_artifact}"; then
    die "atomic no-replace artifact publication failed: ${project_artifact}"
  fi
  [[ ! -e "${artifact_staging}" && ! -L "${artifact_staging}" ]] || {
    die "artifact destination won the publish race; staging was not installed"
  }
  [[ -f "${project_artifact}/metadata.json" ]] || {
    die "atomic artifact publication did not install complete metadata"
  }
  flock -u "${artifact_lock_fd}"
fi

artifact_mode="reused"
if (( materialized_this_job == 1 )); then
  artifact_mode="materialized"
fi
artifact_metadata_sha256="$(
  sha256sum -- "${project_artifact}/metadata.json" | awk '{print $1}'
)"
[[ "${artifact_metadata_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
  die "failed to hash published artifact metadata"
}
artifact_evidence="${job_dir}/artifact-materialization.json"
artifact_evidence_temporary="$(mktemp "${job_dir}/.artifact-materialization.XXXXXX")"
if ! {
  printf '{\n'
  printf '  "artifact_metadata_sha256": "%s",\n' "${artifact_metadata_sha256}"
  printf '  "config_hash": "%s",\n' "${config_sha}"
  printf '  "hf_inventory_sha256": "%s",\n' "${PRORM_HF_INVENTORY_SHA256}"
  printf '  "mode": "%s",\n' "${artifact_mode}"
  printf '  "schema_version": "prorm-artifact-materialization/v1",\n'
  printf '  "seed": %s\n' "${seed}"
  printf '}\n'
} > "${artifact_evidence_temporary}"; then
  rm -f -- "${artifact_evidence_temporary}"
  die "failed to write artifact materialization evidence"
fi
mv -T -- "${artifact_evidence_temporary}" "${artifact_evidence}"

artifact_link="${project_run}/artifact"
[[ ! -e "${artifact_link}" && ! -L "${artifact_link}" ]] || {
  die "refusing to overwrite run artifact link: ${artifact_link}"
}
artifact_relative="$(realpath --relative-to="${project_run}" "${project_artifact}")"
[[ "${artifact_relative}" != /* ]] || die "artifact link target must be relative"
ln -s -- "${artifact_relative}" "${artifact_link}"

assert_execution_checkout
"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-compare.json" \
  "SRM_MEMORY_REPORT=${job_dir}/memory-compare.json" \
  python -m smart_reward.cli controlled-compare \
  "${config_path}" "${artifact_dir}" "${job_dir}/comparison.json" \
  --seed "${seed}" --device cuda --run-manifest "${job_dir}/run-manifest.json"

assert_execution_checkout
"${container[@]}" env \
  "PRORM_MEMORY_REPORT=${job_dir}/memory-rollout.json" \
  "SRM_MEMORY_REPORT=${job_dir}/memory-rollout.json" \
  python -m smart_reward.cli controlled-rollout \
  "${config_path}" "${artifact_dir}" "${job_dir}/comparison.json" \
  "${job_dir}/rollout.json" --seed "${seed}" --device cuda

assert_execution_checkout
echo "controlled workload complete; finalizing run=${run_name} seed=${seed} job=${SLURM_JOB_ID}"
