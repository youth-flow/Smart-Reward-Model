#!/usr/bin/env bash
#SBATCH --job-name=prorm-gpu-smoke
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00

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

reject_bind_delimiters() {
  local name="$1" value="$2"
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || {
    die "${name} may not contain commas or newlines"
  }
  [[ "${value}" != *":"* ]] || die "${name} may not contain ':' in an Apptainer bind path"
}

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
resolve_compat_env PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
resolve_compat_env PRORM_IMAGE SRM_IMAGE
resolve_compat_env PRORM_IMAGE_SHA256 SRM_IMAGE_SHA256
resolve_compat_env PRORM_REPO_ROOT SRM_REPO_ROOT
: "${PRORM_PROJECT_ROOT:?absolute persistent project root is required}"
: "${PRORM_SCRATCH_ROOT:?absolute per-user scratch root is required}"
: "${PRORM_IMAGE:?set PRORM_IMAGE to the absolute path of the validated Apptainer image}"
: "${PRORM_IMAGE_SHA256:?set PRORM_IMAGE_SHA256 to the expected lowercase SHA256}"
: "${PRORM_REPO_ROOT:?set PRORM_REPO_ROOT to the absolute repository root}"
: "${SLURM_JOB_ID:?GPU smoke must execute inside a Slurm job}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || {
  die "SLURM_JOB_ID must be a positive integer"
}
reject_apptainer_control_environment

normalize_absolute_root PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
normalize_absolute_root PRORM_SCRATCH_ROOT SRM_SCRATCH_ROOT
roots_overlap "${PRORM_PROJECT_ROOT}" "${PRORM_SCRATCH_ROOT}" \
  && die "PRORM_PROJECT_ROOT and PRORM_SCRATCH_ROOT may not be equal or nested"
[[ -w "${PRORM_PROJECT_ROOT}" ]] || die "PRORM_PROJECT_ROOT is not writable: ${PRORM_PROJECT_ROOT}"
[[ -w "${PRORM_SCRATCH_ROOT}" ]] || die "PRORM_SCRATCH_ROOT is not writable: ${PRORM_SCRATCH_ROOT}"
[[ "${PRORM_REPO_ROOT}" = /* ]] || die "PRORM_REPO_ROOT must be absolute"
[[ -d "${PRORM_REPO_ROOT}/.git" ]] || {
  die "PRORM_REPO_ROOT is not a Git worktree root: ${PRORM_REPO_ROOT}"
}
[[ "${PRORM_IMAGE}" = /* ]] || {
  die "submit script must resolve PRORM_IMAGE to an absolute path"
}
[[ -f "${PRORM_IMAGE}" ]] || die "PRORM_IMAGE is not a file: ${PRORM_IMAGE}"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  die "PRORM_IMAGE_SHA256 must be 64 lowercase hexadecimal characters"
}
if ! printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status; then
  die "PRORM_IMAGE_SHA256 does not match PRORM_IMAGE: ${PRORM_IMAGE}"
fi

repo_root="${PRORM_REPO_ROOT}"
for bind_name in PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_REPO_ROOT; do
  reject_bind_delimiters "${bind_name}" "${!bind_name}"
done

container_visibility=()
for variable in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    container_visibility+=(--env "${variable}=${!variable}")
  fi
done

job_dir="${PRORM_SCRATCH_ROOT}/jobs/${SLURM_JOB_ID}"
mkdir -p "${job_dir}" "${PRORM_PROJECT_ROOT}/system-reports"

report="${job_dir}/gpu-smoke-${SLURM_JOB_ID}.txt"
persisted_report="${PRORM_PROJECT_ROOT}/system-reports/gpu-smoke-${SLURM_JOB_ID}.txt"
finalize_report() {
  local workload_exit=$? final_exit temporary=""
  final_exit="${workload_exit}"
  trap - EXIT
  set +e
  if [[ ! -f "${report}" ]]; then
    echo "GPU smoke report was not created: ${report}" >&2
    exit 1
  fi
  if [[ -e "${persisted_report}" || -L "${persisted_report}" ]]; then
    echo "refusing to overwrite existing GPU smoke report: ${persisted_report}" >&2
    exit 1
  fi
  temporary="$(mktemp \
    "${PRORM_PROJECT_ROOT}/system-reports/.gpu-smoke-${SLURM_JOB_ID}.XXXXXX")" \
    || exit 1
  if ! cp -p -- "${report}" "${temporary}"; then
    rm -f -- "${temporary}"
    exit 1
  fi
  if ! {
    printf 'workload_exit_code=%s\n' "${workload_exit}"
    if (( workload_exit == 0 )); then
      printf 'status=passed\n'
    else
      printf 'status=failed\n'
    fi
  } >> "${temporary}"; then
    rm -f -- "${temporary}"
    exit 1
  fi
  if ! mv -T -- "${temporary}" "${persisted_report}"; then
    rm -f -- "${temporary}"
    final_exit=1
  fi
  exit "${final_exit}"
}
trap finalize_report EXIT
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "git_commit=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "image_sha256=${PRORM_IMAGE_SHA256}"
  hostname
  nvidia-smi
  echo "== allowlisted Slurm identity =="
  for variable in \
    SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_JOB_NAME \
    SLURM_JOB_ACCOUNT SLURM_CLUSTER_NAME SLURM_JOB_PARTITION \
    SLURM_JOB_NODELIST SLURM_NNODES SLURM_NTASKS SLURM_CPUS_PER_TASK \
    SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_JOB_GPUS \
    SLURM_PROCID SLURM_LOCALID SLURM_NODEID; do
    if [[ -v "${variable}" ]]; then
      printf '%s=%s\n' "${variable}" "${!variable}"
    fi
  done
} | tee "${report}"

apptainer exec --cleanenv --nv \
  --bind "${repo_root}:${repo_root}" \
  --env "PYTHONPATH=${repo_root}/src" \
  "${container_visibility[@]}" \
  "${PRORM_IMAGE}" \
  python - <<'PY' | tee -a "${report}"
import torch
import transformers
from importlib.metadata import version
from packaging.version import Version

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("transformers", transformers.__version__)
for package in ("accelerate", "datasets", "peft", "safetensors"):
    print(package, version(package))
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("GPU smoke failed: CUDA GPU is not visible inside the image")
if torch.cuda.device_count() != 1:
    raise SystemExit("GPU smoke failed: exactly one allocated GPU must be visible")
if not hasattr(transformers, "Qwen3ForSequenceClassification"):
    raise SystemExit("GPU smoke failed: Transformers lacks the pinned Qwen3 oracle class")
if not Version("4.52.3") <= Version(transformers.__version__) < Version("5"):
    raise SystemExit("GPU smoke failed: Transformers version is outside [4.52.3, 5)")
x = torch.arange(16, device="cuda", dtype=torch.float32)
print("cuda_sum", x.sum().item())
PY

apptainer exec --cleanenv "${PRORM_IMAGE}" git --version | tee -a "${report}"

apptainer exec --cleanenv "${PRORM_IMAGE}" python -m pip check | tee -a "${report}"

apptainer exec --cleanenv "${PRORM_IMAGE}" python -m pip freeze \
  | sort | tee -a "${report}"
