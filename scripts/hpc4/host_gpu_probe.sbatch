#!/usr/bin/env bash
#SBATCH --job-name=prorm-host-gpu-probe
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

allowed_partition() {
  case "$1" in
    gpu-a30|gpu-l20|gpu-rtx5880|gpu-rtx4090d) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_compat_env PRORM_PROJECT_ROOT SRM_PROJECT_ROOT
: "${PRORM_PROJECT_ROOT:?set PRORM_PROJECT_ROOT to the absolute persistent project root}"
: "${SLURM_JOB_ID:?host GPU probe must run inside a Slurm allocation}"
: "${SLURM_JOB_ACCOUNT:?Slurm account evidence is required}"
: "${SLURM_JOB_PARTITION:?Slurm partition evidence is required}"
[[ "${SLURM_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SLURM_JOB_ID must be a positive integer" >&2
  exit 2
}

[[ "${PRORM_PROJECT_ROOT}" = /* ]] || {
  echo "PRORM_PROJECT_ROOT must be absolute" >&2
  exit 2
}
[[ "${SLURM_JOB_ACCOUNT}" == "sigroup" ]] || {
  echo "host GPU probe requires Slurm account sigroup" >&2
  exit 2
}
allowed_partition "${SLURM_JOB_PARTITION}" || {
  echo "unsupported HPC4 GPU partition: ${SLURM_JOB_PARTITION}" >&2
  exit 2
}
test -d "${PRORM_PROJECT_ROOT}"
test -w "${PRORM_PROJECT_ROOT}"
command -v nvidia-smi >/dev/null

project_root="$(realpath "${PRORM_PROJECT_ROOT}")"
PRORM_PROJECT_ROOT="${project_root}"
SRM_PROJECT_ROOT="${project_root}"
export PRORM_PROJECT_ROOT SRM_PROJECT_ROOT

report_dir="${project_root}/system-reports"
mkdir -p "${report_dir}"
test -w "${report_dir}"
report="${report_dir}/host-gpu-probe-${SLURM_JOB_ID}.txt"
[[ ! -e "${report}" ]] || {
  echo "refusing to overwrite existing host GPU report: ${report}" >&2
  exit 2
}
temporary_report="$(mktemp "${report_dir}/.host-gpu-probe-${SLURM_JOB_ID}.XXXXXX")"

finalize_report() {
  local original_status=$? final_status
  final_status="${original_status}"
  trap - EXIT
  if ! {
    echo "exit_code=${original_status}"
    if (( original_status == 0 )); then
      echo "status=passed"
    else
      echo "status=failed"
    fi
  } >> "${temporary_report}"; then
    final_status=1
  fi
  if [[ -e "${report}" ]] || ! mv "${temporary_report}" "${report}"; then
    echo "failed to install host GPU report: ${report}" >&2
    final_status=1
  fi
  exit "${final_status}"
}
trap finalize_report EXIT

gpu_query="$(
  nvidia-smi \
    --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader,nounits
)"
host_visible_gpu_count="$(
  printf '%s\n' "${gpu_query}" | awk 'NF { count += 1 } END { print count + 0 }'
)"

{
  echo "schema_version=prorm-host-gpu-probe/v1"
  echo "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "account=${SLURM_JOB_ACCOUNT}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "hostname=$(hostname)"
  echo "requested_gpus_per_node=1"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
  echo "slurm_gpus=${SLURM_GPUS:-unset}"
  echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset}"
  echo "host_visible_gpu_count=${host_visible_gpu_count}"
  echo "== host-visible GPU query: index, uuid, name, driver_version, memory.total MiB =="
  printf '%s\n' "${gpu_query}"
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
} | tee "${temporary_report}"

(( host_visible_gpu_count >= 1 )) || {
  echo "host GPU probe requires nvidia-smi to report at least one GPU" \
    | tee -a "${temporary_report}" >&2
  exit 2
}

{
  echo "== full nvidia-smi =="
  nvidia-smi
} | tee -a "${temporary_report}"

echo "host GPU probe passed: job_id=${SLURM_JOB_ID}"
