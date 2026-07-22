#!/usr/bin/env bash
#SBATCH --job-name=srm-gpu-smoke
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00

set -euo pipefail

scratch_base="/scratch/${USER}/smart-reward-model"
project_base="/project/sigroup/smart-reward-model"
job_dir="${scratch_base}/jobs/${SLURM_JOB_ID}"
mkdir -p "${job_dir}" "${project_base}/system-reports"

report="${job_dir}/gpu-smoke-${SLURM_JOB_ID}.txt"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "submit_dir=${SLURM_SUBMIT_DIR}"
  echo "git_commit=$(git -C "${SLURM_SUBMIT_DIR}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  hostname
  nvidia-smi
  python --version || true
  python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("available", torch.cuda.is_available())' || true
  env | grep '^SLURM_' | sort
} | tee "${report}"

cp -f "${report}" "${project_base}/system-reports/"
