#!/usr/bin/env bash
set -euo pipefail

account="sigroup"
project_root="/project/${account}"
scratch_root="/scratch/${USER}"

echo "== identity =="
id

echo "== quota =="
squota
squota -A "${account}"

echo "== scheduler =="
savail
squeue -u "${USER}"
scontrol show partition

echo "== storage =="
test -d "${project_root}"
test -w "${project_root}"
test -d "${scratch_root}"
test -w "${scratch_root}"
df -h "${project_root}" "${scratch_root}"

echo "== software =="
module avail 2>&1 | head -n 200 || true
command -v apptainer
apptainer --version

echo "Preflight passed. Submit a GPU smoke job next."
