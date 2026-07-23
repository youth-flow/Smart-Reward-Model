# HKUST HPC4 运行规范

本文只描述仓库当前 Slurm/Apptainer 接口。邮件确认 ITSO 账号已关联 Slurm account
`sigroup`，登录地址为 `hpc4.ust.hk`，并列出 CPU 分区 `amd/intel` 与 GPU 分区
`gpu-a30`、`gpu-l20`、`gpu-rtx5880`、`gpu-rtx4090d`。邮件没有证明当前 QoS、wall-time、
显存、驱动、节点健康或计算节点外网；这些都必须在首次登录和 GPU smoke job 中实测。

正式方法名是 ProRM+，正式对照是 repeated-label BT-MLE。仓库目录
`Smart-Reward-Model`、Python package `smart_reward` 和 project/scratch 路径作为兼容基础设施
保留。公开 CLI 为 `prorm`，公开 environment keys 统一为 `PRORM_*`；旧 `SRM_*` keys 只作为
迁移期兼容 alias 被脚本接受。

## 0. 从登录到结论的六个门

| Gate | 执行动作 | 通过标准 | 失败后 |
|---|---|---|---|
| 1. Account | 私密完成 password+Duo/2FA，再运行 `preflight.sh` | account、quota、partition、project/scratch、Apptainer 可用 | 不申请 GPU |
| 2. Driver + candidate | host probe、committed definition、SIF SHA256、两份 config inventory | driver 已观察；candidate image/cache 身份完整 | 不进入 CUDA smoke |
| 3. CUDA | `submit_gpu_smoke.sh` | 单 GPU、版本、Qwen3 class、`pip check` 全通过 | 修 image/partition |
| 4. Model smoke | `submit_controlled.sh configs/smoke.yaml ...` | artifact、BT-MLE/ProRM+、KL、rollout、memory evidence 完整 | 定位实现或容量问题 |
| 5. Main | 同一环境提交 `configs/main.yaml` | 五个 paired seeds 主链完整 | 保留失败证据后诊断 |
| 6. Aggregate | `aggregate-results` | 身份一致并写出预注册判据 | 不手改结果 |

这些 gate 必须顺序执行。后一个 gate 成功不能补偿前一个 gate 的失败；尤其不能在登录节点
直接运行模型命令，也不能因为单个 smoke 成功就宣称实验结论成立。

Gate 2 产生的只是 **candidate image**。只有 Gate 3 在目标 HPC4 partition 上以 exit code 0
完成后，该 image SHA 才能称为 **HPC4-validated**。host probe 报告不是 CUDA 容器验证的替代。

## 1. 不可违反的原则

- 登录节点只做 Git、quota/partition 检查、文件传输和 `sbatch`；禁止加载模型、提取 score
  或训练。
- 正式实验只使用 SHA256 校验过的 Apptainer `.sif`，不在计算 job 中 `pip install`。
- model、tokenizer 和 dataset 必须提前缓存 config 指定的 commit revision；正式 job 强制
  Hugging Face offline。
- 同一个 paired comparison 固定同一 GPU partition；smoke 后记录实际 GPU 型号，不混用
  不同型号的 seed。
- 第一阶段是一节点、一 GPU。未获得明确 GPU-hour 预算前，array concurrency 默认 1，
  不提交无界 sweep 或多节点作业。
- credential 只通过受控 secret 注入；不得写入仓库、YAML、manifest、artifact 或 Slurm log。
- ITSO password、Duo/2FA response、SSH private key 和 recovery code 只由用户在受信任的 SSH
  客户端中处理；绝不发送给 Codex，也不要求 Codex 代填。
- formal array 排队和执行期间不得切换、pull、修改或复用其 checkout 做开发。submit 绑定
  当时的 clean `HEAD`；compute 从该 commit 建立私有 detached scratch clone 并在每个 phase
  边界复验，登录 checkout 后续变化不会进入 job。
- 提交与 compute 都拒绝任何导出的 `APPTAINER*`/`SINGULARITY*` ambient control variable；
  所有 bind 与容器内环境只能来自受审脚本的显式 allowlist。

## 2. 存储分层

按邮件给出的目录与配额规划；首次登录仍必须用 `squota`、`df` 和实际权限验证：

| 逻辑路径 | 用途 | 约束 |
|---|---|---|
| `$HOME` | Git checkout、小配置 | 不放模型、dataset、artifact 或 checkpoint |
| `$PRORM_PROJECT_ROOT` | validated image、pinned HF snapshots、可复用 artifact、正式结果 | 必须是绝对组共享根；建议 `/project/sigroup/smart-reward-model` |
| `$PRORM_SCRATCH_ROOT` | 每 job staging、临时 cache/checkpoint | 必须是绝对个人 scratch 根；建议 `/scratch/$USER/smart-reward-model` |

脚本为每个 job 建立：

```text
$PRORM_SCRATCH_ROOT/jobs/$SLURM_JOB_ID/
```

正式持久化布局：

```text
$PRORM_PROJECT_ROOT/
├── images/                         # validated .sif
├── hf-cache/
│   ├── hub/                        # Qwen/Skywork/MultiPref raw snapshots
│   ├── datasets/                   # processed Datasets/Arrow cache
│   └── inventories/                # config-specific content inventories
├── system-reports/                 # build/staging/host/GPU evidence
├── slurm-logs/                     # scheduler stdout/stderr
├── artifacts/<config-sha>/<image-sha>/<inventory-sha>/<git-commit>/seed-N/
│                                    # candidates, repeated labels, S/features/oracle
└── runs/<run-name>/<config-sha>/
    ├── seed-N/job-J/               # manifest/comparison/rollout + artifact symlink
    └── aggregate/aggregate.json    # accepted five-seed aggregate
```

Qwen 和 Skywork base weights 只保存在 `hf-cache/hub`。训练得到的 bias-free linear RM head
直接序列化在 `comparison.json`；当前 controlled experiment 不保存另一份完整 Qwen checkpoint，
也不导出 production LoRA adapter。policy LoRA-B update 在 rollout 时从 head 与固定 geometry
重构。

不要并发提交相同 config/image/Git/inventory/seed 身份；project artifact 以该身份为唯一主副本。
每个成功 `project_run` 必须有相对 symlink `artifact` 指向该 content-addressed artifact；这样
scratch 删除后，`comparison.json` 和 `rollout.json` 中的相对 POSIX path 仍可解析。

## 3. 首次登录与 preflight

校外先连接 HKUST VPN。首次 SSH 会要求 ITSO password 与 Duo/2FA；这些值只在 SSH 客户端
中输入，不复制到对话、命令历史、文件或环境变量。验证服务器 host key 后完成登录，再取得
公开 Git checkout。仓库内命令路径都相对 Git 根目录；只有 project/scratch 两个跨节点存储
锚点是绝对路径：

```bash
ssh YOUR_ITSO@hpc4.ust.hk
git clone https://github.com/youth-flow/Smart-Reward-Model.git
cd Smart-Reward-Model
test "$(git remote get-url origin)" = \
  "https://github.com/youth-flow/Smart-Reward-Model.git"
git rev-parse --verify HEAD

# .env.hpc4 is ignored by Git. Never edit the tracked example for a formal run.
test -e .env.hpc4 || cp scripts/hpc4/env.example .env.hpc4
source .env.hpc4
mkdir -p \
  "${PRORM_PROJECT_ROOT}"/{images,hf-cache,system-reports,slurm-logs,artifacts,runs} \
  "${PRORM_SCRATCH_ROOT}/jobs"
bash scripts/hpc4/preflight.sh
```

已经 clone 的用户跳过 `git clone`，但仍必须执行 remote、commit 与 clean-worktree 检查。公开
asset staging 不需要 Hugging Face token。SSH key 可在首次 password+Duo 登录后按 ITSO 官方
流程配置；private key 始终留在用户设备，不能提交到 Git。

`preflight.sh` 使用 account `sigroup`，检查：

- `squota` 与 `squota -A sigroup`；
- scheduler/partition 可见性和当前 queue；
- `PRORM_PROJECT_ROOT`、`PRORM_SCRATCH_ROOT` 存在且可写；
- filesystem capacity；
- module 信息与 `apptainer --version`。

任一步非零即停止。不要为了“通过”而在脚本后追加 `|| true`；只有 `module avail` 的展示性
输出被脚本有意允许失败。

## 4. 固定镜像与离线输入

仓库不负责猜测适合 HPC4 驱动的 CUDA image。正式模型 smoke 前必须先得到 host driver
report、hash-frozen **candidate** Apptainer image、完整 environment lock 与
snapshot-staging record。host probe 不依赖 image，因此不存在“必须先有 image 才能查询
driver”的循环；candidate 是否真正兼容只能由下一节的 GPU smoke 判定。

首次环境固化必须完成以下闭环：

1. 提交 `submit_host_gpu_probe.sh`，在最小 GPU allocation 中记录 host driver、
   `nvidia-smi` 和实际 GPU 型号；
2. 选择与该 driver 兼容的 CUDA/PyTorch base，并把完整 `.def` 纳入 Git；
3. 在 definition 中安装本 commit 的 `.[llm]` 依赖，构建 `.sif`，运行 `pip check`；
4. 保存 definition Git commit、build log 与 image SHA256；排序后的 `pip freeze` 由 GPU
   smoke 在真实 host driver 上记录；
5. 分别按 `configs/smoke.yaml` 与 `configs/main.yaml` 的固定 revision 预缓存全部 HF assets，
   保存 config-specific `repo_id/revision/file SHA256` inventory；
6. 在 `TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 下验证 snapshot、config、tokenizer 与
   dataset resolution；这一步不虚称已实例化模型权重；
7. GPU smoke 以 exit code 0 验证 candidate image 后，才把该 SHA 称为 validated image。

先提交不依赖容器的 host probe：

```bash
bash scripts/hpc4/submit_host_gpu_probe.sh gpu-l20
squeue -u "$USER"
```

等待 job 完成，先用 `sacct` 确认 exit code 0，再读取
`$PRORM_PROJECT_ROOT/system-reports/host-gpu-probe-<job-id>.txt`。只有报告中的实际
driver/GPU 可以决定 CUDA/PyTorch base。完整 definition 必须在 Git 中；SIF、build log 与
SHA256 必须持久化。当前若尚无与该报告匹配并经审查的 definition，流程就在此停止，不能把
任意 Docker/CUDA tag 填成“validated”。SIF 应在 ITSO 允许的 Apptainer build endpoint
构建后复制到 `$PRORM_PROJECT_ROOT/images/prorm.sif`。

image/cache 的公开配置值使用相对于 project root 的路径；submit 脚本会 canonicalize 为
Apptainer 所需的绝对路径，并拒绝 path escape：

```bash
export PRORM_IMAGE=images/prorm.sif
export PRORM_HF_CACHE=hf-cache
export PRORM_IMAGE_SHA256="$(
  sha256sum "${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}" | awk '{print $1}'
)"

test -f "${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}"
mkdir -p "${PRORM_PROJECT_ROOT}/${PRORM_HF_CACHE}"
printf '%s  %s\n' \
  "${PRORM_IMAGE_SHA256}" "${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}" \
  | sha256sum --check
```

`PRORM_PROJECT_ROOT` 与 `PRORM_SCRATCH_ROOT` 必须是绝对路径；`PRORM_IMAGE` 和
`PRORM_HF_CACHE` 可以是相对 project root 的资产路径。`PRORM_IMAGE_SHA256` 必须是 64 位
小写 hex；submit 和 compute 脚本都会再次校验。cache 至少要能离线解析 `configs/*.yaml`
中的以下固定资产：

- `allenai/multipref` dataset revision；
- Qwen policy/reward-feature model 与 tokenizer revision；
- Skywork oracle model 与 tokenizer revision。

首次 staging 必须对 smoke 与 main 两份 config 分别执行并保存 stdout/stderr。两者当前引用
相同 repo revision，但各自的 config hash 与 inventory identity 不可互相替代：

```bash
set -euo pipefail
repo_root="$(pwd -P)"
image_path="${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}"
cache_root="${PRORM_PROJECT_ROOT}/${PRORM_HF_CACHE}"
set -o pipefail
for config in configs/smoke.yaml configs/main.yaml; do
  stem="$(basename "${config}" .yaml)"
  apptainer exec --cleanenv \
    --bind "${repo_root}:${repo_root},${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}" \
    --env "PYTHONPATH=${repo_root}/src" \
    "${image_path}" \
    python scripts/hpc4/stage_hf_assets.py "${config}" "${cache_root}" \
    2>&1 | tee "${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${stem}.log"
done
sha256sum "${cache_root}"/inventories/*.json
```

该工具的 JSON inventory 只含 cache-relative POSIX path、每个 snapshot 文件 SHA256、
package versions 和 offline-resolution evidence。它会 local-only resolve model repository、
`AutoConfig`、tokenizer 与 MultiPref dataset；不会加载 Qwen/Skywork weight tensors。真正的
weight/class/显存验证属于 controlled model smoke。

再次审计现有 cache 使用 `--verify-only`；该模式禁止网络、重算内容并要求 bytes 与现有
inventory 一致，不覆盖正式 inventory：

```bash
for config in configs/smoke.yaml configs/main.yaml; do
  stem="$(basename "${config}" .yaml)"
  apptainer exec --cleanenv \
    --bind "${repo_root}:${repo_root},${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}" \
    --env "PYTHONPATH=${repo_root}/src" \
    "${image_path}" \
    python scripts/hpc4/stage_hf_assets.py \
    "${config}" "${cache_root}" --verify-only \
    2>&1 | tee "${PRORM_PROJECT_ROOT}/system-reports/hf-verify-${stem}.log"
done
```

`submit_controlled.sh` 根据 config hash 固定选择
`$PRORM_HF_CACHE/inventories/<config-hash>.json`，校验其 SHA256，并把绝对 inventory path 与
`PRORM_HF_INVENTORY_SHA256` 送入 job。compute 在 materialization 前再次 `--verify-only`；
manifest、artifact producer、comparison/rollout environment identity 和 aggregate 都绑定
同一 `hf_inventory_sha256`。因此 cache 内容改变不能沿用旧正式身份。

预缓存是独立 staging 操作，不得在正式 allocation 中使用 `--allow-download`。当前容器 job
设置：

```text
HF_HOME=$PRORM_HF_CACHE
HF_HUB_CACHE=$PRORM_HF_CACHE/hub
HF_DATASETS_CACHE=$PRORM_HF_CACHE/datasets
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

正式 Python/CUDA package lock 只能在通过 GPU smoke 的 image 上记录；本地 Windows/CPU
environment 不能冒充 HPC4 lock。当前 LLM dependency 硬锁
`transformers>=4.52.3,<5`，且 image 必须提供 pinned Skywork oracle 所需的
`Qwen3ForSequenceClassification`。

Gate 4 前至少要能定位以下证据：

| 证据 | 持久化位置 |
|---|---|
| Apptainer definition + Git commit | repository |
| Candidate/validated SIF + SHA256 | `$PRORM_PROJECT_ROOT/images/` |
| Definition commit + build log | repository + `$PRORM_PROJECT_ROOT/system-reports/` |
| `pip check` + sorted `pip freeze` | `system-reports/` |
| HF repo/revision/cache inventory | `$PRORM_PROJECT_ROOT/hf-cache/inventories/` |
| Networked stage + offline verify logs | `$PRORM_PROJECT_ROOT/system-reports/` |
| Host driver、partition、GPU model | host-probe and GPU-smoke reports |

## 5. GPU smoke：先验证环境，不跑模型

提交十分钟单 GPU smoke：

```bash
bash scripts/hpc4/submit_gpu_smoke.sh gpu-l20
squeue -u "$USER"
```

参数必须是脚本 allowlist 中的一个分区：
`gpu-a30|gpu-l20|gpu-rtx5880|gpu-rtx4090d`。如果 `gpu-l20` 的 account/QoS 不允许，
根据 preflight 输出选择邮件列出的其他分区；不要盲目反复提交。

`gpu_smoke.sbatch` 在 `apptainer exec --cleanenv --nv` 内验证：

- image SHA256；
- `nvidia-smi`；
- 容器内 `git --version`；
- PyTorch、CUDA、Transformers version；
- Transformers 位于 `[4.52.3,5)` 且暴露 `Qwen3ForSequenceClassification`；
- Accelerate、Datasets、PEFT、Safetensors 可导入并记录版本；
- scheduler visibility 变量被显式传入容器，且 `torch.cuda.is_available()` 为真、GPU count
  **恰好为 1**；
- 一个实际 CUDA tensor operation；
- `python -m pip check` 无依赖冲突；
- 排序后的完整 `python -m pip freeze`。

任一检查失败时 job 非零。无论成功失败，trap 都把报告同步到：

```text
$PRORM_PROJECT_ROOT/system-reports/gpu-smoke-<job-id>.txt
```

只有 Slurm state 为 `COMPLETED`、exit code 0 且报告中的全部检查完成，该 image SHA 才成为
validated identity，才能进入真实模型 smoke。GPU smoke 只验证 runtime/class，不加载模型
权重；下一 gate 才首次实例化 Qwen/Skywork。

## 6. Controlled job 提交

接口严格要求三个位置参数：

```text
submit_controlled.sh <config.yaml> <gpu-partition> <walltime>
```

`walltime` 只能是 `HH:MM:SS` 或 `D-HH:MM:SS`。先用 smoke config，时间根据本集群实测
填写，不在仓库中虚构。formal config 必须是 repo 内已跟踪文件。提交前确认使用已审查并推送
的 commit：

```bash
git fetch origin main
test -z "$(git status --porcelain --untracked-files=normal)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"

export PRORM_SMOKE_WALLTIME=REPLACE_WITH_APPROVED_PILOT_WALLTIME
bash scripts/hpc4/submit_controlled.sh \
  configs/smoke.yaml gpu-l20 "${PRORM_SMOKE_WALLTIME}"
```

submit 脚本在申请 allocation 前 fail-fast：验证 clean worktree、tracked config、exact `HEAD`、
image SHA、config-specific inventory path/SHA，并导出 `PRORM_GIT_COMMIT`、
`PRORM_HF_INVENTORY` 与 `PRORM_HF_INVENTORY_SHA256`。compute 再次要求当前 `HEAD` 等于提交
commit；排队期间切换 checkout 不会静默改变实验，而会使 job 硬失败。

检查 Slurm log 与 project run output，确认下列步骤全部完成：

1. 严格解析 config、selected seed、config SHA256 与 submission Git identity；
2. 对 config-specific cache 做无网络 `--verify-only`，要求 inventory bytes/SHA 不变，并写
   `hf-inventory-verification.json` 作为本次 job evidence；
3. 用 `env-report --seed` 写包含 inventory digest 的 seed-specific `run-manifest.json`；
4. 离线加载 pinned MultiPref/Qwen/Skywork，创建或完整校验 Phase-1 artifact，并用
   `artifact-materialization.json` 明确记录本 job 是 `materialized` 还是 `reused`；
5. `controlled-compare --run-manifest` 先绑定 manifest SHA、formal environment 与 artifact
   producer，再在 config 声明的 damping（main 为三档）运行 fixed-step paired
   BT-MLE/ProRM+；
6. 从主 damping head 构造两个 policy direction，分别匹配 measured KL；
7. 用 common-random test rollouts 和冻结 oracle transform 写 rollout evidence；
8. trap 将小型最终证据同步到 project，创建指向 authoritative artifact 的相对 symlink，并
   原子写入互斥的 `SUCCESS` 或 `FAILED` completion marker。

smoke 保留 main 的 rank-4/最后四层 LoRA tangent、16-candidate KL probe 和 oracle batch 上限，
只缩 prompts 与 outer steps。它必须验证 candidate/token schema、LoRA-A hash/layout、PCG
convergence、KL tolerance 和输出 hash；`memory-materialize.json`、`memory-compare.json`、
`memory-rollout.json` 分别记录 PyTorch allocated/reserved 峰值。smoke 数字不能进入论文主表。

smoke 完成后先填写资源放大表，再申请 main。仓库不预填未经实测的数值：

| 资源 | Smoke 实测 | Main 估算依据 | Main 申请值 |
|---|---:|---|---:|
| Wall-clock / seed | `TBD` | 分阶段 log；不得只按 prompt 数线性外推 | `TBD` |
| Peak GPU allocated/reserved | `TBD` | 三份 `memory-*.json` 的最大值 + safety margin | `TBD` |
| Host MaxRSS / allocated CPUs | `TBD` | `sacct` 的 `MaxRSS,AllocCPUS,ReqMem`；当前 sbatch site defaults 必须实测 | `TBD` |
| Scratch peak | `TBD` | job staging + artifact 临时双副本 | `TBD` |
| Project persistent bytes / seed | `TBD` | artifact + run evidence | `TBD` |
| GPU-hours / 5 seeds | `TBD` | 五个单 GPU job 的 elapsed 之和；近似单 seed wall-clock × 5 | `TBD` |

只有表中数值来自成功 smoke 且符合 quota/account 预算，才能确定 main walltime 和并发度。
array concurrency 只改变 campaign makespan，不减少总 GPU-hours。如果 site-default CPU/RAM
不足，必须先把资源请求作为脚本变更提交并重跑 smoke；不得在 main 中临时使用未经记录的
资源设置。

smoke 通过后，在**同一 partition/GPU 型号**提交 main 五 seed array：

```bash
export PRORM_ARRAY_CONCURRENCY=1
export PRORM_MAIN_WALLTIME=REPLACE_WITH_APPROVED_WALLTIME
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml gpu-l20 "${PRORM_MAIN_WALLTIME}"
```

submit 脚本在 validated image 内读取 config 的 seed 数，自动形成
`--array=0-(seed_count-1)%$PRORM_ARRAY_CONCURRENCY`。`PRORM_ARRAY_CONCURRENCY` 必须是正整数；
没有明确预算时保持 1。

## 7. 作业内的可复现与恢复语义

`controlled.sbatch` 不依赖 shell 当前的 Python。它在 job scratch 中建立 submitted commit 的
私有 detached clone，所有 Python 命令均从该不可被登录 checkout 切换影响的源码进入同一
clean Apptainer 环境；job staging 与 HF cache 被显式 bind，`PYTHONPATH` 指向 detached
commit 的 `src`。manifest 的 `selected_seed` 必须等于 array task 对应 seed；CUDA comparison
还要求 clean Git、submission `PRORM_GIT_COMMIT==HEAD`、image SHA、HF inventory SHA、
Slurm account `sigroup`、partition、CUDA 可见且恰好一个有名称的 GPU。comparison 保存
manifest bytes SHA 与解析后的 formal environment identity；rollout 再将当前进程与该 identity
逐字段匹配并写回结果。

Artifact 路径是：

```text
$PRORM_PROJECT_ROOT/artifacts/<config-sha>/<image-sha>/<inventory-sha>/<git-commit>/seed-<seed>
```

如果该目录已有 `metadata.json`，job 会复制到本次 scratch 并由 loader 重新验证 config
hash、seed、tensor SHA256、schema、shape/dtype/finiteness、split 与 producer
Git/image/inventory digest；验证失败不会静默重建或继续。若不存在，则在 job staging 完整
materialize，并通过 project 内同 filesystem staging、锁与原子 rename 安装，避免并发发布
半成品。

每次提交使用新的 `job-$SLURM_JOB_ID` run 目录，不覆盖旧结果。退出 trap 只同步已经存在的
`run-manifest.json`、`hf-inventory-verification.json`、`artifact-materialization.json`、
`comparison.json`、`rollout.json`、`updated_rollouts.jsonl` 和本 job 实际产生的
`memory-*.json`；复用 artifact 的 job 不伪造 `memory-materialize.json`。`SUCCESS`/`FAILED`
marker 记录
workload/final exit code。部分/失败 job 仍
保留 manifest 与 log 供诊断，但不得纳入 aggregate。大体积 artifact 不在每个 run 目录重复
复制。成功 run 的
`$project_run/artifact` 是指向上面 content-addressed project artifact 的**相对 symlink**；
验收必须执行 `test -e "$project_run/artifact/metadata.json"`。

不要通过修改 project artifact 文件来“续跑”。materialization 是不可变数据层；reward
head comparison 固定步数且成本较小，失败后用同 config/seed 新 job 重跑。只有未来长时
scale-up 才需要另行设计原子 checkpoint/resume contract。

## 8. 监控与验收

常用只读命令：

```bash
squeue -u "$USER"
job_id=REPLACE_WITH_JOB_ID
sacct -j "${job_id}" \
  --format=JobID,State,Elapsed,ExitCode,AllocTRES,AllocCPUS,ReqMem,MaxRSS
tail -n 200 \
  "${PRORM_PROJECT_ROOT}/slurm-logs/prorm-controlled-ARRAY_JOB_TASK.out"
```

每个 seed 只有同时满足以下条件才算有效：

- Slurm state `COMPLETED`、`ExitCode=0:0`、run 有 `SUCCESS` 且没有 `FAILED` marker；
- manifest 的 config hash、`selected_seed`、Git commit、image hash、HF inventory hash、
  account、partition 和唯一 GPU model 正确；
- `hf-inventory-verification.json.inventory_sha256` 等于 manifest 的 inventory hash；
- comparison 的 manifest SHA/environment identity 与同目录 `run-manifest.json` 完全一致；
- artifact identity/integrity 检查通过；
- run-local `artifact` relative symlink 可解析到 content-addressed project artifact；
- 主阻尼 comparison 存在且 PCG converged；每档 sensitivity 要么有完整结果，要么有显式
  failure record，任何 sensitivity seed 都不能被静默删除；
- BT-MLE/ProRM+ measured KL 各自在 `0.01 ± 5%`，rollout 文件完整；
- 该 seed 没有读取 test 进行训练/选择，也没有手工改动产物。

OOM、offline snapshot missing、主阻尼训练 PCG、rollout direction PCG、measured-KL、hash
mismatch 或 dirty/错误 commit 失败，都会使该 seed 的主链无效。sensitivity PCG failure 不
删除整个 seed：作业保留 failure record，继续产出可诊断 aggregate，但最终预注册状态必须是
`not_passed`。任何情况下都不得编辑 JSON 把失败改成成功。

## 9. 五 seed 聚合

Slurm array 不自动做跨 seed 聚合。先逐一用上一节的 `sacct`、log、manifest 和 symlink 条件
验收。每个 seed 必须**显式选择一个** exit-code-0 的 `job-J`；不要用 `latest`、`find | head`
或通配符偷偷选择重复/失败 run。下面五个 job ID 必须由验收记录填写：

```bash
set -euo pipefail
repo_root="$(pwd -P)"
image_path="${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image_path}" \
  | sha256sum --check

main_hash="$(
  apptainer exec --cleanenv \
    --bind "${repo_root}:${repo_root}" \
    --env "PYTHONPATH=${repo_root}/src" \
    "${image_path}" \
    python - "${repo_root}/configs/main.yaml" <<'PY'
import sys
from smart_reward.config import config_hash, load_config
print(config_hash(load_config(sys.argv[1])))
PY
)"
campaign_root="${PRORM_PROJECT_ROOT}/runs/controlled-main/${main_hash}"
aggregate_dir="${campaign_root}/aggregate"
mkdir -p "${aggregate_dir}"
test ! -e "${aggregate_dir}/aggregate.json"
test ! -e "${aggregate_dir}/aggregate.json.sha256"

comparisons=(
  "${campaign_root}/seed-20260722/job-REPLACE_20260722/comparison.json"
  "${campaign_root}/seed-20260723/job-REPLACE_20260723/comparison.json"
  "${campaign_root}/seed-20260724/job-REPLACE_20260724/comparison.json"
  "${campaign_root}/seed-20260725/job-REPLACE_20260725/comparison.json"
  "${campaign_root}/seed-20260726/job-REPLACE_20260726/comparison.json"
)
rollouts=(
  "${campaign_root}/seed-20260722/job-REPLACE_20260722/rollout.json"
  "${campaign_root}/seed-20260723/job-REPLACE_20260723/rollout.json"
  "${campaign_root}/seed-20260724/job-REPLACE_20260724/rollout.json"
  "${campaign_root}/seed-20260725/job-REPLACE_20260725/rollout.json"
  "${campaign_root}/seed-20260726/job-REPLACE_20260726/rollout.json"
)
for path in "${comparisons[@]}" "${rollouts[@]}"; do
  test -f "${path}"
done
for comparison in "${comparisons[@]}"; do
  run_dir="$(dirname "${comparison}")"
  test -f "${run_dir}/SUCCESS"
  test ! -e "${run_dir}/FAILED"
  test -e "${run_dir}/artifact/metadata.json"
done

# Use the validated image and submitted source, never an unpinned host prorm install.
apptainer exec --cleanenv \
  --bind "${repo_root}:${repo_root},${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}" \
  --env "PYTHONPATH=${repo_root}/src" \
  "${image_path}" \
  python -m smart_reward.cli aggregate-results \
  "${repo_root}/configs/main.yaml" "${aggregate_dir}/aggregate.json" \
  "${comparisons[@]}" \
  --repo-root "${repo_root}" \
  --rollouts "${rollouts[@]}"
aggregate_sha_tmp="$(
  mktemp "${aggregate_dir}/.aggregate.json.sha256.XXXXXX"
)"
trap 'rm -f -- "${aggregate_sha_tmp}"' EXIT
(
  cd "${aggregate_dir}"
  sha256sum aggregate.json
) > "${aggregate_sha_tmp}"
ln "${aggregate_sha_tmp}" "${aggregate_dir}/aggregate.json.sha256"
rm -f -- "${aggregate_sha_tmp}"
aggregate_sha_tmp=""
trap - EXIT
```

聚合器会拒绝缺 seed、重复 seed、额外 seed 或非 main damping identity，并验证每个 rollout
绑定的 config/artifact/comparison/JSONL hashes。它还重新读取每个 comparison 同目录的
`run-manifest.json`，校验 manifest SHA 与 selected seed，并硬要求五 seed 的 Git commit、
image SHA256、HF inventory SHA256、Slurm account/partition、GPU model 完全相同；同时要求
aggregation checkout clean 且 `HEAD` 精确等于该 producer commit，并把相对 config path 写入
`aggregation_source`。共享 identity 写入 aggregate。随后在同一 aggregate 中加入 prompt-level
`test_rollout_improvement`，并自动
聚合所有声明 damping 的 local-regret evidence、记录 PCG failure/non-reversal、执行预注册
criteria。正式主结论只能读取
`pre_registered_evidence.status` 与逐项 criteria；不得查看 sensitivity 后换主阻尼。该状态
不是 p-value 或“显著”标签。aggregate 与对应 SHA 文件均采用 no-overwrite 发布；若目标
已存在，必须把它作为既有证据审计，不能静默重算或覆盖。

最终结果目录必须连同五个 manifest、comparison、rollout、updated JSONL、aggregate、
artifact symlink、Slurm log、image/build/staging/smoke reports 与两份 HF inventory 一起长期
保存。

### 9.1 Scratch 清理

scratch 在验收前是故障恢复证据，不自动删除。只有 aggregate 已原子写出并校验、五个 project
run 文件齐全、每个 `artifact/metadata.json` symlink 可解析后，才按**确切 job ID**清理成功
job；失败 job 先保留诊断。先打印并人工复核目标，再执行删除：

```bash
completed_job_ids=(
  REPLACE_20260722 REPLACE_20260723 REPLACE_20260724
  REPLACE_20260725 REPLACE_20260726
)
targets=()
for job_id in "${completed_job_ids[@]}"; do
  target="$(realpath -e -- "${PRORM_SCRATCH_ROOT}/jobs/${job_id}")"
  case "${target}" in
    "${PRORM_SCRATCH_ROOT}"/jobs/*) ;;
    *) echo "refusing scratch escape: ${target}" >&2; exit 2 ;;
  esac
  printf 'review cleanup target: %s\n' "${target}"
  targets+=("${target}")
done

# Run only after reviewing every printed absolute path.
rm -r -- "${targets[@]}"
```

该命令不使用 glob、不删除 `$PRORM_SCRATCH_ROOT/jobs` 本身，也不触碰 project 主副本。删除后
用 `test -f`/`test -e` 再确认 aggregate、run JSON 和 artifact symlink 仍可访问。

## 10. 故障处理顺序

1. **preflight 失败**：先解决 account、partition、quota、权限或 Apptainer；不提交 GPU job。
2. **image/hash 失败**：停止；重新传输或确认 digest，绝不绕过校验。
3. **CUDA smoke 失败**：记录 report，换兼容 image/module；不进入模型 smoke。
   Transformers `<4.52.3`、`>=5` 或缺少 Qwen3 classification class 都属于 image 硬失败。
4. **offline snapshot missing**：在 staging 环境补齐 config 的精确 revision，再重提 job；
   不给正式 job 开网。
5. **OOM**：先从 smoke 记录峰值并验证是否为实现问题；任何改变 microbatch/config 的 run
   都产生新 config hash，不能混入原主实验。
6. **主阻尼 PCG、rollout direction PCG 或 KL 不收敛**：该 seed 主链硬失败；诊断数值几何，
   不得放宽 tolerance 后沿用原实验名。
7. **Sensitivity PCG 不收敛**：保留 comparison 中的 failure record，继续完成可执行的主链和
   aggregate；结果必须为 `not_passed`，不得丢弃该 damping/seed 后重算均值。
8. **artifact/hash mismatch**：隔离损坏目录，保留证据；不要覆盖。确认目标后用新目录重建。
9. **manifest/environment mismatch**：不得手改 manifest 或 comparison；在相同 commit、image、
   account、partition、GPU model 下重跑受影响 seed。若环境必须改变，则全部五 seed 重跑。

任何需要改变预注册 config 的修复，都必须重新运行全部五个 paired seeds。
