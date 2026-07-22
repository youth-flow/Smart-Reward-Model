# HKUST HPC4 运行规范

本文只描述仓库当前 Slurm/Apptainer 接口。邮件确认 ITSO 账号已关联 Slurm account
`sigroup`，登录地址为 `hpc4.ust.hk`，并列出 CPU 分区 `amd/intel` 与 GPU 分区
`gpu-a30`、`gpu-l20`、`gpu-rtx5880`、`gpu-rtx4090d`。邮件没有证明当前 QoS、wall-time、
显存、驱动、节点健康或计算节点外网；这些都必须在首次登录和 GPU smoke job 中实测。

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

## 2. 存储分层

按邮件给出的目录与配额规划；首次登录仍必须用 `squota`、`df` 和实际权限验证：

| 路径 | 用途 | 约束 |
|---|---|---|
| `$HOME` | Git checkout、小配置 | 不放模型、dataset、artifact 或 checkpoint |
| `/project/sigroup/smart-reward-model` | validated image、pinned HF snapshots、可复用 artifact、正式结果 | 长期主副本；组共享，目录权限需预检 |
| `/scratch/$USER/smart-reward-model` | 每 job staging、临时 cache/checkpoint | 非永久存储；job 结束后以 project 副本为准 |

脚本为每个 job 建立：

```text
/scratch/$USER/smart-reward-model/jobs/$SLURM_JOB_ID/
```

正式持久化布局：

```text
/project/sigroup/smart-reward-model/
├── images/                         # validated .sif
├── hf-cache/                       # pinned offline snapshots
├── system-reports/                 # GPU smoke evidence
├── artifacts/<config-sha>/<image-sha>/<git-commit>/seed-N/
│                                    # producer-bound Phase-1 materialization
└── runs/<run-name>/<config-sha>/seed-N/job-J/
                                     # manifest/comparison/rollout evidence
```

不要并发提交相同 config/image/Git/seed 身份；project artifact 以该四元组为唯一身份。

## 3. 首次登录与 preflight

校外先连接 HKUST VPN，再登录并进入仓库根目录：

```bash
ssh YOUR_ITSO@hpc4.ust.hk
cd /absolute/path/to/Smart-Reward-Model
bash scripts/hpc4/preflight.sh
```

`preflight.sh` 使用 account `sigroup`，检查：

- `squota` 与 `squota -A sigroup`；
- scheduler/partition 可见性和当前 queue；
- `/project/sigroup`、`/scratch/$USER` 存在且可写；
- filesystem capacity；
- module 信息与 `apptainer --version`。

任一步非零即停止。不要为了“通过”而在脚本后追加 `|| true`；只有 `module avail` 的展示性
输出被脚本有意允许失败。

## 4. 固定镜像与离线输入

仓库不负责猜测适合 HPC4 驱动的 CUDA image。先在允许的构建环境创建 image，内含当前
project 依赖；把 image 和所需 Hugging Face revision 放进 project，再计算 digest：

```bash
export SRM_IMAGE=/project/sigroup/smart-reward-model/images/srm.sif
export SRM_IMAGE_SHA256="$(sha256sum "${SRM_IMAGE}" | awk '{print $1}')"
export SRM_HF_CACHE=/project/sigroup/smart-reward-model/hf-cache

test -f "${SRM_IMAGE}"
test -d "${SRM_HF_CACHE}"
printf '%s  %s\n' "${SRM_IMAGE_SHA256}" "${SRM_IMAGE}" \
  | sha256sum --check
```

`SRM_IMAGE` 必须是绝对路径，`SRM_IMAGE_SHA256` 必须是 64 位小写 hex；submit 和 compute
脚本都会再次校验。cache 至少要能离线解析 `configs/*.yaml` 中以下固定资产：

- `allenai/multipref` dataset revision；
- Qwen policy/reward-feature model 与 tokenizer revision；
- Skywork oracle model 与 tokenizer revision。

预缓存是独立 staging 操作，不得在正式 allocation 中使用 `--allow-download`。当前容器 job
设置：

```text
HF_HOME=$SRM_HF_CACHE
HF_HUB_CACHE=$SRM_HF_CACHE/hub
HF_DATASETS_CACHE=$SRM_HF_CACHE/datasets
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

正式 Python/CUDA package lock 只能在通过 GPU smoke 的 image 上记录；本地 Windows/CPU
environment 不能冒充 HPC4 lock。当前 LLM dependency 硬锁
`transformers>=4.52.3,<5`，且 image 必须提供 pinned Skywork oracle 所需的
`Qwen3ForSequenceClassification`。

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
/project/sigroup/smart-reward-model/system-reports/gpu-smoke-<job-id>.txt
```

只有报告确认容器/驱动兼容，才进入真实模型 smoke。

## 6. Controlled job 提交

接口严格要求三个位置参数：

```text
submit_controlled.sh <config.yaml> <gpu-partition> <walltime>
```

`walltime` 只能是 `HH:MM:SS` 或 `D-HH:MM:SS`。先用 smoke config，时间根据本集群实测
填写，不在仓库中虚构。formal config 必须是 repo 内已跟踪文件，controlled job 会拒绝
dirty worktree，并把 `HEAD` 作为 `SRM_GIT_COMMIT` producer identity：

```bash
cd /absolute/path/to/Smart-Reward-Model
bash scripts/hpc4/submit_controlled.sh \
  configs/smoke.yaml gpu-l20 "HH:MM:SS"
```

检查 Slurm log 与 project run output，确认下列步骤全部完成：

1. 严格解析 config、selected seed 和 config SHA256；
2. 用 `env-report --seed` 写 seed-specific `run-manifest.json`；
3. 离线加载 pinned MultiPref/Qwen/Skywork，创建或完整校验 Phase-1 artifact；
4. `controlled-compare --run-manifest` 先绑定 manifest SHA、formal environment 与 artifact
   producer，再在 config 声明的 damping（main 为三档）运行 fixed-step paired BT/SRM+；
5. 从主 damping head 构造两个 policy direction，分别匹配 measured KL；
6. 用 common-random test rollouts 和冻结 oracle transform 写 rollout evidence；
7. trap 将小型最终证据同步到 project。

smoke 保留 main 的 rank-4/最后四层 LoRA tangent、16-candidate KL probe 和 oracle batch 上限，
只缩 prompts 与 outer steps。它必须验证 candidate/token schema、LoRA-A hash/layout、PCG
convergence、KL tolerance 和输出 hash；`memory-materialize.json`、`memory-compare.json`、
`memory-rollout.json` 分别记录 PyTorch allocated/reserved 峰值。smoke 数字不能进入论文主表。

smoke 通过后，在**同一 partition/GPU 型号**提交 main 五 seed array：

```bash
export SRM_ARRAY_CONCURRENCY=1
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml gpu-l20 "D-HH:MM:SS"
```

submit 脚本在 validated image 内读取 config 的 seed 数，自动形成
`--array=0-(seed_count-1)%$SRM_ARRAY_CONCURRENCY`。`SRM_ARRAY_CONCURRENCY` 必须是正整数；
没有明确预算时保持 1。

## 7. 作业内的可复现与恢复语义

`controlled.sbatch` 不依赖 shell 当前的 Python，所有 Python 命令均在同一 clean Apptainer
环境中执行。它把 repo、job staging 和 HF cache bind 到容器，并设置 `PYTHONPATH` 为提交
commit 的 `src`。manifest 的 `selected_seed` 必须等于 array task 对应 seed；CUDA comparison
还要求 clean Git、`SRM_GIT_COMMIT==HEAD`、image SHA、Slurm account `sigroup`、partition、
CUDA 可见且恰好一个有名称的 GPU。comparison 保存 manifest bytes SHA 与解析后的 formal
environment identity；rollout 再将当前进程与该 identity 逐字段匹配并写回结果。

Artifact 路径是：

```text
/project/sigroup/smart-reward-model/artifacts/<config-sha>/<image-sha>/<git-commit>/seed-<seed>
```

如果该目录已有 `metadata.json`，job 会复制到本次 scratch 并由 loader 重新验证 config
hash、seed、tensor SHA256、schema、shape/dtype/finiteness、split 与 producer Git/image
digest；验证失败不会静默重建或继续。若不存在，则在 job staging 完整 materialize，成功后
才安装到 project。

每次提交使用新的 `job-$SLURM_JOB_ID` run 目录，不覆盖旧结果。退出 trap 只同步已经存在的
`run-manifest.json`、`comparison.json`、`rollout.json`、`updated_rollouts.jsonl` 和已有的
`memory-*.json`；部分/失败 job 仍保留 manifest 与 log 供诊断，但不得纳入 aggregate。大体积
artifact 不在每个 run 目录重复复制。

不要通过修改 project artifact 文件来“续跑”。materialization 是不可变数据层；reward
head comparison 固定步数且成本较小，失败后用同 config/seed 新 job 重跑。只有未来长时
scale-up 才需要另行设计原子 checkpoint/resume contract。

## 8. 监控与验收

常用只读命令：

```bash
squeue -u "$USER"
job_id=REPLACE_WITH_JOB_ID
sacct -j "${job_id}" --format=JobID,State,Elapsed,ExitCode,AllocTRES
tail -n 200 logs/srm-controlled-ARRAY_JOB_TASK.out
```

每个 seed 只有同时满足以下条件才算有效：

- Slurm state `COMPLETED` 且 exit code 0；
- manifest 的 config hash、`selected_seed`、Git commit、image hash、account、partition 和唯一 GPU
  model 正确；
- comparison 的 manifest SHA/environment identity 与同目录 `run-manifest.json` 完全一致；
- artifact identity/integrity 检查通过；
- main 和 sensitivity comparison 都存在，SRM PCG converged；
- BT/SRM measured KL 各自在 `0.01 ± 5%`，rollout 文件完整；
- 该 seed 没有读取 test 进行训练/选择，也没有手工改动产物。

OOM、offline snapshot missing、PCG/KL 不收敛、hash mismatch 或 dirty/错误 commit 都是硬失败。
修复原因后提交新 job；不得编辑 JSON 把失败改成成功。

## 9. 五 seed 聚合

Slurm array 不自动做跨 seed 聚合。五个 job 全部验收后，在同一 validated checkout/image 中
显式列出五个 comparison 文件：

```bash
smart-reward aggregate-results configs/main.yaml aggregate.json \
  PATH_TO_20260722_COMPARISON.json \
  PATH_TO_20260723_COMPARISON.json \
  PATH_TO_20260724_COMPARISON.json \
  PATH_TO_20260725_COMPARISON.json \
  PATH_TO_20260726_COMPARISON.json \
  --rollouts \
  PATH_TO_20260722_ROLLOUT.json \
  PATH_TO_20260723_ROLLOUT.json \
  PATH_TO_20260724_ROLLOUT.json \
  PATH_TO_20260725_ROLLOUT.json \
  PATH_TO_20260726_ROLLOUT.json
```

聚合器会拒绝缺 seed、重复 seed、额外 seed 或非 main damping identity，并验证每个 rollout
绑定的 config/artifact/comparison/JSONL hashes。它还重新读取每个 comparison 同目录的
`run-manifest.json`，校验 manifest SHA 与 selected seed，并硬要求五 seed 的 Git commit、
image SHA256、Slurm account/partition、GPU model 完全相同；共享 identity 写入 aggregate。随后在
同一 aggregate 中加入 prompt-level `test_rollout_improvement`，并自动聚合所有声明 damping
的 local-regret evidence、记录 PCG failure/non-reversal、执行预注册 criteria。正式主结论
只能读取
`pre_registered_evidence.status` 与逐项 criteria；不得查看 sensitivity 后换主阻尼。该状态
不是 p-value 或“显著”标签。

最终结果目录必须连同五个 manifest、comparison、rollout、updated JSONL、aggregate、
Slurm log 和 smoke report 一起长期保存。

## 10. 故障处理顺序

1. **preflight 失败**：先解决 account、partition、quota、权限或 Apptainer；不提交 GPU job。
2. **image/hash 失败**：停止；重新传输或确认 digest，绝不绕过校验。
3. **CUDA smoke 失败**：记录 report，换兼容 image/module；不进入模型 smoke。
   Transformers `<4.52.3`、`>=5` 或缺少 Qwen3 classification class 都属于 image 硬失败。
4. **offline snapshot missing**：在 staging 环境补齐 config 的精确 revision，再重提 job；
   不给正式 job 开网。
5. **OOM**：先从 smoke 记录峰值并验证是否为实现问题；任何改变 microbatch/config 的 run
   都产生新 config hash，不能混入原主实验。
6. **PCG/KL 不收敛**：视为 seed 失败并诊断数值几何；不得放宽 tolerance 后沿用原实验名。
7. **artifact/hash mismatch**：隔离损坏目录，保留证据；不要覆盖。确认目标后用新目录重建。
8. **manifest/environment mismatch**：不得手改 manifest 或 comparison；在相同 commit、image、
   account、partition、GPU model 下重跑受影响 seed。若环境必须改变，则全部五 seed 重跑。

任何需要改变预注册 config 的修复，都必须重新运行全部五个 paired seeds。
