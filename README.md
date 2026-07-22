# Smart Reward Model（SRM+）

[![CI](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml)

SRM+ 是一个由**局部 KL-regularized policy regret**直接推导出的 policy-aware reward
model 目标。它不把 Bradley–Terry（BT）拟合得更准当作最终目的，而是只惩罚会传递到
下一次 policy update 的 reward error，并用 reference policy 的 Fisher 几何衡量该误差。

当前仓库已经完成理论规格、CPU 数值内核、严格配置、真实 Hugging Face Phase 1
materialization、BT/SRM+ 公平训练、完整性校验 artifact、matched-KL policy rollout API、
Slurm/Apptainer 运行脚本和测试。**当前机器尚未下载并执行固定 revision 的真实模型，也
尚未在 HKUST HPC4 跑正式实验；仓库中没有论文结果，任何 “SRM+ 胜出” 都仍是待检验
假设。** CPU synthetic benchmark 只验证代码链路，不是效果证据。

## 1. 核心结论

固定 prompt 分布 \(\rho\)、reference policy \(\pi_0=\pi_{\theta_0}\) 和真正允许更新的
policy tangent 坐标 \(\theta\)，定义

\[
s_0(x,y)=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0},\qquad
F_0=\mathbb E[s_0s_0^\top],\qquad
g_r=\mathbb E[s_0r(x,y)].
\]

在局部二阶 KL 展开下，reward \(r\) 诱导的最优更新为

\[
\delta_r=\beta^{-1}F_0^\dagger g_r.
\]

因此，用 \(r_\phi\) 代替真实 reward \(r^*\) 的 population local regret **精确**等于

\[
\widetilde{\mathrm{Reg}}(r_\phi)
=\frac{1}{2\beta}(g_{r_\phi}-g_*)^\top
F_0^\dagger(g_{r_\phi}-g_*).
\]

这个目标只看 policy tangent 可见的 reward error。逐点 reward MSE 或 BT likelihood
会同等对待许多不会改变当前 policy update 的误差，而 SRM+ 直接优化下游决策量。

### 从 pairwise label 识别 policy moment

对同一 prompt 独立采样 \(y,y'\sim\pi_0\)，令

\[
z=s_0(x,y)-s_0(x,y'),\qquad
t_\phi=r_\phi(x,y)-r_\phi(x,y').
\]

由 score identity，常数因子固定为

\[
g_r=\frac12\mathbb E[z\,\Delta r],\qquad
F_0=\frac12\mathbb E[zz^\top].
\]

在 BTL 假设 \(a\mid e\sim\mathrm{Bernoulli}(p^*)\)、
\(\operatorname{logit}(p^*)=\Delta r^*\) 下，单个 Bernoulli label 无法逐 edge 无偏恢复
logit。SRM+ 对同一 edge 获取随机次独立 label：令
\(N\sim\mathrm{Geometric}(1-\gamma)\)（支撑从 1 开始）、
\(q_k=P(N\ge k)=\gamma^{k-1}\)、\(S_N=\sum_j a_j\)，并构造

\[
U^+_{k,N}=\frac{\binom{S_N}{k}}{\binom Nk},\qquad
U^-_{k,N}=\frac{\binom{N-S_N}{k}}{\binom Nk},
\]

\[
h=\sum_{k=1}^{N}\frac{U^+_{k,N}-U^-_{k,N}}{kq_k}.
\]

只要重复 label 条件 iid、随机截断独立且尾部条件成立，就有
\(\mathbb E[h\mid e]=\operatorname{logit}(p^*)=\Delta r^*\)。于是

\[
m_\phi=\frac12\mathbb E[z(t_\phi-h)]
=g_{r_\phi}-g_*,
\qquad
\mathcal L_{\mathrm{SRM+}}
=\frac1{2\beta}m_\phi^\top F_0^\dagger m_\phi.
\]

### 必须区分的 exact theorem 与工程目标

population 定理使用 \(F_0^\dagger\)。有限样本中 \(d>n_F\)，经验 Fisher 必然秩亏；
代码明确优化

\[
\widehat m_\phi=\frac1{2n_E}Z^\top(t_\phi-h),\qquad
\widehat F=\frac1{n_F}S^\top S,
\]

\[
\widehat L_\lambda(\phi)
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F+\lambda I)^{-1}\widehat m_\phi,
\quad
\lambda=c\,\mathrm{mean}(\mathrm{diag}(\widehat F))>0.
\]

PCG 求解 \((\widehat F+\lambda I)v=\widehat m_\phi\)，外层反传时 detach \(v\)，并严格
执行

```text
full margins -> full moment -> warm-start PCG -> detach(v)
             -> exactly one optimizer step -> repeat
```

所以 \(\lambda>0\) 的实际实现是 **ridge-regularized empirical target**，不是未阻尼
population pseudoinverse 定理的有限样本复刻。完整假设、推导和禁止越界的论文表述见
[docs/theory.md](docs/theory.md)。

## 2. 固定的 Phase 1 实验

正式设计只检验一条机制链：在有意受限、因而可能 misspecified 的 reward class 中，
SRM+ 是否比 repeated-label BT-MLE 更准确地恢复真实局部 policy update。

| 组件 | 锁定值 |
|---|---|
| Prompt | MultiPref 固定 revision；去重后按 prompt 划分 1536/256/256 |
| Reference policy | `Qwen/Qwen2.5-0.5B-Instruct`，固定 revision，FP32 |
| Candidate | 每 prompt 原分布独立采样 4 个；不使用 beam、top-k/top-p 截断、筛选或去重 |
| Policy tangent | 最后四层 `q_proj/v_proj`、rank-4 fixed-A LoRA；A 冻结、B=0，只有 B 是坐标 |
| Policy score | 完整 response（含 EOS）的 sequence log-prob 对 LoRA-B 的梯度；不做长度归一化 |
| Oracle | `Skywork/Skywork-Reward-V2-Qwen3-0.6B` 固定 revision，FP32 |
| Oracle transform | 仅用 train node 拟合 median/MAD，再做有界 `tanh`；冻结后用于全部 split |
| Label | candidate 0 vs 1；`gamma=0.9` randomized geometric truncation；SRM edge 等权 |
| Reward class | 同一 Qwen frozen final-response-token feature + 零初始化、无 bias linear head |
| Baseline | repeated-label BT-MLE；每个原始 Bernoulli label 都贡献一次 likelihood |
| 优化 | 两者同样本、初始化、720 steps、AdamW、lr `1e-3`、weight decay 0、microbatch 64 |
| Ridge | 主值 `c=1e-3`；必须同时报告 `1e-4`、`1e-2` |
| Seeds | `20260722`–`20260726` 五个严格配对 seed |
| Policy evaluation | shared KL probe 上 measured sequence forward KL `0.01 ± 5%`；test common-random rollout |

Phase 1 模型和数据 revision 均为 40 位 commit SHA，详见
[configs/main.yaml](configs/main.yaml)。`configs/smoke.yaml` 只缩小规模做 GPU 验收，不能作为
正式结果；尚未执行的 CoVal Phase 2 必须使用独立的固定-revision config。

数据流如下；oracle target 不可能进入训练 dataclass：

```text
MultiPref prompts
    -> pi_0: 4 exact-token candidates / prompt
       -> LoRA-B scores S ---------> Fisher geometry
       -> frozen hidden features --> same zero-init linear RM
       -> frozen oracle -----------> train-only calibration -> repeated BTL labels
                                      |                         |
                                      +-> BT-MLE                +-> SRM+
                                             \                 /
                                      held-out local geometry
                                               |
                                  matched measured-KL rollouts
```

### 预注册成功判据

本项目不以 pairwise accuracy 上升定义成功。只有以下链条全部满足，才支持“SRM+ 改善
policy-relevant reward learning”的主结论：

1. 主阻尼 \(c=10^{-3}\) 上，五 seed 的 `SRM+ - BT` test local regret 配对均值小于 0，
   且 95% paired bootstrap interval 上界小于 0；
2. 同一主阻尼上，squared Fisher direction error 满足同一负向判据，Fisher cosine 的
   配对差为正；
3. 两种 learner 都在同一 KL probe 上实际达到 `0.01 ± 5%`，且 test common-random
   rollout 中 `SRM+ improvement - BT improvement` 的五 seed 配对均值及其 95% bootstrap
   interval 下界均大于 0；
4. \(c\in\{10^{-4},10^{-2}\}\) 的 `SRM+ - BT` local-regret 配对均值都严格 `<0`，
   所有正式 PCG 与 KL line search 均收敛；exact zero 视为 inconclusive/`not_passed`。

Pairwise accuracy、raw oracle score 和单 seed 结果只作诊断。任一主条件失败，就按失败或
证据不足报告，不用额外调参、挑 checkpoint 或改 seed 挽救主结论。

## 3. 安装与本地验证

Python 要求 `>=3.10`。CPU 数值内核：

```bash
python -m pip install -e ".[dev]"
smart-reward config-check configs/smoke.yaml
smart-reward synthetic-check --seed 0 --output outputs/synthetic.json
pytest -q
ruff check .
ruff format --check .
```

`synthetic-check` 会经过真实 randomized label、BT/SRM trainer、PCG 和 held-out metric，
但输出明确标记 `benchmark_only=true`，也不断言 SRM+ 必胜。

真实 Hugging Face 管线额外安装：

```bash
python -m pip install -e ".[llm,dev]"
```

完整跑 `safetensors` artifact contract 测试也使用这一组 extras；安装依赖本身不会下载模型。
LLM extra 锁定 `transformers>=4.52.3,<5`：下界来自 pinned Qwen3 Skywork oracle 所需的
`Qwen3ForSequenceClassification`，正式 image 不接受 Transformers 5.x。

正式 run 默认 `local_files_only=True`；必须提前缓存 config 中固定 revision。只有显式
传 `--allow-download` 才允许 materialization 访问网络，HPC 作业不使用该开关。

## 4. 可执行实验链

以下以 smoke config 展示接口；正式实验替换为 `configs/main.yaml` 并逐个运行五个
configured seed。

```bash
seed=20260722
run_dir="outputs/smoke/seed-${seed}"
mkdir -p "${run_dir}"

smart-reward env-report configs/smoke.yaml \
  --seed "${seed}" --repo-root . --output "${run_dir}/run-manifest.json"

smart-reward controlled-materialize configs/smoke.yaml \
  "${run_dir}/artifact" --seed "${seed}" --device cuda

smart-reward controlled-compare configs/smoke.yaml \
  "${run_dir}/artifact" "${run_dir}/comparison.json" \
  --seed "${seed}" --device cuda \
  --run-manifest "${run_dir}/run-manifest.json"
```

这组直接命令展示 `controlled.sbatch` 内部接口，不得在登录节点运行。正式 comparison 要求
manifest 是本 seed 的 formal Slurm record：clean Git、相同 `SRM_GIT_COMMIT`、image SHA、
account `sigroup`、partition 与唯一可见 GPU 均须完整，而且当前进程必须逐字段匹配。无
Slurm/formal 环境变量时可在本地用 CPU 或 CUDA 调试 comparison，但输出明确标为
`formal: false`，不能进入 matched-KL rollout 或正式 aggregate。

Matched-KL evaluator已实现为 `evaluate_matched_kl_rollouts(...)`，CLI/Slurm 入口见下文的
最终命令（正式运行同样保持离线）：

```bash
smart-reward controlled-rollout configs/smoke.yaml \
  "${run_dir}/artifact" "${run_dir}/comparison.json" "${run_dir}/rollout.json" \
  --seed "${seed}" --device cuda
```

它重新建立同一 fixed-A/zero-B 坐标，分别由两个 head 构造 train-only natural direction，
Fisher 二次型只提供步长初值；最终接受条件始终是保存的 reference candidate 上 **先对
response token 求和、再对 sequence 求均值**的全词表 forward KL，与 sequence log-prob
score/Fisher 保持同一尺度，绝不除以 response token 总数。zero-B、BT update、SRM+ update
在每个 test prompt 上重置同一派生 seed，得到 candidate-index 对齐的 common-random 三路
rollout；policy 卸载后只加载一次 oracle，并复用 Phase 1 冻结的 transform。Phase 1 原
test candidates 只作不配对的 descriptive sanity check，不能充当新 rollout 的 paired
reference。统计单位是 prompt：先在每个 prompt 内平均 4 个 candidate 的
`updated-reference`，再跨 test prompts 计算均值与 sample SE。

五 seed comparison 聚合：

```bash
smart-reward aggregate-results configs/main.yaml outputs/main/aggregate.json \
  outputs/main/seed-20260722/comparison.json \
  outputs/main/seed-20260723/comparison.json \
  outputs/main/seed-20260724/comparison.json \
  outputs/main/seed-20260725/comparison.json \
  outputs/main/seed-20260726/comparison.json \
  --rollouts \
  outputs/main/seed-20260722/rollout.json \
  outputs/main/seed-20260723/rollout.json \
  outputs/main/seed-20260724/rollout.json \
  outputs/main/seed-20260725/rollout.json \
  outputs/main/seed-20260726/rollout.json
```

聚合器要求 comparison/rollout seed 集与 config **完全相等**，并逐 seed 强校验 config、
同目录 `run-manifest.json`、manifest SHA、artifact metadata、comparison bytes 和
`updated_rollouts.jsonl` SHA256 绑定。五个 manifest 还必须具有完全相同的 Git commit、
image SHA256、Slurm account/partition 和唯一 GPU model；不一致时拒绝聚合。输出同时包含
主阻尼 held-out metric 与 prompt-level `test_rollout_improvement` 的每 seed `SRM - BT`、
配对均值、样本标准差、标准误和确定性 percentile bootstrap interval；它不会自动输出
p-value 或“显著”标签。它还对 config 声明的每个 damping multiplier 自动形成五 seed
local-regret evidence，显式记录 sensitivity PCG failure 与 strict-negative non-reversal
（paired mean `<0`；exact zero 不通过），并按上文固定判据写出
`pre_registered_evidence.status` 和 `supports_pre_registered_claim`。这是执行预注册决策
规则，不是从多个结果中事后挑主值。

### 产物契约

`controlled-materialize` 新建且拒绝覆盖以下 artifact：

```text
artifact/
├── metadata.json             # config/seed/split/evidence/tensor SHA256
├── tensors.safetensors       # 固定 11-key tensor schema
├── prompts.jsonl             # prompt/v1
├── candidates.jsonl          # exact token IDs、response mask、终止状态
├── training_edges.jsonl      # repeated labels、N、left wins、h；无 true reward
└── evaluation_edges.jsonl    # validation/test true margin；evaluation only
```

训练 tensor schema 只有 `policy_scores`、`reward_features`、`h`、`left_wins` 和
`num_annotations`；`true_rewards` 只存在于 validation/test。读取时先校验 safetensors
SHA256，再校验 key、shape、dtype、有限性、config hash、seed 和 prompt split。候选、
chat template、LoRA-A、参数 layout、pinned revisions 与正式 producer（Git/image digest）
也进入 evidence。

后续输出为：

- `comparison.json`：config 声明的 damping（main 为三档）下两个 head、训练/held-out
  metric、head SHA256、PCG evidence、artifact metadata SHA256、run-manifest SHA256 与
  formal environment identity；
- `rollout.json`：`matched-kl-rollout/v1`，包含 direction、measured-KL 和相对同 seed zero-B
  rollout 的 paired transformed-oracle improvement；同目录生成不可覆盖的
  `updated_rollouts.jsonl`（reference/BT/SRM+ 三路），并记录 artifact metadata、comparison、
  rollout JSONL 与 run-manifest 的 bytes-level SHA256，以及逐字段匹配的 formal environment；
- `aggregate.json`：`paired-seed-aggregate/v1`，含 held-out/rollout 五 seed paired metrics、
  每档 `damping_evidence`、逐项 `pre_registered_evidence.criteria`、config hash 与逐 seed
  source digests，以及五 seed 共享的 `environment_identity`；
- `run-manifest.json`：规范化 config、`selected_seed`、named seeds、Git dirty state、
  包/CUDA/GPU/Slurm account/partition 与设备可见性 allowlist；不会转储整个环境或 credential。

## 5. HKUST HPC4

邮件提供的信息只确认 account `sigroup`、登录地址及候选分区；QoS、wall-time、GPU
显存、驱动和外网状态必须在集群上验证。登录节点只做检查、传输和 `sbatch`，不加载模型。

```bash
# 1. 预检 project/scratch、quota、partition、Apptainer
bash scripts/hpc4/preflight.sh

# 2. 指向预先构建且校验的镜像与离线 snapshot cache
export SRM_IMAGE=/project/sigroup/smart-reward-model/images/srm.sif
export SRM_IMAGE_SHA256="$(sha256sum "${SRM_IMAGE}" | awk '{print $1}')"
export SRM_HF_CACHE=/project/sigroup/smart-reward-model/hf-cache

# 3. 十分钟单 GPU 容器验收
bash scripts/hpc4/submit_gpu_smoke.sh gpu-l20

# 4. smoke 配置；walltime 由 smoke 实测后填写
bash scripts/hpc4/submit_controlled.sh configs/smoke.yaml gpu-l20 "HH:MM:SS"

# 5. smoke 全部通过后，再在同一分区提交五 seed 正式 array
export SRM_ARRAY_CONCURRENCY=1
bash scripts/hpc4/submit_controlled.sh configs/main.yaml gpu-l20 "D-HH:MM:SS"
```

脚本在容器内强制 `TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，校验镜像 SHA256，
显式传递 scheduler 的 GPU visibility；GPU smoke 还硬检恰好一个可见 GPU、容器内 Git、
Transformers `[4.52.3,5)`、Qwen3 classification class、关键包版本并保存排序后的
`pip freeze`。controlled smoke 对三个模型阶段分别保存 PyTorch allocated/reserved 峰值。每个 job 隔离到
`/scratch/$USER/smart-reward-model/jobs/$SLURM_JOB_ID`；可复用 artifact 和最终 run evidence
同步到 `/project/sigroup/smart-reward-model`。完整运行规范、目录和故障门槛见
[docs/hpc4.md](docs/hpc4.md)。

## 6. 工程不变量

- candidate generation 与 policy score 使用**同一 FP32 model instance、同一 token IDs、
  同一 zero-B state**；EOS 属于 response，禁止换模型重算 score。
- fixed-A LoRA 只训练/更新 B；A、参数名、shape、flatten offset 和 SHA256 必须完全匹配。
- Fisher 使用全部 4 个 node，不能用主动选择 edge 的 endpoint 频率代替；训练 moment 只用
  canonical candidate `0 - 1` edge。
- SRM+ 每个 edge 权重为 1；BT-MLE 每个原始 label 都计入。按随机 \(N\) 再给 SRM edge
  加权会改变目标。
- optimizer step 前必须用全部 edge 刷新 moment/PCG；microbatch 只用于梯度累积，不得形成
  batch-local Fisher 或 stale dual。
- held-out moment 使用每 prompt 无偏 covariance，分母为 \(P(M-1)\)；node Fisher 分母为
  \(PM\)。
- measured KL 是唯一 rollout 步长接受标准；quadratic KL 只作 initializer，每次 line-search
  trial 都从 zero-B 原点覆盖，失败必须恢复原点。
- 一次局部更新离开 \(\pi_0\) 后，下一轮必须重新生成 candidate、score 和 Fisher。

## 7. 项目结构

```text
Smart-Reward-Model/
├── configs/                  # 严格 closed-schema smoke/main 配置
├── docs/
│   ├── theory.md             # 数学规格与 theorem boundary
│   ├── experiment_protocol.md# 预注册实验协议
│   └── hpc4.md               # HPC4 运维规范
├── scripts/hpc4/             # preflight、GPU smoke、controlled Slurm array
├── src/smart_reward/
│   ├── annotations.py        # randomized geometric repeated labels / h
│   ├── scores.py             # per-sample LoRA-B score 与 layout
│   ├── objective.py          # moment、dual、envelope gradient
│   ├── pcg.py / linear.py    # matrix-free damped Fisher solve
│   ├── training.py           # 公平的 BT-MLE 与 SRM+ trainers
│   ├── phase1.py             # 真实模型 materialization 与 leakage-safe join
│   ├── artifacts.py          # 原子、完整性校验的 artifact I/O
│   ├── experiment.py         # 固定步数 paired comparison
│   ├── rollout.py            # natural direction 与 measured-KL update
│   ├── phase1_rollout.py     # 真实 test rollout + frozen oracle evaluation
│   ├── statistics.py         # paired seed bootstrap aggregation
│   └── cli.py                # fail-closed command line control plane
└── tests/                    # CPU 单元、性质、泄漏与集成测试
```

## 8. 当前边界与后续顺序

| 状态 | 内容 |
|---|---|
| 已实现并有 CPU 测试 | 数学内核、randomized estimator、score/layout、PCG/envelope、训练器、严格 schema、artifact、comparison、matched-KL 逻辑、统计聚合 |
| 已实现但未在当前机器真实执行 | pinned HF materialization、真实 Qwen/Skywork rollout、GPU/Apptainer/Slurm 作业 |
| 尚无结果 | 五 seed 主实验、damping sensitivity、matched-KL downstream comparison |
| 主链通过后才做 | 高容量 LoRA reward model scale-up、CoVal 人类 robustness |

CoVal 只有固定有限标签时，实验名称必须是 **candidate-restricted truncated SRM+
robustness**；它不能援引无限 randomized truncation 的精确无偏定理，也不能替代 controlled
Phase 1。

## 9. 官方依赖与固定资产

- [PyTorch 文档](https://docs.pytorch.org/docs/stable/index.html)
- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [PEFT LoRA API](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [MultiPref dataset](https://huggingface.co/datasets/allenai/multipref)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Skywork Reward V2 Qwen3 0.6B model card](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
- [CoVal dataset](https://huggingface.co/datasets/openai/coval)

精确 revision 以 config 为准，不以网页当前 `main` 为准。
