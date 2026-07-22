# Smart Reward Model（SRM+）

[![CI](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml)

> 不去拟合所有 reward error；只拟合会改变下一次 policy update 的那一部分。

SRM+ 是一个从**局部 KL-regularized policy regret**直接推导出的 policy-aware reward
learning 目标。标准 Bradley–Terry（BT）训练优化 preference prediction；SRM+ 优化
reward error 经 reference-policy Fisher 几何传递后造成的局部决策误差。两者在受限、
可能 misspecified 的 reward class 中一般不是同一个目标。

`SRM+` 是本仓库的方法名；当前正式 baseline 是 repeated-label BT-MLE，并不存在一个单独
实现、名为 `SRM` 的中间 baseline，因此“+”不应被解读为已完成的消融层级。

## 当前状态

| 项目 | 状态 |
|---|---|
| 数学规格、CPU 数值内核、真实模型管线、artifact、统计聚合 | 已实现 |
| 测试 | 216 tests；synthetic 只验证链路 |
| Slurm/Apptainer 控制面 | 已实现 |
| 经 HPC4 验证的 `.sif`、environment lock、HF offline cache | **尚未固化** |
| pinned Qwen/Skywork GPU smoke 与五 seed 主实验 | **尚未执行** |
| “SRM+ 优于 BT”效果结论 | **不存在，仍是待检验假设** |

仓库当前可称为“实验代码与运行控制面就绪”，不能称为“正式 GPU 环境和论文结果已就绪”。

## 1. 三个公式理解 SRM+

### 1.1 下游真正关心的是 update error

固定 prompt 分布、reference policy `pi_0=pi_{theta_0}`，以及下一步真正允许更新的 policy
tangent。定义

$$
s_0(x,y)=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0},\qquad
F_0=\mathbb E[s_0s_0^\top],\qquad
g_r=\mathbb E[s_0r(x,y)].
$$

在局部二阶 KL surrogate 中，reward `r` 产生
`delta_r=beta^{-1}F_0^dagger g_r`。因此 learned reward `r_phi` 在这个局部 surrogate 上的
regret 精确等于

$$
\boxed{
\widetilde{\operatorname{Reg}}(r_\phi)
=\frac1{2\beta}(g_{r_\phi}-g_*)^\top
F_0^\dagger(g_{r_\phi}-g_*)
}.
$$

它只惩罚 policy tangent 可见的 reward error。prompt 内常数或 score 零空间中的误差即使
pointwise MSE 非零，也不会改变当前局部 update。

### 1.2 Pairwise labels 可以识别 reward moment error

对同一 prompt 独立采样 `y,y' ~ pi_0`，令

$$
z=s_0(x,y)-s_0(x,y'),\qquad
t_\phi=r_\phi(x,y)-r_\phi(x,y').
$$

score identity 给出 `g_r=E[z*Delta r]/2`。单个 Bernoulli label 不能逐 edge 无偏恢复 BTL
logit；SRM+ 对同一 edge 使用随机数量的条件 iid labels，构造 randomized estimator `h`，使

$$
\mathbb E[h\mid e]=\operatorname{logit}(p^*(e))=\Delta r^*(e).
$$

于是

$$
\boxed{
m_\phi=\frac12\mathbb E[z(t_\phi-h)]
=g_{r_\phi}-g_*
}.
$$

### 1.3 有限样本优化 ridge Fisher-GMM

实现用全部 on-policy node scores `S` 估计 Fisher，用 canonical edge differences `Z`
估计 moment：

$$
\widehat F=\frac1{n_F}S^\top S,\qquad
\widehat m_\phi=\frac1{2n_E}Z^\top(t_\phi-h),
$$

$$
\boxed{
\widehat L_\lambda
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F+\lambda I)^{-1}\widehat m_\phi
},
\qquad
\lambda=c\,\operatorname{mean}(\operatorname{diag}\widehat F)>0.
$$

| 层级 | 可以声称什么 |
|---|---|
| Population、`lambda=0`、`F_0^dagger` | 与局部二阶 policy regret 精确等价 |
| Finite sample、`lambda>0` | ridge-regularized empirical surrogate |
| `c={1e-4,1e-3,1e-2}` | 检查结论是否依赖阻尼尺度 |

PCG、detached envelope gradient、二倍因子、randomized estimator 和全部假设见
[理论规格](docs/theory.md)。

## 2. Phase 1：一个可识别的算法对照

研究问题是：

> 在同一个受限 reward class 中，SRM+ 能否比 repeated-label BT-MLE 更准确地恢复
> operational oracle 所诱导的局部 policy update？

Phase 1 的 `r*` 是冻结并变换后的 Skywork reward-model score，是实验中的
**operational ground truth**，不是人类真实 utility。该阶段建立 controlled internal
validity，不允许直接外推成人类偏好结论。

BT 与 SRM+ 共享 candidate、labels、features、零初始化 head、optimizer、步数、GPU 和停止
规则；唯一改变的是 reward learning objective。这是 controlled paired algorithmic
contrast，不是对现实部署效果的无条件因果声明。

```text
MultiPref prompts
    -> pi_0: 4 exact-token candidates / prompt
       -> LoRA-B scores S ---------> Fisher geometry
       -> frozen hidden features --> zero-init linear reward class
       -> frozen oracle -----------> train-only calibration
                                      -> repeated BTL labels
                                             |          |
                                           BT-MLE      SRM+
                                             \          /
                                      held-out geometry
                                               |
                                  matched measured-KL rollouts
```

| 组件 | 锁定设计 |
|---|---|
| Prompt | MultiPref fixed revision；1536/256/256 prompt-level split |
| Reference policy | Qwen2.5-0.5B-Instruct fixed revision，FP32 |
| Candidates | 每 prompt 原分布独立采样 4 个；不筛选、不去重 |
| Policy tangent | 最后四层 `q_proj/v_proj`，rank-4 fixed-A LoRA-B |
| Oracle | Skywork-Reward-V2-Qwen3-0.6B fixed revision，FP32 |
| Labels | candidate `0-1`；`gamma=0.9` randomized repeated BTL labels |
| Reward class | frozen final-response-token feature + bias-free linear head |
| Training | 720 fixed steps；BT/SRM 共享全部优化条件 |
| Evaluation | held-out Fisher geometry + measured sequence-KL `0.01 ± 5%` rollout |
| Statistics | 5 paired seeds；主 damping + 两档 sensitivity |

linear head 施加了可审计的 capacity bottleneck，但“限制容量”本身不逻辑保证 oracle 一定
不可表示。Phase-1 artifact 因此在 `train_reward_class_projection` 中记录 train-only、
prompt-centered linear-projection residual；
它只作机制诊断，不参与调参或 checkpoint 选择。若 residual 接近数值零，只能说明在 train
candidates 上没有观察到线性不可表示证据；这不能排除 held-out 或 population
misspecification，但论文不得声称已在训练样本上实证建立 misspecification。

## 3. 什么结果才算成功

本项目不以 pairwise accuracy 上升定义成功。`aggregate.json` 只有在以下条件全部满足时才把
`pre_registered_evidence.status` 写为 `passed`：

| 证据 | 五 seed 的固定判据 |
|---|---|
| 主阻尼 held-out ridge local-regret proxy | `SRM-BT` mean `<0`，bootstrap upper `<0` |
| Squared Fisher direction error | `SRM-BT` mean `<0`，bootstrap upper `<0` |
| Fisher cosine | `SRM-BT` mean `>0`，且方向 Fisher norm 非零 |
| Matched-KL rollout improvement | 两者均达 KL 容差；`SRM-BT` mean `>0`，bootstrap lower `>0` |
| Damping sensitivity | 两档 local-regret mean 均 `<0`，PCG 全部收敛 |
| 数值与身份完整性 | 主链 PCG/KL 收敛，五 seed Git/image/GPU/manifest 身份一致 |

这里的 interval 是对 **5 个预注册 paired seeds** 做的 deterministic percentile-bootstrap
工程判定区间，不是 population p-value，也不授权使用“统计显著”措辞。

| 结果模式 | 允许的结论 |
|---|---|
| Geometry、rollout、sensitivity 全通过 | 支持预注册的 policy-aware mechanism claim |
| Geometry 通过，rollout 未通过 | 只支持局部 surrogate 改善，未建立 downstream transfer |
| Geometry 未通过 | 核心机制未获支持 |
| Sensitivity failure/reversal | 保留失败证据，主结论 `not_passed` |
| 只有 pairwise accuracy 改善 | 不构成 SRM+ 成功证据 |

## 4. 本地快速验证

CPU 数值内核：

```bash
python -m pip install -e ".[dev]"
smart-reward config-check configs/smoke.yaml
smart-reward config-check configs/main.yaml
smart-reward synthetic-check --seed 0 --output outputs/synthetic.json
pytest -q
ruff check .
ruff format --check .
```

`synthetic-check` 输出固定标记 `benchmark_only=true`，且测试不会断言 SRM+ 必须胜过 BT。

真实 Hugging Face 管线额外安装：

```bash
python -m pip install -e ".[llm,dev]"
```

正式 run 默认 `local_files_only=True`。model/dataset revision 全部由 config 的 commit SHA
锁定；网页当前 `main`、本地 mtime 或缓存下载时间都不是实验身份。

内部单 seed CLI、artifact schema、数据泄漏边界和聚合命令完整写在
[实验协议](docs/experiment_protocol.md)，不在 README 重复维护。

## 5. HKUST HPC4 入口

正式提交前仍有一个现实 blocker：必须在 HPC4 实测驱动/partition 后固化兼容的 Apptainer
definition、`.sif` SHA256、`pip freeze` 与 pinned HF offline cache。仓库不会猜测未验证的
CUDA base image。

```bash
# 登录节点：只做预检和提交，不加载模型
bash scripts/hpc4/preflight.sh

export SRM_IMAGE=/project/sigroup/smart-reward-model/images/srm.sif
export SRM_IMAGE_SHA256="$(sha256sum "${SRM_IMAGE}" | awk '{print $1}')"
export SRM_HF_CACHE=/project/sigroup/smart-reward-model/hf-cache

# 先验收容器，再跑真实模型 smoke
bash scripts/hpc4/submit_gpu_smoke.sh gpu-l20
export SRM_SMOKE_WALLTIME=REPLACE_WITH_MEASURED_WALLTIME
bash scripts/hpc4/submit_controlled.sh configs/smoke.yaml gpu-l20 "${SRM_SMOKE_WALLTIME}"

# smoke 全通过后，保持同一 image/partition/GPU 型号运行五 seed main array
export SRM_ARRAY_CONCURRENCY=1
export SRM_MAIN_WALLTIME=REPLACE_WITH_APPROVED_WALLTIME
bash scripts/hpc4/submit_controlled.sh configs/main.yaml gpu-l20 "${SRM_MAIN_WALLTIME}"
```

wall-time、GPU-hour 和存储预算只能用实际 smoke 记录填写，不能在仓库中虚构。完整的 storage
layout、offline staging、身份闭环、监控和故障处理见 [HPC4 运行规范](docs/hpc4.md)。

## 6. 文档与代码地图

| 目标 | 入口 |
|---|---|
| 理解全部推导、假设和 contribution boundary | [docs/theory.md](docs/theory.md) |
| 执行 Phase 0–1、理解指标和产物 | [docs/experiment_protocol.md](docs/experiment_protocol.md) |
| 在 HPC4 准备环境和提交 Slurm | [docs/hpc4.md](docs/hpc4.md) |
| 查看正式设计身份 | [configs/main.yaml](configs/main.yaml) |

```text
Smart-Reward-Model/
├── configs/                  # closed-schema smoke/main configs
├── docs/                     # theory、experiment protocol、HPC4 runbook
├── scripts/hpc4/             # preflight、GPU smoke、controlled array
├── src/smart_reward/
│   ├── annotations.py        # randomized repeated-label estimator
│   ├── objective.py          # moment、reported value、envelope gradient
│   ├── training.py           # paired BT-MLE / SRM+ trainers
│   ├── phase1.py             # real-model immutable materialization
│   ├── artifacts.py          # atomic integrity-checked artifact I/O
│   ├── rollout.py            # natural direction、measured-KL update
│   ├── phase1_rollout.py     # common-random test rollout
│   ├── statistics.py         # paired-seed aggregation
│   └── cli.py                # fail-closed control plane
└── tests/
```

## 7. Claim boundary 与后续顺序

1. 先完成 validated image/cache、GPU smoke 和 controlled smoke；
2. 再运行同一环境下的五 seed Phase 1；
3. 只有 aggregate 全部通过，才启动高容量 LoRA reward-model scale-up；
4. 最后做 CoVal human robustness。

CoVal 的固定有限 labels 只能识别 logit series 的截断，因此该阶段必须称为
**candidate-restricted truncated SRM+ robustness**，不能援引 Phase 1 的精确无偏定理，
也不能把 transformed Skywork operational ground truth 外推成人类 utility。

理论基础与本项目的组合贡献边界见 [docs/theory.md](docs/theory.md) 最后一节。官方工程资产：

- [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [MultiPref](https://huggingface.co/datasets/allenai/multipref)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Skywork Reward V2 Qwen3 0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
- [CoVal](https://huggingface.co/datasets/openai/coval)
