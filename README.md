# Smart Reward Model

本仓库实现一个由局部 RLHF regret 直接导出的、policy-aware 的 reward model
训练目标。核心不是给 Bradley–Terry 交叉熵附加启发式正则，而是估计进入局部
policy update 的 reward error moment，并用 reference policy 的 Fisher 几何度量它。

## 已锁定的数学对象

在 reference policy `pi_0`、允许移动的 policy tangent 参数 `theta` 和 KL 系数
`beta` 下，定义

```text
s_0(x,y) = grad_theta log pi_0(y|x)
F_0        = E[s_0 s_0^T]
A_0 r      = E[s_0 r(x,y)]
```

局部最优 policy update 是 `delta_r = beta^-1 F_0^dagger A_0 r`。用学到的
`r_phi` 代替潜在 reward `r*` 所产生的二阶局部 regret 为

```text
R_local(r_phi) = (1 / (2 beta))
                 ||A_0(r_phi-r*)||^2_{F_0^dagger}.
```

对 `e=(x,y,y')`，令 `z=s_0(x,y)-s_0(x,y')`、
`t_phi=r_phi(x,y)-r_phi(x,y')`。同一条边的随机次独立 BTL 标签可构造
`E[h|e]=Delta r*(e)` 的无偏 randomized-truncation U-statistic，因此

```text
m_phi = (1/2) E[z (t_phi-h)] = A_0(r_phi-r*)
L_SRM+ = (1/(2 beta)) m_phi^T F_0^dagger m_phi.
```

工程中通过 PCG 解 `(F_hat + lambda I)v=m_hat`，再用 envelope gradient 更新
reward model。`lambda>0` 时实现的是明确的 ridge-regularized local regret；所有实验
必须报告 `lambda` 并做灵敏度检查，不能把它写成与未阻尼伪逆目标完全相等。

完整推导、识别条件与论文表述边界见 [docs/theory.md](docs/theory.md)，实验契约见
[docs/experiment_protocol.md](docs/experiment_protocol.md)。

## 第一阶段固定实验

- Prompt：MultiPref 去重后按 prompt 划分，主实验取 2,048 个。
- Reference policy：`Qwen/Qwen2.5-0.5B-Instruct`，每个 prompt 原分布独立采样
  4 个 response；前两个形成训练 edge，四个都进入 Fisher node pool。
- Policy tangent：最后四层 `q_proj/v_proj` 的 fixed-A LoRA，只把 LoRA-B 视为
  `theta`。
- Controlled oracle：`Skywork/Skywork-Reward-V2-Qwen3-0.6B`，压缩 margin 后生成
  严格 BTL 重复标签。
- 对照：相同数据、RM 架构、初始化和计算预算下比较 repeated-label BT-MLE 与 SRM+。
- 主指标：held-out ridge local regret、natural-gradient direction error，以及相同
  近似 KL budget 下的 oracle reward improvement。
- 人类验证：CoVal candidate-restricted 实验；固定有限标签只能称为 truncated SRM+
  proxy，不能冒充主定理中的无偏 on-policy 实验。

模型和数据 revision 已写入 `configs/*.yaml`，正式 run 不允许使用浮动的 `main`。

## 本地数值验证

当前阶段的纯数学内核只依赖 PyTorch，可在 CPU 上运行：

```bash
python -m pip install -e ".[dev]"
pytest
```

LLM 管线依赖：

```bash
python -m pip install -e ".[llm,dev]"
```

CUDA/PyTorch 的最终 lock 只在 HPC4 GPU smoke job 确认驱动和容器后生成；不会用
未经节点验证的本地 CPU 环境锁替代正式实验环境。

## HPC4

邮件已确认 Slurm account 为 `sigroup`。训练禁止在登录节点执行。首次登录先运行：

```bash
bash scripts/hpc4/preflight.sh
```

然后提交十分钟单 GPU 验收：

```bash
bash scripts/hpc4/submit_gpu_smoke.sh gpu-l20
```

缓存和临时 checkpoint 放 `/scratch/$USER/smart-reward-model`，模型主副本、最终结果
和可恢复 checkpoint 放 `/project/sigroup/smart-reward-model`。详细说明见
[docs/hpc4.md](docs/hpc4.md)。
