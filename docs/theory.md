# SRM+ 理论规格：从局部 policy regret 到 Fisher-GMM

本文档是仓库的数学源规范。符号、常数因子、估计对象以及“精确等价”和“工程近似”的
边界均以此处为准；实验设计见 [experiment_protocol.md](experiment_protocol.md)。

## 0. 一页阅读指南

SRM+ 的出发点不是“怎样更准确地预测人类偏好”，而是：

> 当 reward model 只用于产生下一次局部 policy update 时，应该只惩罚那些会改变该
> update 的 reward error。

推导分成四步：

1. 局部二阶 KL 几何把 downstream regret 写成一个 Fisher 逆范数；
2. score identity 把不可观测的 reward moment 改写成 pairwise moment；
3. randomized repeated labels 为 BTL logit margin 提供逐 edge 无偏估计 `h`；
4. 有限样本中用 ridge Fisher 和 PCG 得到可训练的 Fisher-GMM 目标。

下表先固定最常用的符号。除非特别说明，期望均包含
`x ~ rho, y ~ pi_0(.|x)`。

| 符号 | 含义 |
|---|---|
| `rho` | 固定 prompt 分布 |
| `pi_0=pi_{theta_0}` | 产生 candidate 的 reference policy |
| `theta` | 下一步真实允许更新的 policy tangent 坐标 |
| `s_0(x,y)` | `nabla_theta log pi_theta(y|x)` 在 `theta_0` 的 score |
| `F_0` | reference policy 在上述坐标中的 Fisher |
| `r*`, `r_phi` | 目标/oracle reward 与待训练 reward model |
| `g_r` | reward 对局部 policy update 的一阶驱动力 |
| `z`, `t`, `h` | pair score difference、预测 margin、真实 margin 的无偏估计 |
| `beta` | KL regularization 系数 |

## 1. 局部决策问题

### 1.1 Reward 如何产生一次局部更新

定义

$$
s_0(x,y)=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0},\qquad
F_0=\mathbb E[s_0s_0^\top],\qquad
g_r=\mathbb E[s_0r(x,y)].
$$

在 `theta_0` 附近，对 KL-regularized policy objective 做二阶展开，reward `r` 所对应的
局部问题是

$$
J_r(\delta)=g_r^\top\delta-\frac{\beta}{2}\delta^\top F_0\delta.
$$

在 identifiable tangent space 上取最小范数解：

$$
\delta_r=\arg\max_\delta J_r(\delta)
=\beta^{-1}F_0^\dagger g_r.
$$

这里的坐标必须和真实下游 update 完全相同。若实验只允许更新 fixed-A LoRA-B，那么
`s_0`、`F_0` 和最终施加的 update 都必须使用同一组 LoRA-B 坐标。

### 1.2 用目标 objective 评价 learned-reward update

目标 reward `r*` 的局部 objective 记为

$$
G^*(\delta)=g_*^\top\delta-\frac{\beta}{2}\delta^\top F_0\delta,
\qquad
\delta^*=\beta^{-1}F_0^\dagger g_*.
$$

reward model 的错误不是 `r_phi-r*` 的逐点误差，而是它产生的 update `delta_phi` 在
真实 objective 上损失多少：

$$
\widetilde{\operatorname{Reg}}(r_\phi)
=G^*(\delta^*)-G^*(\delta_\phi).
$$

对二次函数配方，并使用
`delta_phi-delta*=beta^{-1}F_0^dagger(g_phi-g*)`，得到

$$
G^*(\delta^*)-G^*(\delta_\phi)
=\frac{\beta}{2}(\delta_\phi-\delta^*)^\top
F_0(\delta_\phi-\delta^*),
$$

$$
\boxed{
\widetilde{\operatorname{Reg}}(r_\phi)
=\frac1{2\beta}(g_{r_\phi}-g_*)^\top
F_0^\dagger(g_{r_\phi}-g_*)
}.
$$

这是 population、未阻尼、局部二阶问题中的精确等式。

### 1.3 几何直觉：只看 policy 可见的 reward error

在有限离散表示中，令 `A_theta` 的列为各 `(x,y)` 的 score，
`D_0=diag(rho*pi_0)`，并定义 `B=A_theta D_0^(1/2)`。则

$$
F_0=BB^\top,
$$

且上面的 regret 等于

$$
\frac1{2\beta}\left\|
P_{\operatorname{row}(B)}D_0^{1/2}(r_\phi-r^*)
\right\|_2^2.
$$

因此：

- prompt 内加常数不会改变 update；
- 落在 score 零空间中的 reward error 不会被惩罚；
- 两个 pointwise fit 同样好的 reward model，可能产生完全不同的 policy update；
- 在 reward class misspecified 时，BT likelihood 最优不保证等于 policy regret 最优，通常不同。

这正是 SRM+ 与普通 preference prediction 目标的根本区别。

一个最小例子可以直接看出差别。设一个 prompt 有三个等概率 candidates，唯一 policy
score 坐标为 `s=(-1,0,1)`，reward error 为 `e=(1,-2,1)`。逐点 MSE 明显非零，但
`E[s*e]=0`，所以该 error 在当前 tangent 下不改变 policy update，local regret 为零。
SRM+ 忽略它不是“少学了信息”，而是没有为当前决策支付无用的拟合成本。

## 2. 从 node reward moment 到 pairwise observation

对同一 prompt 条件独立采样 `y,y' ~ pi_0(.|x)`，定义

$$
z=s_0(x,y)-s_0(x,y'),\qquad
t_r=r(x,y)-r(x,y').
$$

方向约定始终是 `left - right`：`a=1` 表示 left/`y` 胜出；交换 edge 时必须同时翻转
`z,t,h` 并把 `left_wins` 映射为 `N-left_wins`。

score identity 给出 `E[s_0|x]=0`。展开 pair moment：

$$
\begin{aligned}
\mathbb E[zt_r\mid x]
&=\mathbb E[(s-s')(r-r')\mid x]\\
&=2\,\mathbb E[sr\mid x],
\end{aligned}
$$

交叉项因为 `y,y'` 条件独立且 `E[s|x]=0` 消失。因此

$$
\boxed{g_r=\frac12\mathbb E[zt_r]}.
$$

同理，

$$
\boxed{F_0=\frac12\mathbb E[zz^\top]}.
$$

两个等式在 population 中等价，但工程估计器不必使用同一种表示：

- Fisher 用每个 prompt 的全部 on-policy node score 估计，方差更低；
- reward error moment 用被标注 edge 的 score difference 估计；
- 不能用主动选择 edge 的 endpoint 频率替代原 on-policy node Fisher。

Phase 1 中，`P` 个 train prompts、每 prompt `M=4` 个 candidates、policy tangent 维数 `d`、
reward feature 维数 `H` 对应：

```text
S:          (P*M, d)   # 全部 on-policy nodes，可与 edge endpoints 重叠
Z:          (P, d)     # 每 prompt canonical candidate 0 - 1
left/right: (P, H)     # frozen reward features
t, h:       (P,)
F_hat = S.T @ S / (P*M)
m_hat = Z.T @ (t-h) / (2*P)
```

node Fisher 与 pair moment 估计同一 population 几何，但有限样本数值不要求彼此相等。

## 3. 为什么单个 Bernoulli label 不够

### 3.1 单标签不可能逐 edge 无偏恢复 logit

在单位温度 BTL 模型下（等价地，温度已吸收到 reward 尺度），给定 edge `e`：

$$
a\mid e\sim\operatorname{Bernoulli}(p^*(e)),\qquad
p^*(e)=\sigma(t^*(e)),\qquad t^*(e)=\Delta r^*(e).
$$

对单标签任意统计量 `H(a)`，

$$
\mathbb E[H(a)\mid e]=(1-p^*)H(0)+p^*H(1),
$$

它关于 `p*` 必为仿射函数，不可能在一个区间上等于非线性的 `logit(p*)`。所以，在
“逐 edge、distribution-free、不跨 edge 拟合条件概率”的限定下，单标签不能给出精确
SRM+ target。这个结论不否定参数化概率模型或跨样本平滑，但那会引入另一层模型误差。

### 3.2 Logit 级数与 randomized truncation

对 `p in (0,1)`，

$$
\operatorname{logit}(p)
=\sum_{k=1}^{\infty}\frac{p^k-(1-p)^k}{k}.
$$

对同一 edge 获取条件 iid labels `a_1,...,a_N`。记 `S_N=sum_j a_j`，并定义

$$
U^+_{k,N}=\frac{\binom{S_N}{k}}{\binom Nk},\qquad
U^-_{k,N}=\frac{\binom{N-S_N}{k}}{\binom Nk}.
$$

条件于 `N>=k`，两个 U-statistic 的期望分别是 `p^k` 与 `(1-p)^k`。令随机重复数 `N`
独立于 edge 和 labels，生存概率为 `q_k=P(N>=k)>0`，则 Russian-roulette 估计量

$$
h=\sum_{k=1}^{N}\frac{U^+_{k,N}-U^-_{k,N}}{kq_k}
$$

满足

$$
\boxed{\mathbb E[h\mid e]=\operatorname{logit}(p^*(e))=t^*(e)}.
$$

关键的 survival correction 是

$$
\mathbb E\!\left[
\frac{\mathbf 1\{N\ge k\}}{q_k}U^+_{k,N}\mid e
\right]=p^{*k},
$$

负项同理。`1/q_k` 抵消第 `k` 项只有在 `N>=k` 时才被计算的概率；U-statistic 则用同一批
labels 无偏估计幂。实现只需要 `(S_N,N)` 的组合计数，不需要枚举 label 子集。

主实验固定

$$
P(N=n)=(1-\gamma)\gamma^{n-1},\qquad
q_k=\gamma^{k-1},\qquad \gamma=0.9.
$$

此时 `E[N]=10`。本实现要求
`gamma > max(p*,1-p*)`：oracle transform 保证 `p* in [0.25,0.75]`，所以主实验满足
`0.9>0.75`。这为级数项保留足够的 survival tail，并满足本实验使用的有限二阶矩条件。
不得硬截断、clip 或因
`N` 较大而重采样；这些操作都会改变原估计量并引入偏差，guard 只能使 run 失败。

### 3.3 权重语义

随机 `N` 只决定 `h` 的构造成本，不改变 edge 分布：

- SRM+ 中每条 edge 对 moment 恰好贡献一次；
- 不得再按 `N` 给 SRM+ edge 加权；
- BT-MLE baseline 使用全部原始 Bernoulli labels，因此等价于按 `N` 累计 likelihood。

## 4. Fisher-GMM 目标

定义预测 margin `t_phi=r_phi(x,y)-r_phi(x,y')` 和 moment

$$
m_\phi=\frac12\mathbb E[z(t_\phi-h)].
$$

利用 `E[h|e]=t*(e)` 与第 2 节的 pair identity：

$$
m_\phi
=\frac12\mathbb E[z(t_\phi-t^*)]
=g_{r_\phi}-g_*.
$$

于是 population SRM+ loss 与 local regret 精确相同：

$$
\boxed{
\mathcal L_{\mathrm{SRM+}}(\phi)
=\frac1{2\beta}m_\phi^\top F_0^\dagger m_\phi
=\widetilde{\operatorname{Reg}}(r_\phi)
}.
$$

对应的 Fenchel 形式为

$$
\min_\phi\max_v\;
\frac1{2\beta}\mathbb E[(z^\top v)(t_\phi-h)]
-\frac1{2\beta}\mathbb E[(s^\top v)^2].
$$

最小范数最优 dual witness 是 `v=F_0^dagger m_phi`。它指出哪些 policy coordinates 正在放大
当前 reward moment error，但不是某个 reward learner 的 rollout update。为避免术语混淆：

- `u_r=(F+lambda I)^-1 g_r` 是未乘 `1/beta` 的 natural-gradient direction；
- `delta_r=u_r/beta` 才是局部 policy displacement；
- `v=(F+lambda I)^-1 m` 是 reward-error 的 dual witness；population、`lambda=0` 时
  `v=beta*(delta_phi-delta*)`。

## 5. 从 population theorem 到有限样本目标

令 `S in R^(n_F x d)` 的每一行为一个 node score，`Z in R^(n_E x d)` 的每一行为
一个标注 edge difference。实现固定使用

$$
\widehat F=\frac1{n_F}S^\top S,
\qquad
\widehat m_\phi=\frac1{2n_E}Z^\top(t_\phi-h).
$$

通常 `d>n_F`，所以经验 Fisher 必然秩亏。工程目标显式采用

$$
\widehat L_\lambda(\phi)
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F+\lambda I)^{-1}\widehat m_\phi,
$$

其中

$$
\lambda=c\,\operatorname{mean}(\operatorname{diag}\widehat F)>0.
$$

这里必须区分：

| 层级 | 对象 | 可以声称什么 |
|---|---|---|
| population theorem | `F_0^dagger`, `lambda=0` | 与局部 regret 精确等价 |
| finite-sample experiment | `F_hat+lambda I`, `lambda>0` | ridge-regularized empirical surrogate |
| sensitivity | `c in {1e-4,1e-3,1e-2}` | 结论是否依赖阻尼尺度 |

不得把有限样本 ridge loss 写成 population pseudoinverse theorem 的“精确实现”。

isotropic ridge `lambda*I` 也不具任意 tangent 重参数化不变性；
`lambda=c*mean(diag(F_hat))` 只消除统一的全局尺度变化，不能抵消各坐标的非等比例变换。
因此 fixed-A seed/state、LoRA alpha、B 参数顺序、shape、scale 和 hash 都是经验目标定义的一部分，
不只是运行 provenance。训练与 rollout 必须复用完全相同的坐标系。

同一机制在实验中出现为三个不同对象：

| 用途 | Moment/Fisher | 阻尼语义 |
|---|---|---|
| theorem | population `g_phi-g*`, `F_0` | pseudoinverse，`lambda=0` |
| train | canonical-pair `m_hat`, train node `F_hat` | `lambda_train` 由 train Fisher 解析 |
| held-out | prompt covariance `g_hat_error`, held-out node Fisher | 每个 split 独立解析 `lambda_split` |

held-out covariance 直接估计 `g_error`，所以没有 pair identity 的 `1/2`；这不是常数因子冲突。

## 6. PCG、目标值与 envelope gradient

### 6.1 Matrix-free solve

每一步先求解

$$
(\widehat F+\lambda I)v=\widehat m_\phi.
$$

PCG 只需要矩阵向量积

$$
u\mapsto \frac1{n_F}S^\top(Su)+\lambda u,
$$

无需显式构造 `d x d` Fisher。求解必须满足相对残差门槛；不收敛不是可忽略 warning。

### 6.2 两个容易混淆的 scalar

求得 `v` 后，报告的目标值是

$$
\widehat L_\lambda=\frac1{2\beta}\widehat m_\phi^\top v.
$$

但若在 autograd 中 detach `v`，直接对上述 scalar 反传会少一个来自二次型对称性的因子 2。
正确的 envelope gradient 是

$$
\nabla_\phi\widehat L_\lambda
=\frac1{2\beta n_E}\sum_i
(z_i^\top v)\nabla_\phi t_{\phi,i}.
$$

因此代码使用 mean-reduced surrogate

$$
\widehat L_{\mathrm{env}}
=\frac1{n_E}\sum_i
\frac{z_i^\top\operatorname{stopgrad}(v)}{2\beta}
(t_{\phi,i}-h_i).
$$

`L_env` 的数值不是要报告的 objective；它的职责只是产生精确 envelope gradient。仓库分别
实现 `dual_loss`（报告值）和 `envelope_surrogate`（训练梯度），禁止混用。

同一个 full batch、精确 solve 下，二者数值关系为

$$
\widehat L_{\mathrm{env}}
=\frac1\beta v^\top\widehat m
=2\widehat L_\lambda.
$$

完整 saddle diagnostic 是

$$
\widehat L_{\mathrm{saddle}}(\phi,v)
=\frac1\beta\left(v^\top\widehat m-\frac12v^\top
(\widehat F+\lambda I)v\right).
$$

当 `v` 精确最优时它的数值等于 `L_lambda`，并给出同一个 envelope gradient。训练 surrogate
省略了与 `phi` 无关的二次项，所以数值变为 `2*L_lambda`；代码同时记录 reported value 与
saddle value，二者在 PCG 未收敛时不必相同。

### 6.3 外层更新顺序

每次 optimizer update 必须执行：

```text
all edge margins
  -> one full moment m_hat
  -> warm-start PCG solve
  -> detach v
  -> accumulate one full-data envelope gradient by microbatches
  -> exactly one optimizer step
  -> recompute everything
```

microbatch 只改变梯度累积的内存占用，不能创建 batch-local `m` 或 `F`；一个 `v` 不能作为
stale dual direction 连续训练多个 outer steps。

## 7. Held-out 几何与 downstream direction

对于每个 held-out prompt 的 `M` 个 candidates，prompt-level reward gauge 必须被消去。使用

$$
\widehat g_r
=\frac1{P(M-1)}\sum_{i=1}^{P}\sum_{j=1}^{M}
(s_{ij}-\bar s_i)(r_{ij}-\bar r_i).
$$

这里 covariance 的无偏分母是 `P(M-1)`；held-out node Fisher 仍使用 `PM`。主要几何指标为：

1. reward moment error 的 ridge local regret；
2. predicted 与 target damped natural direction 的 undamped-Fisher squared error；
3. 两个 direction 的 Fisher cosine。

这里报告的是 **held-out ridge local-regret proxy**。令
`Delta u=(F+lambda I)^-1 m_error`，则 squared Fisher direction error 是
`Delta u^T F Delta u`。只有 `lambda=0` 时它与 regret 满足简单的 `2*beta` 比例；
`lambda>0` 时二者是相关但不同的预注册指标。

真正下游 rollout 进一步把 BT/SRM+ direction 分别匹配到同一个**实测 sequence-level
forward KL**。Fisher 二次近似只用于提供 line-search 初值，不能作为接受更新的最终证据。

## 8. 假设、失效方式与工程保护

| 假设 | 若违反会发生什么 | 当前保护 |
|---|---|---|
| 局部二阶 KL 有效 | regret 等式不能外推到大更新 | measured-KL budget `0.01`；一次局部更新 |
| candidate 真来自 `pi_0` | score identity/Fisher 分布错配 | 同一 FP32 instance 生成并计算 score |
| tangent 坐标一致 | 优化了不会被实际 update 使用的方向 | fixed-A、zero-B、同一参数 layout/hash |
| 同一 edge labels 条件 iid BTL | `h` 不再识别同一个真实 margin | controlled oracle Phase 1；现实数据只作 robustness |
| `N` 独立且生存概率正确 | randomized estimator 有偏 | named RNG stream、禁止硬截断 |
| reward/score 二阶可积 | moment/Fisher 不稳定 | bounded oracle transform、概率 floor |
| train/evaluation 隔离 | target leakage 使比较失效 | train dataclass 不允许 true reward |
| PCG/KL search 收敛 | direction 或步长没有定义 | fail closed，不丢弃失败 seed |
| 所搜正向 ray 上 measured KL 可局部单调 bracket | 二分法不能可靠定位 target | 只接受实测 KL 达容差；否则 fail closed |

policy 一旦离开 `pi_0` 并继续训练，必须重新生成 candidates、scores 和 Fisher；本定理不支持
把旧几何无限复用到多轮 RLHF。

## 9. 数学对象与代码位置

| 数学对象 | 实现 |
|---|---|
| randomized `h` | `src/smart_reward/annotations.py` |
| `m_hat`、reported value、envelope weights | `src/smart_reward/objective.py` |
| matrix-free Fisher/PCG | `src/smart_reward/linear.py`, `pcg.py` |
| fixed-A LoRA score/layout | `src/smart_reward/hf.py`, `scores.py` |
| BT/SRM+ fixed-step trainers | `src/smart_reward/training.py` |
| held-out local geometry | `src/smart_reward/metrics.py` |
| natural direction与 measured KL | `src/smart_reward/rollout.py`, `policy_update.py` |
| real-model orchestration | `src/smart_reward/phase1.py`, `phase1_rollout.py` |

## 10. 理论基础与组合贡献边界

本项目使用的基础工具各有明确来源：

- Fisher/natural policy-gradient 几何建立在
  [Kakade, 2001](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
  以及 trust-region policy optimization
  [Schulman et al., 2015](https://arxiv.org/abs/1502.05477) 一类工作之上；
- pairwise logistic preference model 来自
  [Bradley & Terry, 1952](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091)；
- 幂的无偏组合估计使用经典
  [Hoeffding U-statistics](https://www.jstor.org/stable/2235637)；
- randomized truncation/debiasing 与
  [McLeish, 2011](https://arxiv.org/abs/1005.2228)、
  [Rhee & Glynn, 2015](https://web.stanford.edu/~glynn/papers/2015/RheeG15.pdf)
  的一般思想同源；
- moment objective 的统计语言建立在
  [Hansen, 1982](https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/)
  的 GMM 框架之上。

这些基础本身不是本项目的原创主张。本项目要检验的组合点是：把“一次局部 policy update
的 Fisher-inverse regret”连接到“可由逐 edge randomized repeated labels 识别的 reward
moment error”，再把该对象实现为固定 tangent、泄漏隔离、matched-KL downstream evaluation
和 fail-closed provenance 的完整实验协议。

这一定义说明了当前 method name `SRM+` 的作用，但仓库没有单独实现一个名为 `SRM` 的
baseline；“+”不是一条已完成 ablation hierarchy。是否构成文献意义上的新颖贡献，仍需在
论文阶段完成更广泛的 related-work review，不能仅由本工程文档断言。

## 11. 论文表述边界

1. 只有 `lambda=0` 的 population pseudoinverse 目标与 local regret 精确等价；`lambda>0`
   是 ridge empirical target，必须报告 sensitivity。
2. SRM+ 在当前实验中比较的是受限、可能 misspecified 的 reward class；是否在有限 train
   candidates 上观察到线性不可表示证据由 projection diagnostic 描述，不能由 capacity
   bottleneck 预先断言。该实验不宣称对所有 reward class、所有数据都优于 BT。
3. 若真实 annotator 偏离同质 BTL 或重复 labels 不条件 iid，`h` 的识别对象会改变。
4. 固定 `L` 个 labels 只能识别 logit 级数的前 `L` 项。CoVal 阶段必须称为
   **candidate-restricted truncated SRM+ robustness**，不能援引精确无偏定理。
5. pairwise accuracy 是诊断指标。只有 held-out policy geometry 与 matched-KL rollout 的
   预注册证据同时通过，才能支持主要机制结论。
