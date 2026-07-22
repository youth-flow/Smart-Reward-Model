# ProRM/ProRM+ 理论规格：从全局 policy utility 到可训练 Fisher–GMM

论文标题固定为：

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret**

本文档是仓库的数学源规范。它固定 ProRM 与 ProRM+ 的含义、全部常数因子、population
identity、有限样本正则化以及工程近似的边界。实验设计见
[experiment_protocol.md](experiment_protocol.md)，三边四响应解析构造见
[closed_form_example.md](closed_form_example.md)。

## 0. 名称与结论层级

Prospective Reward Modeling 的出发点是：

> Reward model 应按它将诱导的未来 policy optimization 来训练，而不应只按它解释过去
> preference labels 的能力来训练。

两个名称严格对应两个数学层级：

| 名称 | 固定含义 |
|---|---|
| **ProRM** | 含不可观测目标 reward 的理想 population/local downstream-regret loss |
| **ProRM+** | 用 repeated labels 识别该目标 moment，并以 Fisher–GMM dual、ridge 与 PCG 实现的可训练方法 |

“+”表示从不可观测理想 target 到可训练 moment method 的识别与优化闭环。ProRM 不是一个
单独实现的 baseline。正式训练对照始终是 repeated-label BT-MLE 与 ProRM+。

整条推导是：

```text
global downstream utility
  -> local quadratic policy problem
  -> ideal ProRM regret in Fisher geometry
  -> natural-pair moment identity
  -> repeated-label unbiased margin h
  -> observable ProRM+ Fisher-GMM dual
  -> damped empirical objective + PCG
  -> held-out geometry + matched measured-KL rollout
```

最常用符号如下。除非另行说明，node expectation 包含
`x ~ rho, y ~ pi_0(.|x)`。

| 符号 | 含义 |
|---|---|
| `rho` | 固定的目标 prompt 分布 |
| `pi_0=pi_{theta_0}` | 生成候选并定义局部几何的 reference policy |
| `theta` | 下一次 policy optimization 真正允许改变的 tangent 坐标 |
| `s_0(x,y)` | `nabla_theta log pi_theta(y|x)` 在 `theta_0` 的 sequence score |
| `A_0 r` | reward 对局部 policy update 的一阶 moment |
| `F_0` | reference policy 在同一 tangent 坐标中的 Fisher |
| `r*`, `r_phi` | 目标/operational-oracle reward 与学习到的 reward model |
| `Q_0` | natural pair law `rho*pi_0*pi_0` |
| `z_0`, `Delta r_phi`, `h` | pair score difference、预测 margin、真实 margin 的无偏估计 |
| `m_phi` | `A_0(r_phi-r*)` 的可观测 repeated-label moment 表达 |
| `beta` | downstream KL regularization 系数，严格为正 |
| `lambda` | finite-sample Fisher ridge，严格为正 |

## 1. 全局 Prospective Reward Modeling 问题

### 1.1 先定义未来 policy utility

给定 reward `r`，下游 KL-regularized policy optimization 定义为

$$
J(\theta;r)
=\mathbb E_{x\sim\rho,y\sim\pi_\theta(\cdot|x)}[r(x,y)]
-\beta\mathbb E_{x\sim\rho}
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot|x)\,\Vert\,\pi_0(\cdot|x)
\right),
\qquad \beta>0.
$$

令

$$
\theta_r\in\arg\max_\theta J(\theta;r)
$$

表示把 `r` 交给下游 optimizer 后得到的 policy。reward model 的真实价值不能由它自己的
训练 loss 定义，而应由该 policy 在目标 reward `r*` 下的效用定义：

$$
U(r)=J(\theta_r;r^*).
$$

因此全局 prospective regret 是

$$
\boxed{
\operatorname{Reg}_{G}(r)
=J(\theta_{r^*};r^*)-J(\theta_r;r^*)
}.
$$

理想问题是 `min_{r in R} Reg_G(r)`。这个定义回答“哪个 reward 会诱导更正确的 policy”，
而 BT-MLE likelihood 回答“哪个 reward 更能解释 preference observations”。当 reward class
misspecified 时，两者没有理由选择同一个近似。

### 1.2 为什么不能直接训练全局目标

全局定义有两个不可直接实施的部分：

1. `theta_r` 是完整 policy optimization 的解映射，reward learning 与 policy learning 形成
   双层问题；
2. 评价需要目标 reward `r*`，训练时却只观察随机 pairwise preferences。

因此 `Reg_G` 是正确的效用定义，不是本项目直接反向传播的 loss。ProRM 通过局部 Taylor
近似解决第一个障碍；ProRM+ 再通过 repeated-label identification 解决第二个障碍。

### 1.3 研究所针对的维度错配

有限离散化后，把完整 reward 写成 `r in R^m`，reference policy 的可更新 tangent 维数为
`d`，受限 reward class 的有效维数为 `p`。目标设定是

$$
m\gg d\gg p.
$$

第一层压缩来自 policy：大量 reward directions 不改变当前 tangent 中的 update。第二层压缩
来自 reward class：只能在受限函数类中选择代表。这个关系是方法动机，不是实验证明；Phase 1
是否在有限 train candidates 上表现出线性不可表示性，必须由 projection diagnostic 报告。

## 2. 局部二次问题与理想 ProRM loss

### 2.1 Score、reward moment 与 Fisher

定义 reference-policy sequence score

$$
s_0(x,y)
=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0}\in\mathbb R^d,
$$

reward moment operator

$$
A_0r
:=\mathbb E_{x\sim\rho,y\sim\pi_0}[s_0(x,y)r(x,y)],
$$

以及 Fisher

$$
F_0
:=\mathbb E_{x\sim\rho,y\sim\pi_0}
[s_0(x,y)s_0(x,y)^\top].
$$

这些对象必须使用下游实际更新的同一坐标。Phase 1 只更新 fixed-A LoRA-B，所以候选 score、
Fisher、natural direction 与实际写回 policy 的 displacement 全部使用同一 LoRA-B layout、
scale、shape 与 hash。

### 2.2 Local policy optimization

令 `delta=theta-theta_0`。在 reference policy 附近，对期望 reward 作一阶展开、对 forward KL
作二阶展开：

$$
J(\theta_0+\delta;r)
\approx
C(r)+\delta^\top A_0r
-\frac\beta2\delta^\top F_0\delta.
$$

记 `g_r=A_0r`。在 identifiable tangent space 上取最小范数解：

$$
\delta_r
=\beta^{-1}F_0^\dagger g_r.
$$

若 `a` 位于 `Null(F_0)`，则 `a^T s_0=0` 几乎处处。只要 reward 二阶可积，`a^Tg_r=0`，
所以 `g_r` 位于 `Range(F_0)`，伪逆表达良定义。

### 2.3 ProRM 的精确局部 regret

用目标 reward 的局部 objective

$$
G^*(\delta)=g_*^\top\delta-\frac\beta2\delta^\top F_0\delta
$$

评价 learned reward 诱导的 `delta_phi`。理想 ProRM loss 定义为

$$
\mathcal L_{\mathrm{ProRM}}(\phi)
:=G^*(\delta^*)-G^*(\delta_\phi).
$$

对二次函数配方：

$$
G^*(\delta^*)-G^*(\delta_\phi)
=\frac\beta2(\delta_\phi-\delta^*)^\top
F_0(\delta_\phi-\delta^*).
$$

再使用
`delta_phi-delta*=beta^{-1}F_0^dagger(A_0r_phi-A_0r*)`，得到

$$
\boxed{
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}
\left\|A_0(r_\phi-r^*)\right\|_{F_0^\dagger}^{2}
=\frac1{2\beta}
(g_{r_\phi}-g_*)^\top F_0^\dagger(g_{r_\phi}-g_*)
}.
$$

这里 `||u||^2_{F_0^dagger}:=u^T F_0^dagger u`。这是 population、无阻尼、局部二阶问题中的
精确 identity；它不是任意大 policy update 的全局保证。

### 2.4 Projection geometry 与 reward 等价类

在有限表示中，令 `A_theta` 的列为每个 `(x,y)` 的 score，
`D_0=diag(rho(x)pi_0(y|x))`，`B_0=A_theta D_0^(1/2)`。则

$$
F_0=B_0B_0^\top,
$$

且

$$
\boxed{
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}
\left\|
P_{\operatorname{row}(B_0)}
D_0^{1/2}(r_\phi-r^*)
\right\|_2^2
}.
$$

因此：

- prompt 内统一加常数不会改变 update；
- 位于 score 零空间的 reward error 不会被惩罚；
- pointwise MSE、BT-MLE NLL 与 policy regret 是不同几何中的投影；
- 在 misspecified reward class 中，BT-MLE optimum 不保证等于 ProRM optimum；
- 若 `A_0r_1=A_0r_2`，两个 rewards 在当前局部 policy problem 中属于同一等价类。

三边四响应 [closed-form example](closed_form_example.md) 给出 BT-MLE 与 population ProRM 的解析排序
反转。它只证明理想 population objectives 可以选出不同 reward；它不使用 randomized `h`，
因此不能单独证明 ProRM+ identification。后者必须在 natural `Q_0` 下由下一节的 identity 建立。

## 3. Natural-pair representation

### 3.1 固定 pair law 与方向

不引入人为 edge reweighting。定义

$$
Q_0(dx,dy,dy')
=\rho(dx)\pi_0(dy\mid x)\pi_0(dy'\mid x).
$$

给定 `e=(x,y,y')`，定义

$$
z_0(e)=s_0(x,y)-s_0(x,y'),
\qquad
t_r(e)=\Delta r(e)=r(x,y)-r(x,y').
$$

方向始终为 `left-right`：`a=1` 表示 left/`y` 获胜。交换 edge 时必须同时翻转 `z_0`、
`Delta r`、`h`，并把 `left_wins` 变为 `N-left_wins`。

### 3.2 两个精确 pair identities

Score identity 给出

$$
\mathbb E_{y\sim\pi_0(\cdot|x)}[s_0(x,y)\mid x]=0.
$$

因为 `y,y'` 条件独立，展开可得

$$
\mathbb E[z_0t_r\mid x]
=2\mathbb E[s_0r\mid x],
$$

所以

$$
\boxed{A_0r=\frac12\mathbb E_{e\sim Q_0}[z_0(e)t_r(e)]}.
$$

同理，

$$
\boxed{F_0=\frac12\mathbb E_{e\sim Q_0}[z_0(e)z_0(e)^\top]}.
$$

Population 中 node 与 pair 表示相同。工程上固定为：

- 全部 on-policy candidate nodes 估计 Fisher，降低方差；
- canonical labeled edge 估计 reward-error moment；
- 不用 edge endpoint 频率替代原 on-policy node Fisher；
- 两个 finite-sample estimators 不要求数值相等。

Phase 1 中，`P` 个 prompts、每 prompt `M=4` 个 candidates、policy dimension `d`、reward
feature dimension `H` 对应：

```text
S:          (P*M, d)   # all on-policy nodes
Z:          (P, d)     # canonical candidate 0 - candidate 1
left/right: (P, H)     # frozen reward features
Delta r,h:  (P,)
F_hat = S.T @ S / (P*M)
m_hat = Z.T @ (Delta r-h) / (2*P)
```

## 4. Repeated labels identify the target margin

### 4.1 单标签不可能逐 edge 无偏恢复 logit

单位温度 BTL 模型为

$$
a\mid e\sim\operatorname{Bernoulli}(p^*(e)),
\qquad
p^*(e)=\sigma(\Delta r^*(e)).
$$

对仅依赖一个 Bernoulli label 的任意统计量 `H(a)`，

$$
\mathbb E[H(a)\mid e]
=(1-p^*)H(0)+p^*H(1),
$$

它关于 `p*` 必为仿射函数，不可能在区间上等于非线性的 `logit(p*)`。这个命题只排除
“逐 edge、distribution-free、单标签”的无偏 target；它不否定跨 edge 参数共享的 MLE。

### 4.2 Randomized U-statistic

利用级数

$$
\operatorname{logit}(p)
=\sum_{k=1}^{\infty}\frac{p^k-(1-p)^k}{k}.
$$

对同一 edge 获取条件 iid labels `a_1,...,a_N`。记 `S_N=sum_j a_j`，定义

$$
U^+_{k,N}=\frac{\binom{S_N}{k}}{\binom Nk},
\qquad
U^-_{k,N}=\frac{\binom{N-S_N}{k}}{\binom Nk}.
$$

令 `N` 独立于 edge 与 labels，生存概率 `q_k=P(N>=k)>0`。定义

$$
h
=\sum_{k=1}^{N}
\frac{U^+_{k,N}-U^-_{k,N}}{kq_k}.
$$

条件于 `N>=k`，两个 U-statistics 分别无偏估计 `p^k` 与 `(1-p)^k`；`1/q_k` 校正第
`k` 项被计算的生存概率。因此

$$
\boxed{
\mathbb E[h\mid e]
=\operatorname{logit}(p^*(e))
=\Delta r^*(e)
}.
$$

实现只使用 `(S_N,N)` 的组合计数，不枚举 label 子集。

### 4.3 主实验的随机截断常数

主实验固定

$$
P(N=n)=(1-\gamma)\gamma^{n-1},
\qquad
q_k=\gamma^{k-1},
\qquad
\gamma=0.9.
$$

因此 `E[N]=1/(1-gamma)=10`。oracle transform 保证
`p* in [0.25,0.75]`，并且 `gamma > max(p*,1-p*)`，即 `0.9>0.75`；这满足本实验采用的
有限二阶矩充分条件。

不得硬截断、clip 大 `N`、按 `N` 重采样或静默丢弃。memory guard 只能使 run fail closed，
不能改变 estimator。

随机 `N` 只决定构造 `h` 的成本：

- ProRM+ 中每个 edge 对 moment 恰好贡献一次；
- 不得按 `N` 再给 ProRM+ edge 加权；
- BT-MLE 使用全部原始 Bernoulli labels，等价于按 `N` 累计 likelihood。

### 4.4 固定重复数只能得到截断 target

若每个 edge 固定收集 `L` 个 labels，`S=sum_j a_j`，则

$$
h_L
=\sum_{k=1}^{L}\frac1k
\left[
\frac{\binom Sk}{\binom Lk}
-\frac{\binom{L-S}k}{\binom Lk}
\right].
$$

其期望只等于 logit 级数的前 `L` 项。若 `p* in [epsilon,1-epsilon]`，

$$
\left|
\mathbb E[h_L\mid e]-\Delta r^*(e)
\right|
\le
\frac{2(1-\epsilon)^{L+1}}{\epsilon(L+1)}.
$$

所以固定 `L` 的人类数据实验必须称为 **candidate-restricted truncated ProRM+ robustness**，
不能援引精确无偏 identity。

## 5. Observable ProRM+ Fisher–GMM objective

### 5.1 Moment identification

定义预测 margin `Delta r_phi(e)` 以及

$$
m_\phi
:=\frac12\mathbb E_{e,h}
[z_0(e)(\Delta r_\phi(e)-h(e))].
$$

利用 repeated-label identity 与 natural-pair identity：

$$
\begin{aligned}
m_\phi
&=\frac12\mathbb E_{Q_0}
[z_0(\Delta r_\phi-\Delta r^*)]\\
&=A_0(r_\phi-r^*).
\end{aligned}
$$

因此既不需要恢复完整 `r*`，也不需要分别估计 `A_0r_phi` 与 `A_0r*`；直接估计二者之差。

### 5.2 Population equivalence theorem

ProRM loss 可写为

$$
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}m_\phi^\top F_0^\dagger m_\phi.
$$

对 `m in Range(F_0)`，Fenchel identity 为

$$
\frac12m^\top F_0^\dagger m
=\max_v\left(v^\top m-\frac12v^\top F_0v\right).
$$

所以 ProRM+ 的 population objective 是

$$
\boxed{
\min_\phi\max_v
\frac1\beta
\left[
v^\top m_\phi
-\frac12v^\top F_0v
\right]
}.
$$

展开两个 expectations：

$$
\min_\phi\max_v
\left\{
\frac1{2\beta}
\mathbb E_{e,h}[(z_0(e)^\top v)(\Delta r_\phi(e)-h(e))]
-\frac1{2\beta}
\mathbb E_{x,y}[(s_0(x,y)^\top v)^2]
\right\}.
$$

在 natural `Q_0`、条件 iid repeated labels、reward/score 可积、局部二阶近似和无阻尼 Fisher
条件下：

$$
\boxed{
\max_v\mathcal J_{\mathrm{ProRM+}}(\phi,v)
=\mathcal L_{\mathrm{ProRM}}(\phi)
=\widetilde{\operatorname{Reg}}(r_\phi)
}.
$$

这是本项目的核心 identity。最小范数 dual witness 为 `v=F_0^dagger m_phi`。

### 5.3 不要混淆 dual witness 与 policy direction

- `u_r=(F+lambda*I)^-1 g_r` 是未除以 `beta` 的 damped natural direction；
- `delta_r=u_r/beta` 才是局部 policy displacement；
- `v=(F+lambda*I)^-1 m` 是 reward-error dual witness；
- population、`lambda=0` 时 `v=beta*(delta_phi-delta*)`。

## 6. Finite-sample ridge ProRM+

令 `S in R^(n_F x d)` 为 node scores，`Z in R^(n_E x d)` 为 edge score differences。固定

$$
\widehat F_0=\frac1{n_F}S^\top S,
\qquad
\widehat m_\phi
=\frac1{2n_E}Z^\top(\Delta r_\phi-h).
$$

由于通常 `d>n_F`，经验 Fisher 必然秩亏。实际训练的 ProRM+ objective 必须显式写成

$$
\boxed{
\min_\phi\max_v
\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F_0+\lambda I)v
\right]
},
$$

其中

$$
\lambda
=c\,\operatorname{mean}(\operatorname{diag}\widehat F_0)>0.
$$

内层唯一解与报告值为

$$
v_\phi^*=(\widehat F_0+\lambda I)^{-1}\widehat m_\phi,
$$

$$
\boxed{
\widehat L_{\mathrm{ProRM+},\lambda}(\phi)
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F_0+\lambda I)^{-1}\widehat m_\phi
}.
$$

必须区分三个层级：

| 层级 | Fisher | 可以声称什么 |
|---|---|---|
| Population theorem | `F_0^dagger`, `lambda=0` | ProRM+ inner optimum 与 local ProRM regret 精确相等 |
| Finite-sample experiment | `F_hat+lambda*I`, `lambda>0` | Ridge empirical surrogate |
| Sensitivity | `c in {1e-4,1e-3,1e-2}` | 检查结论是否依赖阻尼尺度 |

不得把 empirical ridge objective 称为 population identity 的“精确实现”。

### 6.1 Ridge 的坐标依赖性

`lambda*I` 不具任意 tangent reparameterization invariance。
`lambda=c*mean(diag(F_hat))` 只消除统一全局尺度，不能抵消各坐标的非等比例变化。因此
fixed-A seed/state、LoRA alpha、B layout、shape、scale、flatten order 与 hash 都是 empirical
objective 的科学定义，而不只是 provenance。训练与 rollout 必须复用同一坐标。

### 6.2 Train 与 held-out 不是同一个 estimator

| 用途 | Moment/Fisher | Damping |
|---|---|---|
| theorem | population `A_0(r_phi-r*)`, `F_0` | pseudoinverse，`lambda=0` |
| train | canonical-edge `m_hat`, train node `F_hat` | 由 train Fisher 解析 |
| held-out | prompt covariance `g_hat_error`, held-out node Fisher | 每个 split 独立解析 |

Held-out covariance 直接估计 `g_error`，所以没有 pair identity 中的 `1/2`；这不是常数冲突。

## 7. PCG、reported value 与 envelope gradient

### 7.1 Matrix-free dual solve

每个 outer optimizer step 先求解

$$
(\widehat F_0+\lambda I)v=\widehat m_\phi.
$$

PCG 只调用

$$
u\longmapsto\frac1{n_F}S^\top(Su)+\lambda u,
$$

无需形成 `d x d` Fisher。主配置固定 relative tolerance `1e-5`、最多 100 iterations，并使用
damped Fisher diagonal 的 Jacobi preconditioner。最终 evidence 保存真实 `rhs-Ax` residual；
未收敛是 fail-closed condition，不是 warning。

### 7.2 三个 scalar 必须分开

解得 `v` 后，报告的 ridge quadratic 是

$$
\widehat L_{\mathrm{reported}}
=\frac1{2\beta}\widehat m_\phi^\top v.
$$

其精确 envelope gradient 为

$$
\nabla_\phi\widehat L_{\mathrm{reported}}
=\frac1{2\beta n_E}
\sum_i(z_i^\top v)\nabla_\phi\Delta r_{\phi,i}.
$$

若在 autograd 中 detach `v`，必须使用 mean-reduced surrogate

$$
\widehat L_{\mathrm{env}}
=\frac1{n_E}\sum_i
\frac{z_i^\top\operatorname{stopgrad}(v)}{2\beta}
(\Delta r_{\phi,i}-h_i).
$$

同一 full batch、精确 solve 下：

$$
\widehat L_{\mathrm{env}}
=\frac1\beta v^\top\widehat m_\phi
=2\widehat L_{\mathrm{reported}}.
$$

它的数值不是论文 objective；它只产生正确 gradient。完整 saddle diagnostic 为

$$
\widehat L_{\mathrm{saddle}}(\phi,v)
=\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F_0+\lambda I)v
\right].
$$

最优 `v` 时，saddle value 等于 reported value；PCG 近似时两者可能不同。仓库分别记录
reported quadratic、saddle diagnostic 和 training surrogate，禁止混用。

### 7.3 外层更新顺序

每次 optimizer update 固定执行：

```text
all training-edge margins
  -> one full moment m_hat
  -> warm-started PCG solve
  -> detach v
  -> microbatch accumulation of one full-data envelope gradient
  -> exactly one optimizer step
  -> recompute margins, moment and dual
```

Microbatch 只改变内存占用，不创建 batch-local moment/Fisher。一个 dual direction 不得跨多个
outer steps stale reuse。主实验固定 720 次 fresh-dual updates，validation 不选择 checkpoint。

## 8. Held-out geometry and downstream evaluation

### 8.1 Prompt-centered held-out moment

每个 held-out prompt 有 `M` 个 candidates。为有限样本中精确消除 prompt reward gauge，使用

$$
\widehat g_r
=\frac1{P(M-1)}
\sum_{i=1}^{P}\sum_{j=1}^{M}
(s_{ij}-\bar s_i)(r_{ij}-\bar r_i).
$$

Covariance 的无偏分母是 `P(M-1)`；held-out node Fisher 仍用 `PM`。每个 split 从自身 Fisher
独立解析 absolute damping。主要 geometry metrics 是：

1. reward moment error 的 held-out ridge local-regret proxy；
2. predicted 与 target damped natural directions 的 undamped-Fisher squared error；
3. 两个 directions 的 Fisher cosine。

若任一 direction 的 Fisher norm 为零，cosine 未定义，正式结果必须 fail closed。

### 8.2 Prediction metrics 只是描述指标

Held-out preference diagnostics 报告：

- BTL negative log-likelihood；
- pairwise ordering accuracy；
- predicted BTL probability 与 operational-oracle BTL probability 的 mean absolute error。

这些指标回答 preference fit，不是 ProRM+ 的 primary success gate。它们的作用是显示
preference geometry 与 policy geometry 是否出现预期分离，不能替代 downstream evidence。

### 8.3 Matched measured-KL policy optimization

分别由 BT-MLE 与 ProRM+ reward moments 构造 natural direction。Fisher quadratic

$$
\alpha_{\mathrm{init}}
=\sqrt{\frac{2\kappa}{d^\top Fd}}
$$

只提供 line-search 初值。每个方法必须独立测量更新后 policy 对 reference policy 的
sequence-level forward KL，并匹配到

$$
\kappa=0.01\quad\text{with relative tolerance }0.05.
$$

最终接受依据是 measured KL，不是二阶预测。随后用 common-random candidate indices 比较
zero-B、BT-MLE update 与 ProRM+ update 的 transformed-oracle reward improvement。

## 9. Assumptions, failure modes and protections

| Assumption | Violation | Current protection |
|---|---|---|
| Local reward/KL Taylor model is adequate | ProRM identity cannot be extrapolated to a large update | One update; measured-KL budget `0.01` |
| Candidates are sampled from `pi_0` | Score identity and Fisher distribution are wrong | Same FP32 instance; exact tokens; no filtering |
| Tangent coordinates match | Objective weights unusable policy directions | Fixed-A, zero-B, identical layout/scale/hash |
| Edge law is natural `Q_0` | Pair moment no longer identifies `A_0r` without correction | Canonical `0-1` endpoints are independent base samples |
| Repeated labels are conditionally iid BTL | `h` no longer identifies one target margin | Controlled Phase 1 oracle; human data only robustness |
| `N` is independent with correct survival law | Randomized estimator becomes biased | Named RNG streams; no clipping or silent resampling |
| Rewards and scores have sufficient moments | Fisher/moment variance can diverge | Bounded oracle transform; probability floor |
| Train and evaluation targets are isolated | Target leakage invalidates comparison | Train tensor schema cannot contain true rewards |
| PCG and measured-KL search converge | Direction or step size is undefined | Fail closed; failed seed cannot be discarded |
| Measured KL is locally bracketable on the positive ray | Bisection cannot locate target | Accept only measured KL within tolerance |

If policy optimization moves materially away from `pi_0`, candidates, scores, Fisher and repeated-label
moments must be regenerated around the new reference. The one-reference theorem does not justify reusing
old geometry indefinitely.

## 10. Mathematical objects and compatibility code paths

Public terminology is ProRM/ProRM+. The repository and Python namespace remain `Smart-Reward-Model` and
`smart_reward`; several internal identifiers retain historical names for artifact compatibility.

| Object | Current implementation |
|---|---|
| randomized `h` | `src/smart_reward/annotations.py` |
| `m_hat`, reported quadratic, envelope weights | `src/smart_reward/objective.py` |
| matrix-free Fisher and PCG | `src/smart_reward/linear.py`, `pcg.py` |
| fixed-A LoRA score/layout | `src/smart_reward/hf.py`, `scores.py` |
| BT-MLE / ProRM+ fixed-step trainers | `src/smart_reward/training.py` |
| held-out policy geometry | `src/smart_reward/metrics.py` |
| natural directions and measured KL | `src/smart_reward/rollout.py`, `policy_update.py` |
| real-model orchestration | `src/smart_reward/phase1.py`, `phase1_rollout.py` |

The public CLI is `prorm`; the historical `smart-reward` executable remains a compatibility alias during
migration.

## 11. Foundations and contribution boundary

The project composes established tools:

- Fisher/natural policy-gradient geometry builds on
  [Kakade, 2001](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
  and trust-region policy optimization
  [Schulman et al., 2015](https://arxiv.org/abs/1502.05477);
- pairwise logistic preferences use the
  [Bradley–Terry model](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091);
- unbiased estimates of powers use classical
  [Hoeffding U-statistics](https://www.jstor.org/stable/2235637);
- randomized truncation follows the general debiasing ideas of
  [McLeish, 2011](https://arxiv.org/abs/1005.2228) and
  [Rhee & Glynn, 2015](https://web.stanford.edu/~glynn/papers/2015/RheeG15.pdf);
- the moment-estimation language builds on
  [Hansen, 1982](https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/).

These ingredients are not individually claimed as new. The proposed combination is:

1. define reward-model quality prospectively through downstream policy regret;
2. reduce the global bilevel objective to a local Fisher-inverse ProRM target;
3. identify its reward-error moment from natural pairs using randomized repeated labels;
4. train the resulting ProRM+ objective with a matrix-free Fisher–GMM dual;
5. test the mechanism under fixed tangent coordinates, leakage isolation and matched measured-KL policy
   optimization.

## 12. Paper claim boundary

1. Exact equality holds for the population, local, undamped pseudoinverse target. Finite-sample ridge is a
   regularized surrogate and requires damping sensitivity.
2. Phase 1 uses a restricted reward class and an operational oracle. It does not establish human utility.
3. A capacity bottleneck does not prove misspecification. The train-only projection residual is descriptive.
4. If real annotators violate homogeneous conditionally iid BTL, `h` identifies a different object.
5. Fixed-`L` human data supports only truncated ProRM+ robustness.
6. The closed-form example proves a population ProRM/BT-MLE ordering reversal, not the ProRM+ repeated-label
   theorem or an empirical effect.
7. Preference NLL, accuracy and probability MAE are diagnostics. A positive mechanism claim requires both
   held-out policy geometry and matched measured-KL rollout evidence.
8. Until the pinned HPC4 runs and preregistered aggregation finish, the repository contains no empirical
   claim that ProRM+ outperforms BT-MLE.
