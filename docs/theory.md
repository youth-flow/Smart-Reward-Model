# SRM+：从局部 RLHF regret 到可训练 Fisher-GMM

本文档是实现的唯一数学规格。符号、常数因子和“精确/近似”的边界均以此处为准。

## 1. 局部决策问题

固定 prompt 分布 `rho` 和 reference policy `pi_0=pi_{theta_0}`。`theta` 只包含下一步
真实允许更新的 policy 坐标。令

\[
s_0(x,y)=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0},\quad
F_0=\mathbb E_{\rho,\pi_0}[s_0s_0^\top],\quad
g_r=\mathbb E_{\rho,\pi_0}[s_0r(x,y)].
\]

在 `theta_0` 附近，真实 reward `r*` 的 KL-regularized objective 为

\[
G^*(\delta)=g_*^\top\delta-\frac{\beta}{2}\delta^\top F_0\delta.
\]

在 identifiable tangent space 上，

\[
\delta^*=\beta^{-1}F_0^\dagger g_*,\qquad
\delta_r=\beta^{-1}F_0^\dagger g_r.
\]

所以用 `r` 产生更新的局部 regret 是

\[
\widetilde{\mathrm{Reg}}(r)
=G^*(\delta^*)-G^*(\delta_r)
=\frac1{2\beta}(g_r-g_*)^\top F_0^\dagger(g_r-g_*).
\]

离散表示中，若 `A_theta` 的列是 score，`D_0=diag(rho*pi_0)`，则
`A_0=A_theta D_0`、`F_0=A_theta D_0 A_theta^T`。令
`B=A_theta D_0^(1/2)`，上式等于

\[
\frac1{2\beta}\left\|
P_{\mathrm{row}(B)}D_0^{1/2}(r-r^*)
\right\|_2^2.
\]

这正是“概率加权 reward error 在 policy tangent 可见子空间中的投影长度”。它忽略
不会改变该局部 policy class 的 reward error，而不是逐点拟合整个 reward vector。

## 2. Pair 表示

对同一 `x` 独立采样 `y,y'~pi_0`，定义

\[
e=(x,y,y'),\quad z(e)=s_0(x,y)-s_0(x,y'),\quad
t_r(e)=r(x,y)-r(x,y').
\]

由 score identity `E[s_0|x]=0`，有精确恒等式

\[
g_r=\frac12\mathbb E[z t_r],\qquad
F_0=\frac12\mathbb E[zz^\top].
\]

实现中用 node score 矩阵 `S` 估计 `F_0`，用 edge difference `Z` 估计 reward
error moment。两种表达服务于不同的低方差估计器，不应把 Fisher 改成由主动选择后的
edge endpoint 频率决定。

## 3. 为什么需要同一 edge 的重复标签

假设 Bradley–Terry–Luce (BTL) 模型

\[
a\mid e\sim\mathrm{Bernoulli}(p^*(e)),\quad
p^*(e)=\sigma(t^*(e)),\quad t^*(e)=\Delta r^*(e).
\]

单个 Bernoulli 标签的任意逐记录统计量 `H(e,a)` 的条件均值关于 `p` 必为仿射，
不可能在一个区间上等于 `logit(p)`。因此，在以下限定同时成立时，单标签不能给出
精确 SRM+：逐 edge、distribution-free、不估条件概率、不引入伪 logit。这个结论
不否定带参数模型或跨样本平滑的单标签学习。

令随机重复数 `N` 独立于 edge 和标签，`q_k=P(N>=k)>0`；给定 edge，
`a_1,...,a_N` 条件 iid。记 `S_N=sum_j a_j`，并定义

\[
U^+_{k,N}=\frac{\binom{S_N}{k}}{\binom Nk},\qquad
U^-_{k,N}=\frac{\binom{N-S_N}{k}}{\binom Nk},
\]

\[
h=\sum_{k=1}^{N}\frac{U^+_{k,N}-U^-_{k,N}}{kq_k}.
\]

因为条件于 `N>=k` 时两个 U-statistic 的期望分别为 `p^k` 和 `(1-p)^k`，

\[
\mathbb E[h\mid e]
=\sum_{k=1}^\infty\frac{p^{*k}-(1-p^*)^k}{k}
=\log\frac{p^*}{1-p^*}=t^*(e).
\]

主实验固定 `N~Geometric(1-gamma)`（支撑从 1 开始），故
`q_k=gamma^(k-1)`。`gamma=0.9`，并通过 oracle transform 保证
`p* in [0.25,0.75]`；这满足有限二阶矩所需的尾部条件。不得硬截断 `N`，否则会
破坏精确无偏性。

## 4. Fisher-GMM 目标和对偶

定义

\[
m_\phi=\frac12\mathbb E[z(t_\phi-h)].
\]

则 `m_phi=A_0(r_phi-r*)`，精确 population loss 为

\[
\mathcal L_{\mathrm{SRM+}}(\phi)
=\frac1{2\beta}m_\phi^\top F_0^\dagger m_\phi
=\widetilde{\mathrm{Reg}}(r_\phi).
\]

其 Fenchel 对偶为

\[
\min_\phi\max_v\;
\frac1{2\beta}\mathbb E[(z^\top v)(t_\phi-h)]
-\frac1{2\beta}\mathbb E[(s^\top v)^2].
\]

经验量固定为

\[
\widehat m_\phi=\frac1{2n_E}Z^\top(t_\phi-h),\qquad
\widehat F=\frac1{n_F}S^\top S.
\]

由于 `d>n_F`，经验 Fisher 必然秩亏。工程目标明确采用

\[
\widehat L_\lambda(\phi)
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F+\lambda I)^{-1}\widehat m_\phi.
\]

PCG 解 `v=(F_hat+lambda I)^-1 m_hat`，loss 为 `m_hat^T v/(2 beta)`，
envelope gradient 为

\[
\nabla_\phi\widehat L_\lambda
=\frac1{2\beta n_E}\sum_i(z_i^\top v)\nabla_\phi t_\phi(e_i).
\]

`v` 在外层反传中必须 detach。每次 PCG 后只做一个 optimizer step，随后重新计算
全量 margin、moment 和 `v`；不固定 stale `v` 训练一整个 epoch，也不动态归一化
edge weight。

## 5. 定理的适用条件

- `pi_theta` 在 `theta_0` 二阶可微、support 固定，可交换微分与期望；更新在局部
  trust region 内。
- score 与 reward 二阶可积，`rho` 固定。
- BTL link 及温度固定。同一 edge 的标签条件 iid；`N` 与 edge/标签独立，生存概率
  已知，偏好概率远离 0 和 1。
- candidate 的实际生成分布与计算 score 的 `pi_0` 完全相同。
- Fisher 坐标与真实下游 policy update 坐标完全相同。
- 定理描述一次局部更新；policy 离开展开点后必须重建 candidate、score 和 Fisher。

## 6. 不得越界的论文表述

1. `lambda=0` 与 population pseudoinverse 才有精确 regret 等价；`lambda>0` 是
   regularized empirical target，且需报告 sensitivity。
2. 固定 `L` 个人类标签只识别 logit 级数的前 `L` 项。CoVal 实验称为
   **candidate-restricted truncated SRM+ robustness experiment**，不援引精确无偏定理。
3. 真实标签若偏离同质 BTL、annotator 不是条件 iid，`h` 的识别对象会改变。
4. SRM+ 的单 edge 权重为 1，不能按随机 `N` 加权；BT-MLE 基线才按全部原始标签
   等价加权。
5. 主实验用于验证 policy-aware misspecification。首个模型必须限制 reward class，
   使用 frozen backbone feature + linear scalar head；高容量 LoRA RM 是后续 scale-up，
   不能取代可识别主实验。
