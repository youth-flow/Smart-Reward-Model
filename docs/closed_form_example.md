# 最小 population 闭式例：BT-RM 与 ProRM 为什么选择不同奖励

本文档对应论文标题：

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret**

这个四响应例子的唯一用途，是在排除有限样本噪声、优化失败、三条已选边之间的权重失衡和
图覆盖不足后，
证明 preference likelihood 与 downstream policy regret 可以对同一受限 reward class 给出不同
的最优解。

> [!IMPORTANT]
> 这是 **population ProRM 闭式反例**，不是 ProRM+ repeated-label 训练例。三条等权比较边并不
> 构成从参考策略独立采样的 natural pair distribution，不能直接代入 ProRM+ raw moment。

可执行的标准库实现与回归测试分别位于
[closed_form.py](../src/smart_reward/closed_form.py) 和
[test_closed_form.py](../tests/test_closed_form.py)。二者不调用优化器、不使用随机数；Python
`float` 在 CPython 中为 IEEE-754 binary64。

## 1. 命名与四个角色

本文档固定使用以下命名：

| 方法 | quotient-class 选择 | class 内代表选择 |
|---|---|---|
| BT-RM | 最小化 population BTL NLL | 固定辅助系数 `eta=0` |
| Aux-BT-RM | 最小化 population BTL NLL | 同时允许 `eta` 拟合比较数据 |
| ProRM | 最小化 population downstream local regret | 固定 `eta=0` |
| Aux-ProRM | 先用 ProRM 选择 quotient class，再在其中最小化 NLL | lexicographic 地选择 `eta` |

这里的 `Aux-ProRM` **不是** `ProRM+`。二者含义完全不同：

- **ProRM** 是 population prospective principle：用 downstream policy regret 训练 RM；
- **ProRM+** 是通过 natural Fisher stream、无偏 repeated-label signal 和 Fisher-GMM saddle
  实现 ProRM 的可训练估计器；
- **Aux-ProRM** 只是本闭式例中用于分离 quotient-class 选择与 class 内拟合的分析对照。

为避免符号冲突，辅助零空间系数一律写作 $\eta$；ProRM+ 的随机无偏 margin signal 一律写作
$H(e)$，满足

$$
\mathbb E[H(e)\mid e]=\Delta r^*(e).
$$

## 2. 策略、参考分布与真实奖励

只有一个 prompt 和四个 response $y_1,\ldots,y_4$。策略固定两组的总质量均为 $1/2$，
只调节组内赔率：

$$
\begin{aligned}
\pi_\theta(y_1)&=\tfrac12\sigma(2\theta_1), &
\pi_\theta(y_2)&=\tfrac12\sigma(-2\theta_1),\\
\pi_\theta(y_3)&=\tfrac12\sigma(2\theta_2), &
\pi_\theta(y_4)&=\tfrac12\sigma(-2\theta_2).
\end{aligned}
$$

参考点为 $\theta_0=(0,0)$，因此

$$
\pi_0=(1/4,1/4,1/4,1/4),\qquad \beta=16.
$$

真实奖励为

$$
r^*=(6,0,0,0).
$$

受限 reward class 由 quotient 坐标 $w$ 和辅助坐标 $\eta$ 参数化：

$$
\begin{aligned}
r_w&=(w/2,-w/2,w/2,-w/2),\\
\delta_\eta&=(\eta/2,\eta/2,-\eta/2,-\eta/2),\\
r_{w,\eta}&=r_w+\delta_\eta.
\end{aligned}
$$

标签严格由 BTL law 生成，所以 **observation model 是 well-specified 的**。但是受限 reward
class 是有意 misspecified 的：第一条组内边要求 $w=6$，第二条组内边要求 $w=0$，不可能
同时满足。不能把整个学习问题笼统称为 well-specified。

## 3. 局部几何与 reward quotient

在 $\theta_0$ 处，四个 policy score 为

$$
s_0(y_1)=(1,0),\quad s_0(y_2)=(-1,0),\quad
s_0(y_3)=(0,1),\quad s_0(y_4)=(0,-1).
$$

定义 reward-to-gradient operator

$$
A_0r=\mathbb E_{y\sim\pi_0}[s_0(y)r(y)],
$$

则

$$
A_0=\frac14
\begin{bmatrix}
1&-1&0&0\\
0&0&1&-1
\end{bmatrix},
\qquad
F_0=\mathbb E_{\pi_0}[s_0s_0^\top]=\frac12 I_2.
$$

policy-visible quotient 坐标可以写成

$$
g(r)=(r_1-r_2,r_3-r_4),\qquad A_0r=\tfrac14g(r).
$$

因此

$$
g(r^*)=(6,0),\qquad g(r_{w,\eta})=(w,w),\qquad A_0\delta_\eta=0.
$$

受限 RM 只能在直线 $\{(w,w):w\in\mathbb R\}$ 上选择 reward quotient class。改变
$\eta$ 只更换同一个 class 内的代表。

这里的 null invariance 比一般的一阶结论更强。因为两个组的策略质量始终为 $1/2$，对所有
$\theta$ 都有

$$
\mathbb E_{\pi_\theta}[\delta_\eta]=0.
$$

所以 $\eta$ 不改变精确 policy optimizer，而不只是局部 update。

## 4. 三条 likelihood 比较边

population NLL 使用三条等权、有向边：

| 边 | response pair | true margin | predicted margin |
|---|---|---:|---:|
| $e_{12}$ | $y_1-y_2$ | 6 | $w$ |
| $e_{34}$ | $y_3-y_4$ | 0 | $w$ |
| $e_{13}$ | $y_1-y_3$ | 6 | $\eta$ |

三条边构成覆盖全部 response 的连通树，且采样质量都为 $1/3$。因此现象不能归因于比较图
不连通或三条已选边之间的权重失衡；它们仍然不是 natural-pair samples。

记

$$
\ell(z,t)=\log(1+e^t)-\sigma(z)t,
$$

其中 $z$ 是 true BTL margin，$t$ 是 predicted margin。population objective 为

$$
\operatorname{NLL}(w,\eta)
=\frac13\bigl[\ell(6,w)+\ell(0,w)+\ell(6,\eta)\bigr].
$$

### 4.1 BT-RM 与 Aux-BT-RM

BT-RM 固定 $\eta=0$。一阶条件为

$$
2\sigma(w)-\sigma(6)-\tfrac12=0,
$$

所以

$$
w_{\mathrm{BT}}
=\operatorname{logit}\!\left(\frac{\sigma(6)+1/2}{2}\right)
=1.0920294543521607.
$$

NLL 关于 $w$ 与 $\eta$ 完全分离，所以 Aux-BT-RM 保持同一个 $w_{\mathrm{BT}}$，并令

$$
\eta_{\mathrm{Aux\text{-}BT}}=6.
$$

### 4.2 ProRM 与 Aux-ProRM

population ProRM loss 为

$$
\mathcal L_{\mathrm{ProRM}}(w)
=\frac1{2\beta}
\left\|A_0(r_w-r^*)\right\|_{F_0^\dagger}^2
=\frac{(w-6)^2+w^2}{16\beta}.
$$

它的唯一最优解是

$$
w_{\mathrm{ProRM}}=3.
$$

Aux-ProRM 是明确的 lexicographic construction：先固定 downstream-optimal $w=3$，再只在
该 equivalence class 内最小化 NLL，得到 $\eta=6$。它不是把 NLL 和 regret 任意加权成一个
新 scalar objective。

## 5. 精确 policy optimization 与结果表

对任意 $r_{w,\eta}$，令 $q(w)=\sigma(w/\beta)$。精确 KL-regularized optimizer 为

$$
\widehat\theta(w)=\left(\frac{w}{2\beta},\frac{w}{2\beta}\right),
\qquad
\widehat\pi(w)=
\left(\frac q2,\frac{1-q}{2},\frac q2,\frac{1-q}{2}\right).
$$

它与 $\eta$ 无关。真实最优解为

$$
\theta^*=(0.1875,0),\qquad
\pi^*=(0.2963333,0.2036667,0.25,0.25).
$$

精确 true-objective regret 是

$$
\operatorname{Reg}_{\mathrm{true}}(w)
=\frac\beta2\left[
D_{\mathrm{KL}}\!\left(\operatorname{Bern}(q(w))\middle\|
\operatorname{Bern}(\sigma(6/\beta))\right)
+D_{\mathrm{KL}}\!\left(\operatorname{Bern}(q(w))\middle\|
\operatorname{Bern}(1/2)\right)
\right].
$$

其导数为

$$
\frac{q(w)(1-q(w))}{2\beta}(2w-6),
$$

所以 $w=3$ 在这个特制策略族中也恰好是 exact-regret 最优解。这个 exact coincidence 是本例
的额外性质，不是所有 ProRM 问题的全局定理。

### 5.1 学到的策略

| 方法 | $(w,\eta)$ | $r_{w,\eta}$ | $\widehat\theta$ | $\widehat\pi$ |
|---|---:|---|---:|---|
| BT-RM | (1.092029, 0) | (0.546015, -0.546015, 0.546015, -0.546015) | (0.034126, 0.034126) | (0.258528, 0.241472, 0.258528, 0.241472) |
| Aux-BT-RM | (1.092029, 6) | (3.546015, 2.453985, -2.453985, -3.546015) | (0.034126, 0.034126) | (0.258528, 0.241472, 0.258528, 0.241472) |
| ProRM | (3, 0) | (1.5, -1.5, 1.5, -1.5) | (0.09375, 0.09375) | (0.273369, 0.226631, 0.273369, 0.226631) |
| Aux-ProRM | (3, 6) | (4.5, 1.5, -1.5, -4.5) | (0.09375, 0.09375) | (0.273369, 0.226631, 0.273369, 0.226631) |

### 5.2 Audited metrics

| 方法 | population NLL ↓ | local regret ↓ | exact true regret ↓ |
|---|---:|---:|---:|
| BT-RM | 0.6068419270 | 0.0987527469 | 0.0979508553 |
| Aux-BT-RM | **0.3815633415** | 0.0987527469 | 0.0979508553 |
| ProRM | 0.7659132511 | **0.0703125000** | **0.0695989247** |
| Aux-ProRM | 0.5406346656 | **0.0703125000** | **0.0695989247** |

排序发生反转：

- NLL：Aux-BT-RM < Aux-ProRM < BT-RM < ProRM；
- exact regret：ProRM = Aux-ProRM < BT-RM = Aux-BT-RM。

相对 BT-RM，ProRM 将 exact regret 降低 **28.945%**。加入 Aux 后，BT-RM 和 ProRM 的 NLL
分别降低 **37.123%** 和 **29.413%**，但 policy 与 regret 完全不变。

## 6. Local approximation sanity

定义相对误差

$$
\epsilon_\beta(w)=
\frac{|\operatorname{Reg}_{\mathrm{local}}(w)-
\operatorname{Reg}_{\mathrm{true}}(w)|}
{\operatorname{Reg}_{\mathrm{true}}(w)}.
$$

固定两个 learner 的 $w$，随着 $\beta$ 增大、policy step 变小，误差在预注册 grid 上严格
下降：

| $\beta$ | BT-RM $w=1.092029$ | ProRM $w=3$ |
|---:|---:|---:|
| 4 | 12.8728% | 16.3937% |
| 8 | 3.2621% | 4.0999% |
| 16 | 0.8187% | 1.0253% |
| 32 | 0.2049% | 0.2563% |
| 64 | 0.0512% | 0.0641% |

这项 sanity 只验证该闭式例符合局部近似预期，不把有限 $\beta$ 的 surrogate 宣称为精确恒等式。

## 7. Natural pair identity：什么条件下可以连接 ProRM+

ProRM+ 的 raw pair moment 需要合法的 natural pair law。若

$$
y,y'\overset{\mathrm{iid}}\sim\pi_0,
\qquad z_0=s_0(y)-s_0(y'),
$$

并利用 $\mathbb E_{\pi_0}[s_0]=0$，则

$$
\frac12\mathbb E[z_0(r(y)-r(y'))]
=\mathbb E_{\pi_0}[s_0(y)r(y)]
=A_0r,
$$

以及

$$
\frac12\mathbb E[z_0z_0^\top]=F_0.
$$

实现与测试显式枚举 $\pi_0\times\pi_0$ 的全部 16 个 ordered pairs，包括四个贡献为零的
self-pairs，并验证上述两个等式。

如果 repeated-label signal 满足 $\mathbb E[H(e)\mid e]=\Delta r^*(e)$，那么在这个
natural $Q_0$ 下才有

$$
\frac12\mathbb E[z_0(\Delta r_\phi-H)]
=A_0(r_\phi-r^*),
$$

从而把 ProRM population target 接到 ProRM+ Fisher-GMM estimator。

## 8. 为什么三条等权树边不能直接作为 ProRM+ 的 raw stream

连通性足以在 unrestricted reward model 中从 noiseless margins 恢复 reward up to gauge；它不
等价于 natural-pair moment identification。

对本 memo 的三条等权边，score differences 为

$$
z_{12}=(2,0),\qquad z_{34}=(0,2),\qquad z_{13}=(1,-1).
$$

因此对真实奖励

$$
\frac12\mathbb E_{e\sim Q_{\mathrm{tree}}}[z_0(e)\Delta r^*(e)]
=(3,-1),
$$

但

$$
A_0r^*=(3/2,0).
$$

两者不相等。对应的 pair Fisher 也不是 $F_0$：

$$
\frac12\mathbb E_{Q_{\mathrm{tree}}}[z_0z_0^\top]
=\begin{bmatrix}5/6&-1/6\\-1/6&5/6\end{bmatrix}
\ne\frac12I_2.
$$

更强的反例是，tree residual moment 对一般 $(w,\eta)$ 等于

$$
m_{\mathrm{tree}}(w,\eta)
=\left(\frac{2w+\eta-18}{6},\frac{2w-\eta+6}{6}\right).
$$

它在 $(w,\eta)=(3,12)$ 处为零，但此时

$$
A_0(r_{3,12}-r^*)=(-3/4,3/4)\ne0.
$$

所以把这三条边直接塞进 raw ProRM+ saddle 会产生虚假的零 moment。正确做法只能是：

1. 使用从 $\pi_0\times\pi_0$ 生成的 natural repeated-label pairs；或
2. 明确定义并证明具有充分 support 的 importance correction / reconstruction operator。

本闭式例没有实现第二条，因此文档绝不把 tree-edge NLL 数据冒充 ProRM+ training stream。

## 9. 这个例子证明什么、不证明什么

它证明：

- 即使使用 population BTL probabilities、三条边等权且比较图连通，BT likelihood 与
  downstream regret 仍可选择不同的 reward quotient class；
- policy-null 的 preference-visible 信息可以显著改善 NLL，而不改善 downstream policy；
- 在这个构造中，ProRM 的 local optimum 同时是受限 class 内的 exact-regret optimum；
- BT-RM 与 ProRM 的 pairwise NLL/downstream-regret 排序严格反转；Aux 变体保持相同
  downstream regret，因而在四行 regret 排序中形成 ties。

它不证明：

- ProRM 或 ProRM+ 在所有 reward class、数据分布和有限样本上都优于 BT-RM；
- 任意 connected comparison graph 都满足 natural pair identity；
- 三条 tree edges 上的 raw moment 是 ProRM+ 的无偏实现；
- 局部二阶 regret 在任意大小 policy update 上都是精确的。

运行闭式审计：

```bash
python -m pytest -q tests/test_closed_form.py
```
