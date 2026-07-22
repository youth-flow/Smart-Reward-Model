# 固定实验协议

## 目标

先验证一个单一因果链：在 reward class misspecified 时，SRM+ 比 repeated-label
BT-MLE 更准确地恢复真实局部 policy update。任何模型扩容、主动采样或全局 RLHF 都在
这条链通过后再做。

## Phase 0：纯数值内核

门槛全部通过才允许下载模型或占用 GPU：

1. `h` 的 Monte Carlo 均值匹配 `logit(p)`，并检查二阶矩；
2. PCG 相对残差 `<1e-5`，与小矩阵 direct solve 一致；
3. primal/dual value 相对误差 `<1e-4`；
4. envelope gradient 通过 float64 finite difference；
5. reward 加 prompt-level 常数后 metric 不变；
6. node/pair 两种 `A_0 r`、`F_0` 估计在模拟中一致。

## Phase 1：controlled on-policy 主实验

### 数据生成

1. 从固定 revision 的 MultiPref 只取 prompt，去重后先做
   `1536/256/256` prompt-level train/validation/test 划分。
2. 对每个 prompt 从固定 revision 的 `pi_0` 独立采样 4 个 response。保留重复文本；
   禁止 beam search、top-k/top-p 截断、质量过滤和 oracle 筛选。
3. `min_new_tokens=0`，不使用改变分布的 logits processor。保存原始 token IDs、
   response mask、EOS/达到最大长度的状态、tokenizer/chat-template revision。score 计算
   必须复用完全相同的 token 序列。
4. candidate 1 与 2 形成唯一 edge；4 个 candidate 全部进入 node Fisher。
   edge 始终使用 canonical `(left_id,right_id)` 方向；给标注者随机交换展示顺序后，
   label 必须映射回 canonical left 胜为 1。禁止改存为 chosen/rejected。

### Policy geometry

使用 Qwen2.5-0.5B reference policy 最后四层 `q_proj/v_proj` 的 rank-4 fixed-A LoRA：

- LoRA-A 初始化一次后冻结；LoRA-B 为零且是唯一 tangent 参数；
- adapter 前后 logits no-op 误差需达门槛；
- score 为 response token（含终止 EOS）的 sequence log-probability 之和对 LoRA-B 的
  梯度，不做长度归一化；
- 保存 A 权重、参数名、shape、flatten offset、score dtype 与 SHA256；
- controlled 主实验的 candidate 生成与 score 提取使用同一个 FP32 policy 实例；
  `S,Z` 存 float32。禁止 BF16 采样后改用 FP32 policy 算 score；
- Monte Carlo 检查每个 prompt 的 score 均值和 node/pair identity。

### Oracle 与重复标签

只用 train node 的 raw oracle 分数拟合 `b=median(R)` 与
`tau=max(1.4826*median(|R-b|),1e-6)`，冻结后应用所有 split：

\[
r^*(x,y)=\frac{\log3}{2}\tanh((R_{oracle}(x,y)-b)/\tau).
\]

于是 `|Delta r*|<=log 3`、`p*=sigmoid(Delta r*) in [0.25,0.75]`。每条 edge
独立采 `N~Geom(0.1)`，再采 `N` 个 BTL 标签并构造 `h`。训练文件保存
`edge_id,N,left_wins,raw_labels,h`；`true_margin` 单独保存在 evaluation-only 文件，
训练 schema 对该字段硬报错。

### Reward learners

主结果只比较三条：

- **BT-MLE**：frozen policy backbone feature + linear head，使用全部重复 Bernoulli 标签；
- **SRM+**：完全相同 feature、head 初始化和 batch，edge 等权，PCG-dual envelope 更新；
- **Oracle**：只给评估上界，不训练。

受控 upper bound 可直接回归 `true_margin`，但不得出现在真实数据结论里。主实验确认
misspecification 信号后，再运行相同 base model 的 LoRA RM scale-up。

frozen feature 固定为 backbone 最后一层归一化后的最后一个 response token hidden state；
正常终止时该 token 是 EOS，达到长度上限时是最后一个生成 token。prompt/padding token
不得参与 pooling。linear head 不使用 bias（prompt-level/common bias 在 pair margin 中不可
识别），权重以全零初始化。

### SRM+ 优化节拍

```text
full margins -> m -> warm-start PCG -> detach v
             -> one RM optimizer step -> repeat
```

首版 `beta=1`、无 weight decay、无动态 edge-weight normalization。阻尼固定为
`lambda=c*mean(diag(F))`，`c in {1e-4,1e-3,1e-2}`；主值 `1e-3`，其余是必须的
sensitivity。

### 评价

训练和模型选择不能读取 test geometry。最终 test moment 用每个 prompt 的无偏、严格
gauge-invariant covariance estimator（`M=4`）：

\[
\widehat g_r(x)=\frac1{M-1}\sum_{j=1}^M
(s_j-\bar s)(r_j-\bar r).
\]

对 `r_phi-r*` 聚合后计算 held-out ridge local regret。另报：

- train-derived natural-gradient direction 与 oracle direction 的 Fisher error/cosine；
- 把 train-derived direction 写回同一 LoRA-B 坐标，通过实际 measured KL line search
  到 `kappa=0.01`；
- 在 test prompt 重新采样，以 transformed `r*` 评分 improvement；raw oracle score
  只作次指标；
- pairwise accuracy 只作次指标。

baseline 和 SRM+ 必须使用同一 seed、candidate、标签、feature、初始化、GPU 分区和
停止规则。至少 5 个 paired seeds，报告 paired bootstrap confidence interval。

## Phase 2：CoVal 人类验证

使用固定 revision 的 CoVal world ranking。保留 annotator identity 只用于阻止重复计数，
不作画像。固定有限 `L` 时，`h_L` 是 truncated logit series；ties 丢弃和低标注 edge
筛选都需报告保留率与 selection analysis。

四候选 policy 的 Fisher/edge moment 必须按 candidate policy 概率加权。对无序边
`{j,k}` 使用 `2*pi_bar(j|x)*pi_bar(k|x)`；不能把 6 条边无权枚举后仍称为原
candidate-policy objective。

该阶段只检验现实鲁棒性，不用于证明主定理。

## 每个 run 必存的证据

- Git commit、dirty flag、展开后的 config、随机种子；
- 模型/数据/tokenizer revision 和本地 SHA256；
- GPU/驱动/CUDA/PyTorch/Transformers/PEFT 版本；
- Slurm job/account/partition、开始结束时间；
- prompt split hash、candidate/label/feature schema version；
- Fisher mean diagonal、damping、PCG iterations/residual；
- train/validation/test 指标与 checkpoint hash。
