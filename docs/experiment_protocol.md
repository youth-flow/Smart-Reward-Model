# SRM+ 固定实验协议

本文是实验执行与结果判定规格。数学定义以 [theory.md](theory.md) 为准，所有正式 run
以 `configs/main.yaml` 的 canonical hash 为身份；任何改变 prompt 数、revision、tangent、
optimizer、seed、metric 或停止规则的实验都必须使用新 config，并作为新实验报告。

## 1. 研究问题与唯一主链

主问题是：**当 reward class misspecified 时，直接最小化 policy-relevant ridge local
regret 的 SRM+，能否比 repeated-label BT-MLE 更准确地恢复真实局部 policy update？**

第一阶段只识别这条链：

```text
同一 pi_0 candidate graph
  -> 同一 repeated BTL observations
  -> 同一 frozen feature linear reward class
  -> BT-MLE vs SRM+（只改变训练目标）
  -> held-out Fisher geometry
  -> same measured-KL downstream policy rollouts
```

高容量 reward model、主动选边、多轮 RLHF 和人类 CoVal robustness 均不得替代该 controlled
experiment。代码和 synthetic benchmark 通过只说明实现自洽，不构成 SRM+ 效果结论。

## 2. 预注册身份

### 2.1 Main config

- run name：`controlled-main`
- paired seeds：`20260722, 20260723, 20260724, 20260725, 20260726`
- prompts：2,048；train/validation/test = `1536/256/256`
- candidates：每 prompt 4 个
- dtype：policy、reward feature、oracle 全部 FP32
- reward optimizer：720 outer steps，AdamW，lr `1e-3`，weight decay `0`，
  microbatch `64`，max grad norm `1`
- objective：`beta=1`，PCG relative tolerance `1e-5`，最多 100 iterations
- main relative damping：`c=1e-3`
- mandatory sensitivity：multiplier `0.1, 1, 10`，即 `c=1e-4,1e-3,1e-2`
- measured sequence forward-KL：target `0.01`，relative tolerance `0.05`
- paired percentile bootstrap：10,000 resamples，seed `20260722`

所有 model/dataset revision 是 config 中的 40 位 commit SHA。不得把浮动 `main`、本地目录
mtime 或下载时间当作 revision。

### 2.2 Smoke config

`configs/smoke.yaml` 使用 64 prompts、48/8/8 split 和 10 steps，但保留 main 的 rank-4、
最后四层 `q_proj/v_proj` tangent 与 16-candidate KL probe。oracle batch 上限同为 16；reward
head 只消费已冻结 feature，因此 smoke 的 head microbatch 16 足以覆盖 backbone 峰值。它验证
真实 snapshot、显存、CUDA/Apptainer、I/O 和端到端命令，禁止与 main 结果合并。

## 3. Phase 0：CPU 数值门槛

占用 GPU 前必须运行：

```bash
python -m pip install -e ".[llm,dev]"
smart-reward config-check configs/smoke.yaml
smart-reward config-check configs/main.yaml
smart-reward synthetic-check --seed 0 --output outputs/synthetic.json
pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests
```

LLM environment 必须满足 `transformers>=4.52.3,<5` 并暴露
`Qwen3ForSequenceClassification`；这是 pinned Skywork Qwen3 oracle 的硬兼容门槛。

测试必须覆盖以下数值与安全不变量：

1. randomized estimator `h` 的 Monte Carlo 均值匹配 `logit(p)`，并检查有限二阶矩；
2. PCG 相对残差达到门槛，并与小矩阵 direct solve 一致；
3. primal/dual value 和 envelope gradient 与解析/finite-difference 结果一致；
4. reward 加 prompt-level 常数后 local metric 不变；
5. node/pair 两种 moment/Fisher identity 在模拟中一致；
6. microbatch 与 full-batch 外层梯度等价，dual 每一步刷新；
7. train schema 无法接收 true/oracle reward，split 必须 disjoint；
8. config、JSONL、artifact、comparison identity 不匹配时 fail closed；
9. KL line-search 每次从 zero-B 覆盖，异常或不收敛恢复原点；
10. synthetic output 标记 `benchmark_only=true`，测试不要求 SRM+ 胜过 BT。

任一门槛失败时停止真实模型实验，不得忽略失败继续提交 HPC job。

## 4. Phase 1A：不可变 candidate graph

### 4.1 Prompt

从固定 revision 的 MultiPref 读取 `prompt_id` 与 `text`，按 `prompt_id` 去重后，再用
named seed 做 deterministic prompt-level split。输入行顺序变化不能改变去重选择与 split。
train、validation、test prompt ID 必须两两不交。

### 4.2 Reference policy 与采样

使用固定 revision 的 `Qwen/Qwen2.5-0.5B-Instruct`。tokenizer 必须提供非空 chat
template；prompt 左截断至 384 tokens，response 最多 128 new tokens。

每个 prompt 从 reference distribution 独立返回 4 个 response：

- `do_sample=true`、temperature `1`、top-p `1`、top-k `0`；
- `min_new_tokens=0`、repetition penalty `1`；
- 禁止 beam、top-k/top-p 截断、质量过滤、oracle 筛选、文本去重；
- 不使用改变分布的 logits processor；
- 保留完整 input token IDs、response mask、EOS/达到长度上限状态；
- candidate 生成与 score 提取必须是同一个 FP32、eval、fixed-A/zero-B model instance。

重复文本是合法的独立样本，不得删除。正常终止时 EOS 属于 response；达到长度上限时以
最后一个生成 token 结束。

### 4.3 Fixed-A LoRA policy tangent

main tangent 是 Qwen 最后四层（20–23）`q_proj/v_proj` 的 rank-4 LoRA，
`alpha=rank`、dropout 0：

- A 只随机初始化一次后冻结；B 精确置零且是唯一 `requires_grad=True` 的坐标；
- 加 adapter 前后 probe logits 必须满足 no-op 门槛；
- 保存 A SHA256，以及每个 B 的参数名、shape、flatten offset 和总维度；
- policy score 是完整 response sequence log-probability 之和对 B 的 per-sample gradient；
- 不做长度归一化，不得在 BF16 采样后换 FP32 实例重算 score；
- `S`、feature 和 oracle assembly tensor 存为 float32、detached CPU tensor。

每 prompt 四个 node 全部进入 Fisher：

\[
\widehat F=\frac{1}{PM}\sum_{i=1}^P\sum_{j=1}^M s_{ij}s_{ij}^\top,
\qquad M=4.
\]

训练 edge 只取 canonical candidate `0 - 1`，其 score difference 为
`z=s_0-s_1`。如果 UI 随机交换展示顺序，label 必须映射回 canonical left-win；存储层禁止
只写 `chosen/rejected`。

### 4.4 Frozen reward feature

reward learner 使用同一 Qwen zero-B forward 的最后一层 hidden state，并只 pool 最后一个
response token。正常结束时是 EOS，长度截断时是最后一个生成 token；prompt 和 padding 不得
参与 pooling。backbone 完全冻结，linear scalar head 无 bias、全零初始化。

这个受限 class 是识别 misspecification 机制所必需的。高容量 LoRA RM 只能在主链通过后
作为 scale-up。

## 5. Phase 1B：oracle、标签与泄漏边界

### 5.1 Controlled oracle

使用固定 revision 的 `Skywork/Skywork-Reward-V2-Qwen3-0.6B` sequence-classification
logit。policy 从 GPU 释放后才加载 oracle；两者不同时驻留。

只用 **train 的全部 node raw score** 拟合

\[
b=\operatorname{median}(R),\qquad
\tau=\max\{1.4826\operatorname{median}|R-b|,10^{-6}\}.
\]

冻结后对全部 split 应用

\[
r^*(x,y)=\frac{\log 3}{2}\tanh((R_{oracle}(x,y)-b)/\tau).
\]

因此每条 edge 的 \(|\Delta r^*|\le\log3\)，BTL probability 位于 `[0.25,0.75]`。不得
用 validation/test 重新拟合 `b,tau`。

### 5.2 Repeated BTL observations

每个 split 使用由 base seed 派生的独立 annotation stream；held-out 数量变化不能改变
train labels。对每条 edge：

1. 独立采 `N ~ Geometric(0.1)`，支撑从 1 开始；
2. 以 `p*=sigmoid(r^*_0-r^*_1)` 采 `N` 个条件 iid Bernoulli label；
3. 用 `gamma=0.9` 和完整 label sequence 构造 randomized-truncation `h`；
4. 不得硬截断 `N`，不得按 `N` 给 SRM edge 加权。

BT-MLE 使用全部原始 Bernoulli label，等价于每个 label 等权；SRM+ 每个 edge 对 moment
贡献一次。两者的 weighting 不可互换。

### 5.3 物理隔离

`TrainingTensorData` 只允许：

```text
prompt_ids, policy_scores, reward_features, h, left_wins, num_annotations
```

`true_rewards` 只允许存在于 validation/test `EvaluationTensorData`。训练 JSONL 只含
`raw_labels,N,left_wins,h` 等 observable fields；`true_margin` 只进入 evaluation JSONL。
任何 true/oracle field 出现在 train schema 都必须硬报错。

## 6. Phase 1C：固定预算训练

### 6.1 BT-MLE

对 canonical edge margin `t_phi`，使用 count-compressed、与逐 label 完全等价的 repeated
Bernoulli negative log-likelihood：每个原始 label 权重相同。

### 6.2 SRM+

经验 moment 和 ridge target 固定为

\[
\widehat m_\phi=\frac{1}{2n_E}Z^\top(t_\phi-h),
\quad
\widehat L_\lambda
=\frac{1}{2\beta}\widehat m_\phi^\top
(\widehat F+\lambda I)^{-1}\widehat m_\phi.
\]

每一步执行：

```text
full margins -> m_hat -> warm-start PCG -> detach v
             -> one AdamW step -> repeat
```

其中 `v=(F_hat+lambda I)^-1 m_hat`。microbatch 只允许累积一个 full moment 对应的 outer
gradient；禁止 batch-local `m/F`、stale `v` 跑多个 step、动态 edge-weight normalization。

### 6.3 公平性

BT 与 SRM+ 必须共享：artifact、seed、candidate、label、feature、canonical edge、零初始化
head、optimizer type、lr、step 数、microbatch、gradient clip、weight decay、GPU 分区/型号和
停止规则。validation 只作描述，不能选 checkpoint、early stop 或调 hyperparameter。

主 run 在三档 damping 各自从同一零 head 完整重训；`comparison.json` 必须含唯一
`damping_multiplier=1` 主结果、head bytes 对应的 SHA256 和 final PCG evidence。

## 7. Phase 1D：held-out metric

对每个 held-out prompt 的四 candidate，用无偏、严格 gauge-invariant covariance moment：

\[
\widehat g_r
=\frac{1}{P(M-1)}\sum_{i=1}^P\sum_{j=1}^M
(s_{ij}-\bar s_i)(r_{ij}-\bar r_i).
\]

注意 moment 分母是 `P(M-1)`，held-out node Fisher 分母是 `PM`。每个 split 独立解析
\(\lambda=c\operatorname{mean}(\operatorname{diag}F)\)。主指标为：

1. held-out ridge local regret

   \[
   \frac1{2\beta}m_{error}^\top(F+\lambda I)^{-1}m_{error};
   \]

2. predicted/target damped natural direction 之间的 undamped-Fisher squared error；
3. 同两方向的 Fisher cosine。

Pairwise accuracy 对四 candidate 的全部无序 pair 计算；真实 tie 排除、预测 tie 计 0.5。
它是次指标，不能替代 local regret/direction。

## 8. Phase 1E：matched measured-KL rollout

只使用主阻尼 `c=1e-3` 的两个训练后 head。由 train 的全部四 candidate 构造

\[
d_\phi=\beta^{-1}(F_{train}+\lambda I)^{-1}\widehat g_{r_\phi}.
\]

重新加载相同 revision、相同 named seed 的 fixed-A/zero-B Qwen，并验证 A SHA256、B layout
和 chat-template SHA256 与 artifact 一致。KL probe 是 train 保存 candidate 的共享、
输入顺序无关的确定性子集。

Fisher approximation 给出 `sqrt(2*kappa/(d^T F d))` 作为 line-search 初值，但不得用于
接受更新。每个 trial 从 zero-B 坐标原点覆盖 `alpha*d`，并在保存的完整 token history 上
计算全 vocabulary 的 **sequence-level** forward KL：

\[
\widehat{KL}=\frac1B\sum_{b=1}^{B}\sum_{t\in response_b}
KL\!\left(\pi_0(\cdot\mid h_t)\,\|\,
\pi_{\alpha d}(\cdot\mid h_t)\right).
\]

即每个 response 内先对 token 求和，再对 batch 中 sequence 求均值；禁止除以总 response
token 数。这样 `kappa=0.01` 与 sequence log-prob score/Fisher 的尺度一致。

BT 与 SRM+ 分别达到 `0.01 ± 5%` 才能进入 rollout；不收敛或异常时恢复 zero-B 并使该 seed
失败。test prompt 每个重新采样 4 candidates；zero-B、BT update、SRM+ update 对每个 prompt
重置相同派生 seed，使 candidate index 成为严格 common-random pair。policy 全部卸载后只
加载一次 oracle；raw logit 不落盘，只保存冻结 transform 后的 reward。

结果报告各 learner 的 measured KL、transformed-oracle mean，以及相对**本次同 seed zero-B
rollout** 的 paired improvement。Phase 1 artifact 的原 test reward 使用不同 candidate-
generation stream，只作未配对 descriptive sanity，不能作为 updated rollout 的 paired
reference。experimental unit 是 prompt：对 learner \(\ell\)，先计算

\[
\Delta_i^{(\ell)}=\frac1M\sum_{j=1}^M
\left(r_{ij}^{(\ell)}-r_{ij}^{(0)}\right),
\]

再跨 \(P_{test}\) 个 prompts 报告 \(\bar\Delta^{(\ell)}\) 和 sample
`SE=sd(Delta_i)/sqrt(P_test)`；不得把同 prompt 的四 candidates 当作四个独立实验单位。
最终比较 `SRM improvement - BT improvement`。

## 9. 结果判定与统计

每个 seed 必须完整配对。正式统计以每 seed scalar 的 `SRM - BT` 为单位，报告配对均值、
样本标准差、标准误和 deterministic 95% percentile bootstrap interval；不得把 candidate 或
prompt 当作独立 seed，也不得自动输出 p-value 或“显著”标签。

主结论仅在以下条件全部满足时通过：

1. `c=1e-3` 的 test local regret：配对均值 `<0` 且 interval upper `<0`；
2. `c=1e-3` 的 test squared Fisher error：配对均值 `<0` 且 interval upper `<0`；
3. test Fisher cosine：配对均值 `>0`；
4. 两 learner 每 seed measured KL 都在容差内；rollout improvement 的 `SRM-BT` 配对均值
   `>0` 且 interval lower `>0`；
5. 两个 sensitivity damping 的 `SRM-BT` local-regret 配对均值均严格 `<0`，所有正式
   PCG/KL search 收敛且无数据完整性失败；exact zero 是 inconclusive/`not_passed`。

如果只提高 pairwise accuracy，主想法没有得到验证。如果 local metric 改善而 downstream
rollout 不改善，结论限定为“局部 surrogate 改善但未建立 downstream transfer”。任一主条件
失败后不得更换 seed、挑 checkpoint 或事后改变 primary metric；后续诊断必须标注 exploratory。

## 10. 实际命令链与产物

单 seed：

```bash
seed=20260722
run_dir="outputs/main/seed-${seed}"
mkdir -p "${run_dir}"

smart-reward env-report configs/main.yaml \
  --seed "${seed}" --repo-root . --output "${run_dir}/run-manifest.json"

smart-reward controlled-materialize configs/main.yaml \
  "${run_dir}/artifact" --seed "${seed}" --device cuda

smart-reward controlled-compare configs/main.yaml \
  "${run_dir}/artifact" "${run_dir}/comparison.json" \
  --seed "${seed}" --device cuda \
  --run-manifest "${run_dir}/run-manifest.json"

smart-reward controlled-rollout configs/main.yaml \
  "${run_dir}/artifact" "${run_dir}/comparison.json" "${run_dir}/rollout.json" \
  --seed "${seed}" --device cuda
```

所有写操作都新建/原子替换受控目标；materialization 和 rollout 拒绝覆盖现有完整产物。
`controlled-materialize` 默认离线，只有非正式 staging 时才可显式加 `--allow-download`。
`env-report --seed` 把 manifest 锁到一个 declared seed；CUDA comparison 会校验该 manifest
的 config/selected seed/SHA256、clean Git、`SRM_GIT_COMMIT`、image SHA、Slurm account
`sigroup`、partition 和唯一 GPU model，并要求 artifact producer Git/image 与它一致。正式运行使用 HPC 脚本，
不得在登录节点手工执行这组 CUDA 命令。

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

`aggregate-results` 同时聚合 main-damping held-out metric 与 prompt-level
`test_rollout_improvement`。它要求两组输入的 seed 与 config 完全相同，并验证 artifact
metadata SHA、comparison bytes SHA、rollout JSONL SHA；还会从每个 comparison 同目录重新
读取 `run-manifest.json`，核对 manifest SHA、config、`selected_seed` 与 comparison 记录的
environment identity。任何交叉 artifact/comparison/manifest、缺失或篡改都硬失败。

五个 seed 的 formal identity 必须逐字段相同：Git commit、image SHA256、Slurm account/partition、
唯一 GPU model。聚合器不允许把不同 commit、image、partition 或 GPU 型号的结果放入同一
paired table，并把这份共享 identity 写入 `aggregate.json.environment_identity`。

同一个命令还遍历 config 声明的**每个** damping multiplier，对五 seed test local regret
做配对聚合并写入 `damping_evidence`。每档记录：

- `status=ok|incomplete`、所有 SRM PCG 是否收敛；
- 完整时的 paired local-regret summary 与 `local_regret_nonreversal`；该字段只在 paired
  mean 严格 `<0` 时为 true，exact zero 为 false；
- 不完整时保留逐 seed failure record；solver exception 含 `failure_type/message`，已产出结果
  则保留 `pcg_converged`，不得丢弃失败 seed 后聚合。

随后按第 9 节已经固定的规则写出：

```text
pre_registered_evidence.status = passed | not_passed
pre_registered_evidence.supports_pre_registered_claim = true | false
pre_registered_evidence.criteria = {
  main_local_regret_negative_with_ci,
  main_direction_error_negative_with_ci,
  main_fisher_cosine_positive,
  matched_kl_rollout_positive_with_ci,
  sensitivity_local_regret_nonreversal,
  all_pcg_converged,
  all_measured_kl_updates_converged
}
```

rollout direction PCG、measured-KL convergence 或 KL tolerance 不满足时，输入在写 aggregate
前即被拒绝；sensitivity PCG failure 则保留在 evidence 中并令状态 `not_passed`。`passed`
只表示这组结果满足预注册工程判据，不是 p-value、“统计显著”标签或 population theorem 的
证明。不得根据 `damping_evidence` 事后选择新主阻尼；改变判据/config 必须开启新实验。

Artifact 目录包含 `metadata.json`、`tensors.safetensors`、`prompts.jsonl`、
`candidates.jsonl`、`training_edges.jsonl` 和 `evaluation_edges.jsonl`。rollout 额外生成
`matched-kl-rollout/v1` JSON 和同目录 `updated_rollouts.jsonl`；后者包含 zero-B reference、
BT 和 SRM+ 三路 candidate-index-aligned records。

## 11. 每个 run 必存证据

- Git commit 与 dirty flag、完整 normalized config、config SHA256；
- `selected_seed`、base seed 和 prompt split/candidate generation/LoRA-A/annotation/
  reward-head/minibatch/rollout named seeds；
- dataset/model/tokenizer ID、commit revision、chat-template hash；
- LoRA-A state SHA256、B 参数 layout、zero-B no-op error；
- prompt/candidate/edge JSONL hash、safetensors hash、split prompt IDs；
- artifact producer Git/image digest；formal environment 提供该身份时，consumer 必须逐字节
  匹配，不能跨 commit/image 复用；
- Python、PyTorch、Transformers、PEFT、Datasets、CUDA/cuDNN、GPU 信息；
- GPU smoke 的 Transformers `[4.52.3,5)` / Qwen3 class 验收、`pip check` 和排序后
  `pip freeze`；
- Slurm job/account/partition/node、镜像路径与 SHA256；
- comparison 与 rollout 绑定的 `run-manifest.json` bytes-level SHA256 与 formal environment
  identity；rollout 还会将当前执行进程与 comparison identity 逐字段匹配；
- Fisher mean diagonal、relative/absolute damping、PCG iterations/residual/convergence；
- head init/final SHA256、固定 step 数、validation/test metric；
- shared KL probe IDs、每 learner line-search 轨迹摘要、实际 KL、rollout seed；
- artifact `metadata.json`、`comparison.json`、`updated_rollouts.jsonl` 的 bytes-level SHA256。

跨 seed aggregate 还必须保存并复核共享 Git commit、image SHA256、account、partition 和 GPU model；
其中任一不一致都不得产生 aggregate。

Manifest 只读取明确 allowlist，不得序列化完整 environment。HF/GitHub/W&B credential 不得
进入 config、metadata、stdout、Slurm log 或 artifact evidence。

## 12. Phase 2：CoVal 人类 robustness

只有 Phase 1 主链通过后才启动固定 revision 的 CoVal world-ranking 实验。CoVal 的四个
candidate 是有限支持、非 on-policy 样本；固定有限 label 数只能识别 logit series 的截断，
因此实验必须称为 **candidate-restricted truncated SRM+ robustness**。

保留 annotator identity 仅用于防止重复计数，不作画像。ties、最低 label 数筛选、保留率和
selection analysis 必须完整报告。若给四 candidate 定义 policy probability
\(\bar\pi(j\mid x)\)，无序边 \(\{j,k\}\) 的权重是
\(2\bar\pi(j\mid x)\bar\pi(k\mid x)\)；不得无权枚举六条边后仍称为原 candidate-policy
objective。该阶段检验现实鲁棒性，不证明 Phase 1 的 population theorem。
