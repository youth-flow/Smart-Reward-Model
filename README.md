# Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret

[![CI](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml)
[![HPC4 image](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/build-hpc4-image.yml/badge.svg)](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/build-hpc4-image.yml)

Preference likelihood asks whether a reward model explains past labels. **Prospective Reward Modeling
(ProRM)** instead asks what the downstream policy optimizer will do with that reward model. The method
therefore trains for the reward error that changes the next policy update, rather than for every pointwise
reward error.

The paper title is fixed as:

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret**

This repository implements **ProRM+**, the observable repeated-label Fisher–GMM realization of the ideal
ProRM objective. Its formal comparison is repeated-label Bradley–Terry maximum likelihood (BT-MLE).

## Name and claim contract

The two names refer to different mathematical levels:

| Name | Meaning | Observable/trainable? |
|---|---|---|
| **ProRM** | Ideal population loss: local downstream policy regret measured under the target reward | No; it contains the unobserved target reward |
| **ProRM+** | Repeated-label identification plus Fisher–GMM dual training, implemented with ridge and PCG | Yes, under the stated data contract |

The “+” means that the unobserved ProRM target has been turned into a trainable moment problem. ProRM is
therefore an ideal target, not a separately implemented baseline or an ablation stage. The repository and
Python package retain `Smart-Reward-Model` and `smart_reward` as compatibility infrastructure; public
method terminology is ProRM/ProRM+.

## Current status

| Item | Status |
|---|---|
| Mathematical specification, numerical core, real-model pipeline, immutable artifacts, aggregation | Implemented |
| Automated test suite | Implemented; synthetic runs validate the pipeline, not an effect claim |
| Slurm/Apptainer probe, staging, submission and runtime control plane | Implemented |
| HPC4 account/preflight and host-driver gate | Passed on `gpu-l20`, job `1640437`: NVIDIA L20, driver `570.211.01` |
| Driver-selected image definition and exact Python version lock | Implemented; digest-locked PyTorch 2.7.1/CUDA 12.6 |
| HPC4 GPU environment smoke | **Passed**, job `1640778`; image-build commit `b057bc9e134f1844248d655ed0f6c340af03099f`; validated SIF SHA256 `d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb` |
| Offline Hugging Face cache and config-specific inventories | Separate required gate; stage on a CPU compute node with `submit_hf_stage.sh` |
| Pinned Qwen/Skywork controlled model smoke and five-seed main experiment | **Not yet run** |
| “ProRM+ outperforms BT-MLE” result | **No result yet; this remains the preregistered hypothesis** |

The code and control plane are ready for environment closure and controlled execution. The repository does
not yet contain formal HPC4 results.

## 1. From future policy utility to a reward-model loss

For a candidate reward `r`, let the downstream optimizer return

$$
\theta_r\in\arg\max_\theta
\left\{
\mathbb E_{x\sim\rho,y\sim\pi_\theta}[r(x,y)]
-\beta\mathbb E_{x\sim\rho}
D_{\mathrm{KL}}(\pi_\theta(\cdot|x)\Vert\pi_0(\cdot|x))
\right\}.
$$

The globally correct reward-model criterion is the target-reward utility lost because the optimizer was
given `r_phi` rather than `r*`. That definition is prospective but bilevel and unobservable. ProRM is its
local, closed-form counterpart around the reference policy.

Fix the prompt distribution, `pi_0=pi_{theta_0}`, and the exact tangent coordinates that the next policy
update may change. Define

$$
s_0(x,y)=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0},\qquad
A_0r=\mathbb E[s_0r(x,y)],\qquad
F_0=\mathbb E[s_0s_0^\top].
$$

The ideal population ProRM loss is

$$
\boxed{
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}
\left\|A_0(r_\phi-r^*)\right\|_{F_0^\dagger}^{2}
}.
$$

In the local quadratic policy problem this is exactly the regret of the update induced by `r_phi` under
the target reward. Prompt-only shifts and reward errors in the score null space are not penalized because
they cannot change that update.

## 2. From pairwise labels to ProRM+

Sample a natural pair from

$$
Q_0(dx,dy,dy')=\rho(dx)\pi_0(dy|x)\pi_0(dy'|x),
$$

and define

$$
z_0=s_0(x,y)-s_0(x,y'),\qquad
\Delta r_\phi=r_\phi(x,y)-r_\phi(x,y').
$$

The score identity gives

$$
A_0r=\frac12\mathbb E_{Q_0}[z_0\Delta r].
$$

A single Bernoulli preference cannot provide a per-edge unbiased estimate of a BTL logit. ProRM+ obtains
conditionally iid repeated labels for the same edge and constructs a randomized U-statistic `h` satisfying

$$
\mathbb E[h\mid e]=\operatorname{logit}(p^*(e))=\Delta r^*(e).
$$

Consequently,

$$
\boxed{
m_\phi=\frac12\mathbb E[z_0(\Delta r_\phi-h)]
=A_0(r_\phi-r^*)
}.
$$

The two data streams have separate roles:

```text
Fisher stream:          (x,y) ~ rho*pi_0       -> s_0 -> F_0
Repeated-label stream:  e ~ Q_0, labels -> h   -> z_0 -> m_phi
                                                   |
                                                   v
                                         Fisher-GMM ProRM+
```

At population level and without damping,

$$
\boxed{
\min_\phi\max_v\frac1\beta
\left[v^\top m_\phi-\frac12v^\top F_0v\right]
=\min_\phi\mathcal L_{\mathrm{ProRM}}(\phi)
}.
$$

This identity requires natural `Q_0` pairs and the repeated-label assumptions. The three-edge
[closed-form example](docs/closed_form_example.md) establishes a population ordering reversal between
BT-MLE and the ideal ProRM target; it does **not** by itself establish the ProRM+ identification theorem.
That theorem uses the natural `Q_0` expectation above.

## 3. Empirical ridge ProRM+

With all on-policy node scores in `S` and canonical labeled-edge differences in `Z`, the implementation
uses

$$
\widehat F_0=\frac1{n_F}S^\top S,
\qquad
\widehat m_\phi=\frac1{2n_E}Z^\top(\Delta r_\phi-h),
$$

and trains the explicitly damped empirical objective

$$
\boxed{
\min_\phi\max_v\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F_0+\lambda I)v
\right]
},
$$

equivalently,

$$
\widehat L_\lambda(\phi)
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F_0+\lambda I)^{-1}\widehat m_\phi,
\qquad
\lambda=c\,\operatorname{mean}(\operatorname{diag}\widehat F_0)>0.
$$

| Level | Exact claim |
|---|---|
| Population, `lambda=0`, `F_0^dagger` | ProRM+ inner optimum equals local ProRM regret |
| Finite sample, `lambda>0` | Ridge-regularized empirical surrogate |
| `c in {1e-4,1e-3,1e-2}` | Preregistered damping sensitivity, not post-hoc tuning |

PCG solves `(F_hat + lambda*I)v=m_hat` without forming a dense Fisher. The reported quadratic value and
the detached envelope surrogate differ by a factor of two in value but yield the correct gradient; the
derivation and tests are documented in [theory.md](docs/theory.md).

## 4. Controlled Phase 1 experiment

The fixed question is:

> Under the same restricted reward class and training budget, does ProRM+ recover the operational-oracle
> policy-update direction more accurately than repeated-label BT-MLE, and does that advantage survive
> equal measured-KL policy optimization?

The target `r*` in Phase 1 is a train-calibrated transformation of frozen Skywork scores. It is an
**operational oracle**, not human utility. BT-MLE and ProRM+ share candidates, repeated labels, features,
zero initialization, optimizer, step count, GPU and stopping rule; only the training objective changes.

MultiPref supplies **prompts only** in this controlled experiment; its historical human preference labels
are not training targets. Qwen generates the four candidate responses. For canonical candidate pair
`0-1`, frozen Skywork defines $p^*=\sigma(\Delta r^*)$; a named seed then generates conditionally iid
Bernoulli repeats and the randomized estimator `h`. Thus the Phase-1 “annotator” is a reproducible
Skywork-defined BTL simulator, not a new human-labeling round.

```text
MultiPref prompts
    -> pi_0: four exact-token candidates per prompt
       -> fixed-A LoRA-B scores --------> Fisher geometry
       -> frozen hidden features -------> zero-init linear reward class
       -> frozen operational oracle ----> train-only calibration
                                           -> repeated BTL labels
                                                  |          |
                                               BT-MLE      ProRM+
                                                  \          /
                                           held-out geometry
                                                    |
                                      matched measured-KL rollouts
```

| Component | Locked design |
|---|---|
| Prompts | MultiPref pinned revision; `1536/256/256` prompt-level split |
| Reference policy | Pinned Qwen2.5-0.5B-Instruct, FP32 |
| Candidates | Four independent base-distribution samples per prompt; no filtering or deduplication |
| Policy tangent | Last four `q_proj/v_proj` modules, rank-4 fixed-A LoRA-B |
| Oracle | Pinned Skywork-Reward-V2-Qwen3-0.6B, FP32 |
| Repeated labels | Canonical candidate `0-1`; geometric continuation `gamma=0.9`, hence `E[N]=10` |
| Reward class | Frozen final-response-token feature plus bias-free linear head |
| Training | 720 fixed steps; identical optimization budget |
| Evaluation | Held-out Fisher geometry plus measured sequence-KL `0.01 ± 5%` rollout |
| Statistics | Five paired seeds; fixed main damping plus two sensitivity settings |

The capacity bottleneck does not logically guarantee misspecification. The immutable artifact therefore
records a train-only, prompt-centered linear projection residual under
`train_reward_class_projection`. It is descriptive mechanism evidence and cannot select a checkpoint,
damping or conclusion.

## 5. Evidence required for a positive result

Pairwise prediction is descriptive. Held-out BTL NLL and oracle-probability MAE measure preference fit;
they are not success gates. `aggregate.json` may report `passed` only if all preregistered policy evidence
passes:

| Evidence | Fixed five-seed criterion |
|---|---|
| Main-damping held-out ridge local-regret proxy | `ProRM+-BT-MLE` mean `<0`, bootstrap upper `<0` |
| Squared Fisher direction error | `ProRM+-BT-MLE` mean `<0`, bootstrap upper `<0` |
| Fisher cosine | `ProRM+-BT-MLE` mean `>0`; both direction norms nonzero |
| Matched-KL rollout improvement | Both methods meet KL tolerance; `ProRM+-BT-MLE` mean `>0`, bootstrap lower `>0` |
| Damping sensitivity | Both secondary local-regret means `<0`; all required PCG solves converge |
| Identity and numerical integrity | PCG/KL convergence plus identical Git/image/GPU/manifest identities |

The percentile-bootstrap interval over five preregistered paired seeds is an engineering decision interval,
not a population confidence interval or p-value.

| Observed pattern | Permitted conclusion |
|---|---|
| Geometry, rollout and sensitivity all pass | Supports the preregistered prospective reward-modeling mechanism claim |
| Geometry passes but rollout fails | Local surrogate improved; downstream transfer not established |
| Geometry fails | Core mechanism not supported |
| Sensitivity fails or reverses | Failure remains in evidence; status is `not_passed` |
| Only NLL/accuracy/probability MAE improves | Not evidence that ProRM+ succeeded |

No such result is currently claimed.

## 6. Local verification

```bash
python -m pip install -e ".[dev]"
prorm config-check configs/smoke.yaml
prorm config-check configs/main.yaml
prorm closed-form-check --output outputs/closed-form.json
prorm synthetic-check --seed 0 --output outputs/synthetic.json
pytest -q
ruff check .
ruff format --check .
```

`closed-form-check` is marked `population_example_only=true`; it verifies the analytic ordering reversal
without presenting the three-edge distribution as ProRM+ training data. `synthetic-check` is always marked
`benchmark_only=true`; it validates identities and integration and does not assert that ProRM+ must beat
BT-MLE. Real Hugging Face execution additionally needs:

```bash
python -m pip install -e ".[llm,dev]"
```

`prorm` is the public CLI name. The historical `smart-reward` executable and `smart_reward` import package
remain compatibility surfaces while artifacts and scripts migrate.

## 7. HKUST HPC4 entry

Repository inputs are relative to the checkout. Only the cross-node project and scratch anchors must be
absolute:

| Content | Persistent or temporary location |
|---|---|
| Git checkout | `$HOME/Smart-Reward-Model` |
| Qwen, Skywork and raw MultiPref snapshots | `$PRORM_PROJECT_ROOT/hf-cache/hub` |
| Processed Hugging Face/Arrow dataset cache | `$PRORM_PROJECT_ROOT/hf-cache/datasets` |
| Image, build/staging/GPU evidence | `$PRORM_PROJECT_ROOT/{images,system-reports}` |
| Generated candidates, labels, scores, features and Fisher data | `$PRORM_PROJECT_ROOT/artifacts/...` |
| Learned linear RM heads, rollouts and aggregate | `$PRORM_PROJECT_ROOT/runs/...` |
| Per-job working copy | `$PRORM_SCRATCH_ROOT/jobs/$SLURM_JOB_ID` |

The experiment does not create another full Qwen checkpoint. Base weights stay in the pinned HF cache;
the learned bias-free linear reward heads are serialized in `comparison.json`. The local LoRA-B policy
update is reconstructed for evaluation and is not exported as a production adapter checkpoint.
Each persistent run contains a relative `artifact` symlink to its content-addressed Phase-1 artifact, so
serialized POSIX path references remain valid after scratch cleanup. Heavy assets and results are ignored
by Git.

The first SSH connection is the only interactive identity step: enter the ITSO password and complete
Duo/2FA in the SSH client. Never send a password, 2FA response, private key or recovery code to Codex, put
one in this repository, or place one in a Slurm log. After that private login:

```bash
ssh YOUR_ITSO@hpc4.ust.hk
git clone https://github.com/youth-flow/Smart-Reward-Model.git
cd Smart-Reward-Model
test "$(git remote get-url origin)" = \
  "https://github.com/youth-flow/Smart-Reward-Model.git"
git rev-parse --verify HEAD

# Private, ignored path configuration; never edit the tracked example.
test -e .env.hpc4 || cp scripts/hpc4/env.example .env.hpc4
source .env.hpc4
mkdir -p \
  "${PRORM_PROJECT_ROOT}"/{images,hf-cache,system-reports,slurm-logs,artifacts,runs} \
  "${PRORM_SCRATCH_ROOT}/jobs"
bash scripts/hpc4/preflight.sh

# No image is needed for this first GPU/driver observation.
bash scripts/hpc4/submit_host_gpu_probe.sh gpu-l20
```

The completed gate is job `1640437`. It observed one NVIDIA L20 (46,068 MiB), driver `570.211.01` and
maximum supported CUDA 12.8. The resulting candidate is therefore the digest-locked
PyTorch 2.7.1/CUDA 12.6 definition in
[`containers/prorm-hpc4.def`](containers/prorm-hpc4.def), with the exact Python package lock in
[`containers/requirements-hpc4.lock`](containers/requirements-hpc4.lock).

HPC4 cannot build the definition locally because its Apptainer installation has no SUID builder or
subuid/subgid mapping, and user namespaces are disabled on the login node. The login node is therefore
limited to Git, file checks and Slurm submission; it must not run `apptainer exec` for HF staging. The
dedicated GitHub workflow builds the raw SIF, records build evidence and publishes it through public
GHCR ORAS. Pull the validated artifact by its immutable **image-build commit**, not by the current source
`HEAD`; the source checkout may legitimately contain later staging/control-plane changes. The fetcher
resolves and verifies the OCI manifest digest and requires the local SIF SHA256 to equal the manifest's
SIF-layer digest:

```bash
image_build_commit=b057bc9e134f1844248d655ed0f6c340af03099f
bash scripts/hpc4/fetch_candidate_image.sh "${image_build_commit}"

export PRORM_IMAGE=images/prorm.sif
export PRORM_HF_CACHE=hf-cache
export PRORM_IMAGE_SHA256=d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb
printf '%s  %s\n' \
  "${PRORM_IMAGE_SHA256}" "${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}" \
  | sha256sum --check
```

This exact SIF passed the HPC4 GPU environment smoke in job `1640778`; its persistent report is
`$PRORM_PROJECT_ROOT/system-reports/gpu-smoke-1640778.txt`. A file with any other SHA256 remains an
unvalidated candidate.

HF model and dataset staging is a separate, mandatory gate. Because login-node user namespaces are
disabled, do not run `stage_hf_assets.py` or `apptainer exec` directly there. The two configs share one
HF cache, so their first downloads must be serialized on an allowed CPU compute partition (`amd` or
`intel`). Submit the smoke stage first:

```bash
export PRORM_HF_STAGE_WALLTIME=04:00:00
cache_root="${PRORM_PROJECT_ROOT}/${PRORM_HF_CACHE}"
smoke_stage_job="$(
  bash scripts/hpc4/submit_hf_stage.sh \
    configs/smoke.yaml amd "${PRORM_HF_STAGE_WALLTIME}"
)"
smoke_stage_job="${smoke_stage_job%%;*}"
test -n "${smoke_stage_job}"
squeue -j "${smoke_stage_job}"
```

After the smoke stage leaves the queue, require `COMPLETED`, `ExitCode=0:0` and an exact
`status=passed` report before submitting the main stage:

```bash
sacct -j "${smoke_stage_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition
smoke_stage_report="${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${smoke_stage_job}.log"
tail -n 20 "${smoke_stage_report}"
grep -Fx 'status=passed' "${smoke_stage_report}"

main_stage_job="$(
  bash scripts/hpc4/submit_hf_stage.sh \
    configs/main.yaml amd "${PRORM_HF_STAGE_WALLTIME}"
)"
main_stage_job="${main_stage_job%%;*}"
test -n "${main_stage_job}"
squeue -j "${main_stage_job}"
```

After the main stage leaves the queue, apply the same acceptance check:

```bash
sacct -j "${main_stage_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition
main_stage_report="${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${main_stage_job}.log"
tail -n 20 "${main_stage_report}"
grep -Fx 'status=passed' "${main_stage_report}"
sha256sum "${cache_root}"/inventories/*.json
```

`04:00:00` is the staging default. Change it only if an administrator-enforced partition limit requires
an approved lower value.

Staging downloads only the public pinned snapshots. Its offline proof resolves snapshot revisions,
configs, tokenizers and the MultiPref dataset; it does not claim to have instantiated model weights.
Actual Qwen/Skywork weight loading is tested by the controlled model smoke. Each config-specific inventory
digest is reverified offline and bound into the run manifest, artifact producer identity and final
aggregate.

Only after the validated-image GPU smoke has passed, both CPU staging jobs are `COMPLETED` with
`ExitCode=0:0`, both config-specific inventories exist, the Git checkout is clean and `HEAD` equals the
reviewed remote commit may `submit_controlled.sh` be used:

```bash
git fetch origin main
test -z "$(git status --porcelain --untracked-files=normal)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"

export PRORM_SMOKE_WALLTIME=REPLACE_WITH_APPROVED_PILOT_WALLTIME
bash scripts/hpc4/submit_controlled.sh \
  configs/smoke.yaml gpu-l20 "${PRORM_SMOKE_WALLTIME}"

# Fill these only from the accepted smoke measurements.
export PRORM_ARRAY_CONCURRENCY=1
export PRORM_MAIN_WALLTIME=REPLACE_WITH_SMOKE_DERIVED_WALLTIME
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml gpu-l20 "${PRORM_MAIN_WALLTIME}"
```

Formal jobs never use `--allow-download`. They bind the submission Git commit and config-specific cache
inventory before allocation work begins. The run manifest records that **source Git SHA** separately from
the validated **SIF SHA256**; it does not require the source commit to equal image-build commit
`b057bc9e134f1844248d655ed0f6c340af03099f`. Wall time, GPU-hours and storage budgets come from the
accepted smoke record. The five-seed aggregate must be produced inside the same image under
`$PRORM_PROJECT_ROOT/runs`, not in the Git checkout. See [hpc4.md](docs/hpc4.md) for the exact validation,
aggregation and scratch-retention commands.

## 8. Documentation and code map

| Goal | Entry point |
|---|---|
| Global-to-local derivation, assumptions and contribution boundary | [docs/theory.md](docs/theory.md) |
| Three-edge closed-form population ordering reversal | [docs/closed_form_example.md](docs/closed_form_example.md) |
| Fixed Phase 0–1 design, metrics and artifacts | [docs/experiment_protocol.md](docs/experiment_protocol.md) |
| HPC4 environment closure and Slurm execution | [docs/hpc4.md](docs/hpc4.md) |
| Formal design identity | [configs/main.yaml](configs/main.yaml) |

```text
Smart-Reward-Model/             # retained repository name
├── configs/                    # closed-schema smoke/main designs
├── containers/                 # digest-locked HPC4 definition and exact runtime lock
├── docs/                       # theory, examples, protocol, HPC4 runbook
├── scripts/hpc4/               # preflight, driver probe, staging, GPU smoke, arrays
├── src/smart_reward/           # retained compatibility package
│   ├── annotations.py          # randomized repeated-label estimator
│   ├── objective.py            # moment, reported value, envelope gradient
│   ├── training.py             # paired BT-MLE / ProRM+ trainers
│   ├── phase1.py               # immutable real-model materialization
│   ├── rollout.py              # natural directions and measured-KL updates
│   ├── statistics.py           # paired-seed aggregation
│   └── cli.py                  # fail-closed control plane
└── tests/
```

## 9. Claim boundary and execution order

1. Freeze and validate the image, environment lock and offline cache.
2. Pass GPU environment smoke and the controlled model smoke.
3. Run the five paired Phase 1 seeds without changing design identity.
4. Aggregate only complete identity-matched runs.
5. Scale reward-model capacity only after the controlled mechanism result is known.
6. Treat CoVal as human-label robustness, not as a test of the exact Phase 1 theorem.

With fixed finite labels, CoVal identifies only a truncated logit series. It must be reported as
**candidate-restricted truncated ProRM+ robustness** and cannot inherit the exact unbiasedness or human-
utility interpretation of the controlled experiment.

Primary engineering dependencies and data/model assets:

- [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [MultiPref](https://huggingface.co/datasets/allenai/multipref)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Skywork Reward V2 Qwen3 0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
- [CoVal](https://huggingface.co/datasets/openai/coval)
