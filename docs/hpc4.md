# HKUST HPC4 运行规范

邮件确认 ITSO 账号已关联 Slurm account `sigroup`，登录地址为 `hpc4.ust.hk`；CPU
分区为 `amd/intel`，GPU 分区包括 `gpu-a30`、`gpu-l20`、`gpu-rtx5880`、
`gpu-rtx4090d`。这封邮件没有证明当前 QoS、wall-time、显存、驱动或外网状态，必须
以首次登录预检和计算节点 smoke job 为准。

## 存储分层

- `$HOME`：代码和小型配置，不放模型与数据；配额 200 GB。
- `/project/sigroup/smart-reward-model`：模型/数据主副本、容器、最终结果与可恢复
  checkpoint；组共享 project 配额 10 TB。
- `/scratch/$USER/smart-reward-model`：HF/Torch cache、job staging、临时 checkpoint；
  per-user 500 GB，60 天不活跃文件会清理，不能作为永久存储。

每个作业使用 `/scratch/$USER/smart-reward-model/jobs/$SLURM_JOB_ID`，避免并发覆盖。
长作业需原子保存 checkpoint，并定期同步到 project。

## 首次登录

校外先连接 HKUST VPN，再 SSH 登录。登录节点只做 Git、检查、传输和提交：

```bash
ssh <ITSO>@hpc4.ust.hk
bash scripts/hpc4/preflight.sh
```

预检必须确认 `squota -A sigroup`、分区、project/scratch 写权限、Apptainer 和 module
状态。随后提交单 GPU smoke；初次只申请 GPU 主资源，不猜测 `--mem` 或 CPU：

```bash
bash scripts/hpc4/submit_gpu_smoke.sh gpu-l20
squeue -u "$USER"
```

若 `gpu-l20` 无权限或 QoS 不合适，依次对邮件列出的其他 GPU 分区做同一 smoke；正式
paired comparison 必须锁定同一分区和 GPU 型号。

## 环境

正式实验使用预先构建并校验的 Apptainer 镜像，计算作业不依赖实时联网安装。容器、
模型和数据 revision 先放 project，cache 指向 scratch。只有 smoke 记录实际驱动/CUDA
兼容性后才生成 Python lock；本地 CPU lock 不能作为 HPC 复现环境。

默认离线日志。任何 Hugging Face/GitHub/W&B token 只通过 HPC secret 环境注入，不写入
仓库、配置或 Slurm 输出。

## 计费与并发

HPC4 按月向 PI 计费。第一阶段固定单节点单 GPU；在获得明确 GPU-hour 预算前不提交
无界 job array，不启动多节点。所有 sweep 必须有 seed 数、分区、并发上限和预计
GPU-hour。
