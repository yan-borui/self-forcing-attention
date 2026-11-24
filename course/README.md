# Self Forcing × SageAttention

本分支在冻结的四步 Self Forcing 上比较四种自注意力策略，并用单次写回干预追踪历史缓存对后续视频块的影响。推理与评估在一张 24 GB RTX 4090 上完成。[选题与对照设计](proposal.md)

| 实验 | 结果 | 入口 |
|---|---|---|
| 81 帧，四种策略、官方基线、逐层缓存诊断 | [报告与样例视频](../results/pilot-20261123-194819/README.md) | [生成](run_pilot.sh) · [评估](analyze_pilot.sh) |
| 201 帧，7 块滚动缓存、两个压力提示词与两个种子 | [报告与样例视频](../results/length-201-20261124/README.md) · [逐块曲线](../results/length-201-20261124/summary/length-curves.png) | [生成](run_length.sh) · [评估](analyze_length.sh) · [汇总](summarize_length.py) |

每轮结果目录保留 MP4、解码前潜变量、参数、逐例测量和执行记录。潜变量 `.pt` 使用 Git LFS。只浏览代码、视频和指标时，可先跳过下载潜变量，之后按需取回。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch ybr https://github.com/yan-borui/self-forcing-attention.git
cd self-forcing-attention
# 需要潜变量时再执行下一行。
git lfs pull --include='results/**/*.pt'
```

复现用 Self Forcing DMD EMA、Wan2.1-T2V-1.3B 和 SageAttention 1.0.6。模型版本与下载来源见两轮实验记录。运行脚本从仓库根目录寻找 `checkpoints/`、`wan_models/`、`.venv/` 和评估资源 `.course/eval/`。实验记录里的 `${PROJECT_ROOT}` 代表仓库根目录。查看或复用单例命令时可在该目录执行 `export PROJECT_ROOT="$PWD"`。[模型版本](../results/length-201-20261124/provenance/models.txt) · [核心依赖](../results/length-201-20261124/environment.txt) · [作者的模型准备说明](https://github.com/guandeh17/Self-Forcing#installation)
