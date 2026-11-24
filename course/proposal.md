# 四步视频生成中，把一次原生注意力留给历史写回

Self Forcing（NeurIPS 2025）的官方 DMD 配置对每个视频块做四次去噪，先保存该块潜变量，再用同一份潜变量、零噪声时间步额外前向一次，覆盖持久的键值缓存（KV）。后续块会读这份历史。SageAttention（ICLR 2025）用 INT8 近似查询和键的乘积。我们提出在四次去噪中使用 Sage，仅在写回时恢复原生 BF16 自注意力。另一种同预算分配把 BF16 放在最后一次去噪。两者每块都调用一次原生自注意力，分别保护当前输出和未来的条件信息。[Self Forcing](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html) · [SageAttention](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b286c344d38e10d2466c0514b78e2f36-Abstract-Conference.html) · [生成与写回代码](https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/pipeline/causal_inference.py#L187-L244)

这里有一个可由源码确认的接入问题。Self Forcing 给算子的是持久 K 的切片。SageAttention 1.0.6 默认在传入的 K 上原地减均值。直接调用会改写过去的键，相邻调用读到的历史随之变化。补丁在临时张量上完成相同的中心化，随后关闭算子内部的重复中心化。各组沿用 BF16 的 KV 存储、同一注意力窗口和块内连接方式，临时张量的成本进入实测耗时。[缓存切片](https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/wan/modules/causal_model.py#L201-L235) · [SageAttention v1 源码](https://github.com/thu-ml/SageAttention/blob/2bec28a19edaba8cddfba60c0ab87a0f7f9b58f6/sageattention/core.py#L91-L109)

**已有工作划定了贡献范围。** Q-ARVD 已经测过整块量化如何影响后续帧。Speculative Forcing 已经采用部分 token 提前结束去噪、全块写回。ScoutCache 研究了构建历史时保留哪些 token。我们查到的这些工作没有报告“固定已保存的当前潜变量，只替换一次写回的数值算子，再比较一次原生注意力放在哪里”。Wave Forcing 的 1.3B 四步八卡设置更给出了一个反例。统一更换 Sage 后，端到端吞吐由 134.81 变为 134.79 FPS。单张 4090 的收益需要另测。[Q-ARVD §3.1](https://arxiv.org/html/2605.21072#S3.SS1) · [Speculative Forcing §3.2](https://arxiv.org/html/2610.22347#S3.SS2) · [ScoutCache §3.4](https://arxiv.org/html/2611.21038#S3.SS4) · [Wave Forcing 表 1](https://arxiv.org/html/2611.00859#S5)

**第一组对照只改第二块的一次写回。** 用相同提示词顺序、种子、批大小和软件环境，分别运行全程原生与单块干预。原始短片有 21 个潜帧，按三个潜帧分为七块，解码成 81 帧。检查前两块已保存的潜变量逐元素相等，再画后五块各自相对原生轨迹的误差。第一层的 K/V 在该层自注意力之前就已写入，因此配对输入相同时应相同。更深层的 K/V 可以携带前层误差。若需定位传播深度，可在写回后单独记录两次运行的对应层有效切片。原型已提供按样本导出潜变量和单块干预的入口。

**决定是否值得继续的实验**是在相同提示词和种子上比较下面四组。分别报告预热后的单视频生成耗时中位数、逐块潜变量相对误差，以及主体一致性和运动幅度。后两项配合成对盲看来判断视频质量。按提示词汇总差异。若两种混合策略时间相近，写回策略更能保住末段质量，同时比全原生更快，才支持所提方法。若所有误差都接近原生重复运行的波动，或成本都花在其他模块，则保留这个负结果。

| 补丁参数 | 每个生成块的四次去噪 | 干净历史写回 |
|---|---|---|
| `--attention_policy native` | 四次原生 | 原生 |
| `--attention_policy sage` | 四次 Sage | Sage |
| `--attention_policy native_write` | 四次 Sage | 原生 |
| `--attention_policy native_last_denoise` | 前三次 Sage，末次原生 | Sage |

### 原型入口

把同目录的 `self-forcing-attention.patch` 放到对应的 Self Forcing 源码根目录。基准版本为 `33593df3e81fa3ec10239271dd2c100facac6de1`。沿用作者的 Linux 推理环境、Wan2.1-T2V-1.3B 权重和 `self_forcing_dmd.pt`。Sage 实验固定使用 `sageattention==1.0.6`。作者测试过 24 GB RTX 4090。官方仓库给出了显存、主存和权重要求。[基座与权重](https://github.com/guandeh17/Self-Forcing/blob/33593df3e81fa3ec10239271dd2c100facac6de1/README.md#L31-L74)

```bash
git apply --check self-forcing-attention.patch
git apply self-forcing-attention.patch
head -n 4 prompts/MovieGenVideoBench_extended.txt > pilot-prompts.txt
python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt --use_ema \
  --data_path pilot-prompts.txt --num_output_frames 21 --num_samples 1 \
  --seed 2026 --save_with_index --save_latents --report_latency \
  --output_folder videos/probe-chunk-1 --sage_write_probe_chunk 1
```

参数中的块编号从零开始。运行配对的原生参照时，将最后一行换成 `--output_folder videos/native --attention_policy native`。其余三种策略各用独立的输出目录。相同编号的 `.pt` 保存解码前的潜变量，每连续三个潜帧对应一块。`pipeline_seconds` 包含提示词编码、去噪、写回、解码及其间的设备搬运，排除模型加载和文件写盘。首条样本单报以区分冷启动。四条提示词用来检查机制，最终结论需要另留提示词，覆盖主体运动、遮挡和镜头移动。
