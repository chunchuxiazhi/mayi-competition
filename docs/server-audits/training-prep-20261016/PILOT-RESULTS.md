# 20次更新真实试训结果

已完成：20次optimizer step、80个micro batch、一次DCP保存、最终FP32 HF导出及部署模型严格重载。所有进程已退出；未运行仿真、动作推理或2000步正式训练。

## 实测

| 项目 | 结果 |
|---|---:|
| 初始化 | 指定原始预训练模型，50任务V3 clean |
| 训练设置 | 1个rank，micro batch 1，累积4，FP32，eager，compile关闭 |
| 可训练参数 | 1,938,090,551个；Qwen骨干冻结 |
| 稳定更新耗时中位数 | 4.871秒（剔除含初始化的首步） |
| 稳定耗时范围 | 4.785–5.384秒 |
| 20步训练循环耗时 | 约106秒，包含首步冷启动，不含权重加载和保存 |
| checkpoint保存 | 约38秒 |
| 成功启动至导出及重载结束 | 约6分51秒 |
| CUDA allocated峰值 | 54.594 GiB |
| CUDA reserved最大记录 | 54.805 GiB，与allocated重叠，不相加 |
| 每次更新时最小可用统一内存 | 48.999 GiB |
| checkpoint写入期间一次采样的可用统一内存 | 约25.953 GiB，不是连续监测得到的绝对最小值 |
| 首步/末步loss | 0.34209 / 0.29512 |
| 全程loss范围 | 0.24183–0.46290，全部有限 |
| DCP完整恢复点 | 41,017,739,148 bytes，约41.02 GB |
| HF权重 | 25,503,889,812 bytes，约25.50 GB |
| HF权重及配置等总量 | 约25.52 GB |
| 本次pilot目录磁盘分配 | 约66.55 GB |
| 比赛目录当前磁盘分配 | 约156.56 GB |
| 磁盘剩余 | 约2.80 TB |

GB为十进制，GiB为二进制。GB10的CPU/GPU共用物理内存；CUDA数值不再加到系统总内存上。未测得整个系统的连续内存峰值。比赛目录较启动前增加约66.83 GB，其中运行目录约66.55 GB，其余约0.27 GB包含缓存等增长。

部署模型严格重载通过，动作输出投影与导出文件完全相同，相对原始预训练权重的最大绝对变化为0.00025343894958496094。全部导出参数为F32。一个DCP包含模型、优化器和随机数等状态，最终HF是额外的一份可评测权重；没有保存多个步数的恢复点。

loss来自不同训练样本，末步低于首步不能当作仿真成功率提高的证据。本轮没有测成功率。官方warmup首个optimizer step的学习率为0；20步指官方训练循环及优化器调用计数，后续非零学习率已使权重实际改变。

## 首次启动问题与修复

首次启动在模型构建时因FlashAttention2未安装而退出，参数更新次数为0。当时只设置了`train.attention_implementation=eager`，但独立的`model.vit_attn_implementation`仍继承官方默认`flash_attention_2`。

在派生pilot/stage1配置中显式增加`model.vit_attn_implementation=eager`，并让启动校验同时核对两处注意力配置。没有安装FlashAttention、改动官方模型文件或切换精度。原失败日志和配置已保存在运行目录的`startup-failures/flash-attention-default`。修复后配置解析及12项CPU策略检查再次通过，随后完成本轮真实试训。

## 后续耗时估算

正式首轮拟使用累积8，试训为累积4。按每个微批成本近似线性外推，每次更新约9.74秒：

- 2000步纯训练约5.41小时，可先按约6小时安排，另留评测时间。
- 三天连续纯训练的理论量约26600步；实际需扣除保存、初始化、导出、评测和停机，不能承诺该数量。

上述是基于试训的估算，累积8和长时间运行尚未实测。checkpoint保存间隔、里程碑保留策略仍需按用户后续决定落实，未启动stage1。

尚未实际执行优化器/DataLoader的中断恢复验证；本轮已验证DCP保存、读取模型状态导出及严格模型重载。短试训通过不能保证任意长训不出问题。

## 服务器文件

运行目录：`/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/pilot`

- 恢复点：`checkpoints/global_step_20`
- 评测权重：`exports/global_step_20/hf_ckpt`
- 实测JSON：`pilot-measurements.json`
- 训练逐步记录：`logs/updates.jsonl`
- 严格重载证据：`reload-verification.json`
- 冻结配置及代码摘要：`frozen-config.json`、`launch-manifest.json`

原始预训练模型、原数据和此前仿真基线均保留。
