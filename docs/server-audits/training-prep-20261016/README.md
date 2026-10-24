# GB10 单卡 clean V3 微调准备

> 后续已按用户要求完成20步真实试训，结果见[PILOT-RESULTS.md](PILOT-RESULTS.md)。下文保留准备阶段的说明，其中“尚未训练”描述的是启动前状态。2000步训练仍未启动。

当前已安装依赖、创建配置和适配器，并完成CPU数据/策略检查。**尚未加载模型进行训练，参数更新次数为0，没有训练checkpoint。** 真实前向、反向、峰值内存、保存与重新加载仍需后续20步试训验证。

## 配置如何处理

以官方 `configs/vla/robotwin/robotwin.yaml` 为起点，派生两份项目专用配置；保留官方原文件、训练循环、模型定义和仿真文件，不在原文件上叠加本机特例。

- 配置：`/home/zzj/competitions/lingbot/configs/training/clean-v3-gb10/{pilot,stage1}.yaml`。
- 机器人映射：`/home/zzj/competitions/lingbot/configs/robots/clean-v3-gb10/robotwin.yaml`，指向现有clean-only归一化。
- 适配代码：`/home/zzj/competitions/lingbot/tools/finetune-clean-v3-gb10/`。
- 输出：`/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/{pilot,stage1}/`。
- 安装、测试和来源记录：`/home/zzj/competitions/lingbot/docs/training-prep-20261016/`。

配置是基于官方完整配方的派生，而非重写网络架构。初始化仍为指定原始预训练模型；训练数据为现有合并V3 clean，50任务联合训练一个策略。`train_expert_only`冻结约4.438B元素的Qwen视觉语言骨干；约1.787B动作专家及其他未冻结投影/对齐模块训练。它不是每个任务一个模型，也不是按任务手选MoE专家。

保持官方深度/视频蒸馏和MoE结构。训练模型采用FP32，compile关闭，eager注意力、梯度检查点。视频教师选择仓库已有的`sdpa_block_causal`实现，保留分块因果掩码，避免默认flex路径隐式调用compile；教师目标计算保留官方代码里的BF16 autocast。这不是把全部计算都改成纯FP32。

## 参数更新、步数与并行

| 参数 | 20步功能试训 | 正式首轮预算 |
|---|---:|---:|
| GPU训练rank/常驻VLA模型 | 1/1 | 1/1 |
| 单次微批micro batch | 1 | 1 |
| 梯度累积次数 | 4 | 8 |
| 每次更新的有效样本数 | 4 | 8 |
| 参数更新次数上限 | 20 | 2000 |
| 模型参数更新频率 | 每4个微批一次 | 每8个微批一次 |
| checkpoint间隔（参数更新次数） | 20 | 根据试训耗时校准，最多250 |
| 保留完整恢复点 | 1 | 最近2个 |
| DataLoader worker | 0（主进程同步读取） | 0（主进程同步读取） |
| 预取 | 关闭 | 关闭 |
| DCP写入线程 | 1 | 1 |

正式配置每次参数更新处理8个单样本微批。保存间隔以20步试训的更新耗时中位数为依据，剔除首步冷启动，并按有效batch从4增至8进行估算，目标约30分钟保存一次、最多250次更新；实际值冻结在运行配置及启动清单中。若取上限250，相当于处理约2000个样本后保存一次。**参数始终每8个微批更新一次，保存是另一件事。** 学习率在每次参数更新后由官方scheduler更新：AdamW，初始峰值2e-5、最低1e-5、cosine；试训warmup2步，正式首轮warmup60步。

2000步是首轮预算，不是声称最优或收敛；总计约16000个帧样本，约为全部548893帧的2.9%，并不等于遍历全部2500段示范。动作窗口长50；这也不能直接当作完整训练epoch的替代计数。后续应依据真实秒/step、loss和仿真收益调整预算。正式启动前可审阅配置；一旦冻结到运行目录，恢复时要求相同配置，不能悄悄修改max_steps、LR或batch。

试训只验证管线，正式首轮重新从指定预训练权重开始；不会把20步功能试训的权重误标为最终参赛模型。训练、CPU导出、GPU重新加载串行执行，不并行启动评测或第二份训练模型；也不与其他GPU计算进程抢跑。

最初验证了两个spawn worker的原始视频读取，但进一步使用完整归一化StatefulDataLoader并提前结束迭代时，复现了worker SIGABRT退出异常；显式调用私有shutdown方法也未解决。因此正式默认改为`num_workers=0`，同步读取，实际向原生加载器传`prefetch_factor=None`。这条路径的微批、状态序列化和正常退出已通过。未把退出码0但stderr有原生异常的旧检查当成最终通过。相似症状在[上游PyTorch2.9.1相关报告](https://github.com/Lightning-AI/pytorch-lightning/issues/21703)中出现，但不足以确认本机根因；本轮规避未验证的并行路径，不吞异常。后续只有确认读取成为瓶颈且修复退出问题后才提高worker数。

## 内存预算

GB10的CPU/GPU共用约119.7 GiB物理内存，不把RAM和显存相加。静态按FP32+AdamW估算：全量训练仅权重、梯度、两个moment约95 GiB；冻结Qwen后相应状态约45.4 GiB，尚未包含教师、激活、工作区、CPU staging和系统占用。教师、激活等存在额外开销，真实峰值未测，不能保证当前配置一定在预算内。

代码保护如下：

- 启动前要求MemAvailable至少90 GiB；GPU上已有计算进程时拒绝启动。
- PyTorch CUDA分配器上限72 GiB，属于同一统一内存池，**不是另有72 GiB独立显存**，也不是整个进程的硬RAM上限。
- 训练父进程每2秒检查可用统一内存；低于16 GiB发送停止信号，20秒仍未退出则终止其进程组。优化器更新前也检查内存，保存DCP前要求至少28 GiB可用。
- 梯度检查点、micro batch=1，CPU数据读取同步、无worker子进程和预取、pin_memory=false；训练数据类仍采用可被spawn序列化的模块级子类，保留后续独立验证并行的基础。
- 非有限loss立即停止，原生梯度裁剪启用error_if_nonfinite；记录每次更新的loss、梯度范数、LR、耗时、CUDA分配与保留量、统一内存可用量。
- 不依赖CPU offload“多出一份内存”；不使用在GB10上尚未验证的FlashAttention。

阈值和轮询是保护措施，不是无瞬时峰值/OOM的保证。20步试训必须核实实际峰值；若超限，停止并调整，不自动切精度、不自动删样本或启动第二路训练。

## checkpoint与磁盘

FP32完整HF权重约25.5 GB。按冻结范围估算，每个DCP恢复点约41 GB（模型约25.5 GB+Adam状态约15.5 GB，另有少量状态）；这是估算，实际保存大小待试训确认。

- 训练期间只保存同步DCP，不同时生成HF副本；完成训练、整个训练进程退出后，单独CPU导出最终HF，再单独GPU严格加载验证。
- 一个恢复点包含模型、优化器、步数、学习率scheduler、DataLoader状态、Python/NumPy/Torch CPU/CUDA随机数状态。
- 新点完整写入、fsync并检查DCP组成后写`complete.json`，随后才移除本运行目录里过旧的已完成恢复点。正式首轮稳态2份，保存时可能短暂3份；不清理未知或未完成目录。
- 记录文件大小和关键metadata摘要；恢复时检查完整性与配置签名，要求优化器完整恢复，禁止上游捕获异常后悄悄跳过优化器或从头重来。
- 未完成checkpoint或损坏状态保留并阻止自动覆盖；不自动换成另一个实验。已达到计划步数的运行拒绝再次更新参数。
- 试训和正式首轮共用`runs/finetune-clean-v3-gb10`的250 GiB（约268 GB）新增存储预算；保存前预留64 GiB的新恢复点空间，并要求磁盘另外保留120 GiB。
- CPU HF导出前预留32 GiB磁盘和56 GiB可用统一内存；它只读DCP模型，不复制数据/初始模型。导出完成逐文件记录SHA256，再验证动作输出投影相对预训练权重确实有变化。
- 真正部署类使用FP32、compile关闭、严格加载导出权重，核对训练过的投影与文件完全一致。此验证不启动服务、动作推理或仿真。

估算试训1份DCP+1份HF约67 GB；正式首轮2份DCP+最终HF约108 GB；两阶段合计约175 GB，保存新点时可能约216 GB。250 GiB上限留了余量。它不是250 GiB预分配，当前尚未生成这些文件。正式首轮只留最近两个完整恢复点和最终HF，不累计保存所有中间大文件。

不保存训练逐帧图片、视频或观测轨迹；官方辅助可视化间隔设置为100万步，当前两阶段不会触发。文本训练日志上限50 MiB/阶段，结构化日志也有上限；TensorBoard和小型尾帧审计保留用于诊断。每次启动记录独立attempt标识、配置签名与代码摘要，训练遥测按attempt区分，恢复不覆盖启动历史。CUDA/Triton等缓存可能继续增长，均指向项目cache；CUDA扩展编译并发上限2。真实训练入口关闭core dump，避免异常时产生巨大内存转储。本轮并行读取失败曾生成约643 MB的系统崩溃报告，核实属于本次测试后已清理，诊断摘要保存在项目内。

本轮安装及检查后，比赛目录约89.73 GB，较本轮前增加约0.22 GB；没有新增训练checkpoint、数据副本或整套模型副本。磁盘可用约2.87 TB。这里的GB采用十进制，预算GiB采用二进制。

## 数据读取修复

`safe_video.py`只用于本训练入口，不改原始视频、Parquet、官方数据类文件或仿真判分。

按每个episode、每个相机的真实视频区间计算有效帧：对于已确认只缺末尾1–2帧的情况，保留状态/动作标签，显式重复最后有效图像，并记录原索引和选用索引。只允许这类有限尾部修复；其他长度错误直接报错，不随机重抽、静默跳过或扩大时间容差。不读取前一段或下一段视频的图像。

解码使用double时间计算和已有PyAV，每个解码器1线程、及时关闭容器。模块级训练数据子类确保spawn出的worker仍使用修复，而不是只在父进程里打补丁。

## 已验证与尚未验证

已验证：训练入口与MoGe/LingBotDepth教师类导入；12项视频/存储/保存间隔策略回归；全部7500个视频区间；52个实际样本（首、中、50任务块末尾，原始读取使用两个spawn worker）；首末样本完整归一化，动作50×55、状态55；官方参数解析器；最终同步配置原生DataLoader的4个micro batch、状态序列化及正常退出（CPU单rank布局桩，未执行FSDP）。完整归一化管线两worker提前退出检查失败，未启用该路径。

尚未验证：GB10上真实训练前向/反向、优化器更新、fused MoE反向兼容性、训练速度/峰值、大规模DCP保存与恢复、HF导出及真实模型重载。上述环节已准备代码，不得因数据检查通过就宣称训练已跑通。

依赖已增量安装在`envs/vla`，保存安装前后清单，保留Torch2.9.1+cu130、NumPy等现有版本。安装了matplotlib及依赖、官方MoGe指定commit的utils3d、lxml、pycryptodomex、optree、numpydantic。LeRobot整包依赖声明与官方固定训练栈仍有冲突，numpydantic1.9也声明NumPy>=2，与现有1.26不符；原安装脚本本身采用`--no-deps`保留固定栈，不能宣称pip check全通过。无ARM wheel且本训练路径不需要的moderngl，以及整套机器人控制/采集依赖未强装，不为消除提示降级Torch或升级NumPy。Torch仍提示GB10的sm12.1高于此版本声明的最高sm12.0；已有推理结果不能替代真实训练反向验证。

冻结范围、训练图像末尾补齐和归一化方法等口径并未获得赛事逐项正式确认，继续保留在审计中；不自行宣称合规或违规。最终方案仍以50任务clean联合训练、一个策略权重为基础。当前仿真基线的41个预检查异常并未在本次修改中处理，后续对比成功率必须保留未完成回合与共同有效seed的口径。

## 准备好的命令（未执行训练）

只检查配置，不训练：

```bash
/home/zzj/competitions/lingbot/envs/vla/bin/python -B \
  /home/zzj/competitions/lingbot/tools/finetune-clean-v3-gb10/launch.py --phase pilot
```

后续明确启动20步试训时：

```bash
/home/zzj/competitions/lingbot/envs/vla/bin/python -B \
  /home/zzj/competitions/lingbot/tools/finetune-clean-v3-gb10/launch.py --phase pilot --start
```

试训完成、导出和真实重载都成功后，正式首轮入口是同一命令改`--phase stage1 --start`。代码会检查试训标记；不会自动从20步跳到2000步。不要直接套官方32卡示例命令绕过资源保护。
