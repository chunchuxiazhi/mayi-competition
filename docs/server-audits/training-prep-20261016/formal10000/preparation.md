# 10000步正式训练准备

本文件为上一份“全部改动与10000步方案”之后的增量记录。当前仅完成工程准备和检查，**尚未开始10000步训练**。未把同伴审核或赛事规则确认标记为已完成。

## 固定配置

| 项目 | 值 |
|---|---|
| 初始化 | 指定原始`robbyant/lingbot-vla-v2-6b`，不接续功能试训权重 |
| 数据范围 | 合并V3 clean，50任务联合训练一个策略，2500段、548893帧 |
| 总更新数 | 10000 optimizer steps |
| 单次micro batch / 梯度累积 | 1 / 8，全局batch 8 |
| 动作chunk长度 | 50 |
| 模型与并行 | 1个训练rank、1份模型，0个数据worker |
| 精度与注意力 | FP32模型，教师沿用原BF16 autocast；eager；关闭torch.compile |
| 可训练范围 | 冻结Qwen骨干，动作专家及投影/对齐模块约1.938B元素训练 |
| 优化器与LR | AdamW，峰值2e-5、最低1e-5、cosine |
| warmup | 3%，即300步，首个训练step的LR为0属于原warmup实现 |
| 训练seed | 42 |
| 保存模式 | **固定间隔**，1000、2000……10000共10次 |
| 保留策略 | 新点提交完整后，只留最新和次新，最终9000和10000 |
| HF导出 | 训练进程结束后，仅最终一次，再单独严格重载验证 |
| 自动评测/自动续长 | 关闭；不会自动加到更多步数或启动仿真 |

10000步累计采样80000个帧起点；每样本含50步动作窗口，不代表遍历全部数据10000次，也不保证10k足够收敛。此前20步试训仍保留。

## 代码和配置增量

1. 新增`configs/training/clean-v3-gb10/formal10000.yaml`和工具目录中的`formal10000-control.json`；保留原pilot和2000步stage1配置。
2. `storage_policy.py`新增固定保存模式和保存节点计算；非整除时也要求最后一步保存。1000/5000两种计划分别验证为10/2次保存。
3. `launch.py`新增`formal10000`阶段，使用固定1000间隔，不再进入约30分钟自动校准；核对预期节点，要求真实零更新恢复验证成功后才能正式启动。
4. 独立正式运行目录：`/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/clean50-10000-fixed1000`。准备阶段不在这里写训练回合或checkpoint。
5. `resume_guard.py`新增严格恢复边界：按已校验的官方main结构，在内存AST中只替换恢复块的异常处理，并在恢复后/训练循环前插入检查边界。源码文件未就地修改；原损失和训练循环保留。上游结构不符合预期直接拒绝。
6. 原先只保护`Checkpointer.load`不足以覆盖后面的scheduler/DataLoader恢复错误。现在任何该块内的恢复失败均停止，禁止自动退回旧点或从头训练。
7. `verify_training_resume.py`使用现有20步checkpoint，真实恢复模型、Adam、scheduler、随机数和DataLoader，并在训练循环前退出；不做模型前向/反向、不保存新点。
8. 本机PyTorch DCP会用零梯度、零学习率的内部`optimizer.step`初始化空Adam状态。首轮验证被训练计数钩子误判而中止，已保留失败日志。`gb10_runtime.py`现显式区分恢复阶段，只有LR确为0才忽略该内部初始化的训练计数；非零LR仍报错。CPU测试确认即使设置weight decay，零LR初始化也不改变测试权重。
9. `test_resume_guard.py`覆盖恢复错误不回退、训练边界顺序、源码结构变化拒绝、内部零LR初始化不计入训练及非零LR拒绝；连同原策略测试共22项。
10. `validate_native_loader.py`增加正式配置选项，实际检查8个micro batch及状态序列化，无模型训练。
11. `start_detached.py`准备显式`--start`后台入口，检查就绪清单/源码/配置摘要，防止同一运行重复提交。它的生成和检查不会启动训练。

## 检查结果

- 16项视频/存储/固定保存策略测试及6项恢复保护测试通过，共22项。
- 正式配置原生解析通过：10000步、固定1000保存、累积8、300步warmup。
- 正式配置CPU数据加载实测通过：8个micro batch，每个actions为1×50×55，状态可序列化、正常退出。
- 真实20步checkpoint恢复通过：959组Adam状态均恢复到step20，模型动作投影与已导出的权重完全相同，scheduler及Python/NumPy/Torch CPU/CUDA随机数恢复。
- DataLoader恢复后下一批数据重放一致。没有执行模型前向/反向或新增训练更新；原20条更新记录及唯一checkpoint文件未变。
- PyTorch内部零LR/零梯度状态初始化发生1次，已与训练计数隔离，未产生新训练步骤；首次失败日志保留。
- 上游trainer和clean统计哈希通过检查；六个指定原始模型分片大小核对通过，沿用下载时已有SHA256验证记录，本轮没有重新计算25GB全量文件摘要。
- 源码及配置哈希与服务器准备清单一致；不带`--start`的后台入口拒绝运行，正式输出目录尚未创建。
- 就绪核查时间UTC：2026-10-16T05:11:37.970225+00:00。当时GPU无计算客户端，可用统一内存约109.1 GiB，磁盘可用约2.80 TB。实际启动时会重新检查，不代表资源被预留。
- 比赛目录当前分配约156.56 GB；本次恢复检查只新增小型日志和证据，没有新增41GB checkpoint。
- 就绪证据：`docs/training-prep-20261016/formal10000/readiness.json`；恢复证据：`runs/finetune-clean-v3-gb10/resume-verification/{result,status}.json`。

当前配置摘要：`377cd610f4c7f16a5ab46041302574c6bb8024bcf61c1932da34a6811c5e1c28`；控制文件摘要：`1727cd2c22f25a9f6553b205d923601f77a362550ecc5af674adcca305b837e2`。

真实恢复验证只证明当前20步checkpoint与本机1rank恢复路径，不保证任意未来磁盘故障、任意版本或任意训练方法均可恢复。正式配置累积8的CPU数据路径已检查；其GPU训练速度仍由累积4实测外推，未为准备工作额外启动训练。

## 内存、磁盘与时间

1. GB10约119.7 GiB统一物理内存，CPU/GPU占用不重复相加。单rank、micro=1、梯度检查点、冻结骨干及同步数据读取保持不变。
2. 启动前再次检查GPU无其他计算客户端、MemAvailable>=90 GiB；CUDA分配器上限72 GiB，系统可用低于16 GiB停止；保存前要求至少28 GiB可用。这些不是全进程硬配额或无OOM保证。
3. DCP按实测41.02 GB/份估算；正式运行最终2份约82.04 GB，最终HF另约25.52 GB，合计约107.56 GB。
4. 新点提交后才清理旧点，训练时可能短暂3份DCP约123.05 GB；最终HF在训练进程结束后导出，避免同时常驻模型，也避免训练中每个点额外导出25 GB。
5. 连同pilot的训练root预计稳态约174.11 GB、短暂约189.61 GB，低于现有250 GiB预算检查。预算覆盖pilot和本轮，不把原始数据/模型复制进去。
6. 保留磁盘额外120 GiB，保存前按64 GiB预留新checkpoint；关闭core dump、视频、逐帧图像、完整观测。编译和驱动缓存仍可能增长，均位于项目cache。
7. 试训累积4稳定中位4.871秒/更新。累积8按线性估算9.742秒/更新，10000步纯训练约27.1小时，安排约28–32小时供训练/保存/导出，评测及故障恢复另外留时间。保存10次按实测约6.4分钟总开销。这些是估算，不是完成时间保证。

## 已准备的命令

只检查配置：

```bash
/home/zzj/competitions/lingbot/envs/vla/bin/python -B \
  /home/zzj/competitions/lingbot/tools/finetune-clean-v3-gb10/launch.py --phase formal10000
```

后续明确开始时，后台启动入口（**本轮未执行**）：

```bash
/home/zzj/competitions/lingbot/envs/vla/bin/python -B \
  /home/zzj/competitions/lingbot/tools/finetune-clean-v3-gb10/start_detached.py --start
```

运行状态保存在新run的`execution-status.json`、`logs/updates.jsonl`、`logs/checkpoints.jsonl`和`logs/train.log`；启动时记录PID、冻结配置和代码哈希。

中断后必须复用同一冻结配置及原运行目录，通过`launch.py --phase formal10000 --start`入口恢复，并由后台进程托管；不能直接重复首次`start_detached`请求，也不能把max_steps/LR改掉当作同一实验续跑。内存/保存/恢复错误停止，不自动无限重试。已完成10000步则拒绝多更新一步。

## 仍待确认的赛事口径

之前汇总中的eager/SDPA、骨干冻结及训练方法、clean统计、缺尾帧处理、SAPIEN/OIDN版本差异、专家配额与评测控制流等仍保留待同伴/官方核对。工程准备通过不等于这些规则问题已获批准。最终参赛仍按规定50任务clean联合训练同一个策略。

本次没有更改原始模型、数据、旧基线结果或20步checkpoint；正式训练保持未启动。
