# 正式续训：累计1000至2000步

用户已批准启动。本次接入真实试训通过的保存补丁，从正式1000步checkpoint恢复，累计2000步保存并停止。原10000步学习率计划保留；没有修改冻结训练配置或数据、模型、输出路径。

## 本次改动

- 部署`dcp_stream_writer.py`和候选`gb10_runtime.py`：TORCH_SAVE减少CPU张量引用保留、串行同步写盘、保存后回收及内存记录。保存实现与刚才真实续跑验证一致。
- 新增`segment_stop.py`：组合原有严格恢复保护，核实恢复步数、Adam、调度器和随机数；仅把训练循环的停止判断限制为累计2000步，保留原10000步参数供学习率调度器构造。到2000步使用官方保存流程，再正常退出。
- 更新`train_entry.py`和`launch.py`以传递独立停止控制。原`resume_guard.py`未改。
- 达到2000步后，父进程核查完整checkpoint、实际更新记录和scheduler步数；不自动导出HF权重、不启动评测、不自动重启。
- 30项CPU检查通过；实际官方训练main组合AST编译通过；正式配置dry-run通过。

## 不变配置

- 50任务V3 clean联合训练；micro-batch=1，梯度累积=8，有效batch=8。
- FP32、eager、compile关闭；参数冻结和学习率设置不变。
- 原`max_steps=10000`，每1000步保存，保留最近两份完整checkpoint。独立本段终点为2000，不是重建一个2000步学习率计划。
- 冻结JSON/YAML SHA256均为`ffcebfebe757c20c1ac64015eadeb7142f0398b34e6276ab0d953081cd625f33`，部署时保持字节不变。
- 继续保留16 GiB统一内存停止保护。CPU/GPU共用物理内存，不重复相加。

## 启动命令（已执行）

```bash
/home/zzj/competitions/lingbot/envs/vla/bin/python -B \
  /home/zzj/competitions/lingbot/tools/finetune-clean-v3-gb10/launch.py \
  --phase formal10000 --stop-at-step 2000 --expected-resume-step 1000 --start
```

后台父进程PID：96865。独立session，stdin断开；关闭本地会话不应结束后台训练。

## 保存和占用

正式运行目录：`/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/clean50-10000-fixed1000`。

本段预计新增一份约41.02 GB DCP，完整保留模型和优化器；原1000步checkpoint保留。没有额外HF导出约25.5 GB。此前两份试训验证checkpoint已删除。

## 记录

- 部署及启动授权：`/home/zzj/competitions/lingbot/docs/training-prep-20261016/resume1000-to2000/`。
- 原工具文件有小型备份；安装manifest记录所有补丁哈希。
- `segment-resume-verified.json`记录实际恢复检查；`logs/updates.jsonl`追加新更新。
- `logs/save-memory.jsonl`记录保存阶段；`execution-status.json`记录后台状态。

本文件在启动阶段建立；实际新更新确认另存`resume2000-startup-confirmed.json`。
