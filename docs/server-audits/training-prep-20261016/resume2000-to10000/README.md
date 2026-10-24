# 正式续训：累计2000至10000步

用户已明确授权继续到累计10000步，正式checkpoint只保留最新两份。

## 执行设置

- 从同一正式运行目录的`global_step_2000`完整恢复，新增8000次优化器更新，到10000步保存并停止。
- 原冻结配置、10000步学习率计划、batch、精度、数据和输出路径均不改。
- 每1000步保存一份完整DCP，包括模型、Adam、调度器、DataLoader和随机数状态。
- 保存补丁沿用已通过真实试训与正式2000步保存的版本，本轮未修改训练Python文件。
- `keep_checkpoints=2`：先写完、校验并提交新checkpoint，再删除超过两份的旧正式checkpoint。正常结束时保留9000和10000步。
- 16 GiB统一内存保护保留；单路训练，无自动重启、HF导出或仿真评测。

## 存储和时间

每份正式DCP约41.02 GB。稳定保留两份约82.04 GB；新文件写入完成、旧文件删除前，正式checkpoint临时约123.05 GB。磁盘及项目训练目录预算检查通过。该保留策略作用于正式训练运行目录；本轮未额外清理历史pilot或原始模型、数据。

按前一段约9秒/步及约100秒/次保存的实测推算，8000步加8次保存约20–22小时。实际时间受运行状态影响。

## 后台命令（已执行）

```bash
/home/zzj/competitions/lingbot/envs/vla/bin/python -B \
  /home/zzj/competitions/lingbot/tools/finetune-clean-v3-gb10/launch.py \
  --phase formal10000 --stop-at-step 10000 --expected-resume-step 2000 --start
```

父进程PID：131351，独立session，stdin已断开，日志写入服务器文件。

运行目录：`/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/clean50-10000-fixed1000`。

授权、dry-run、磁盘预算和启动记录：`/home/zzj/competitions/lingbot/docs/training-prep-20261016/resume2000-to10000/`。

原冻结JSON/YAML SHA256：`ffcebfebe757c20c1ac64015eadeb7142f0398b34e6276ab0d953081cd625f33`。

启动阶段建立本文；实际恢复及新更新确认保存在`resume10000-startup-confirmed.json`和服务器的`startup-confirmed.json`。
