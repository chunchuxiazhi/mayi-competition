# 20000步模型 randomized+unseen 更正评测

入口`rerun.py`只针对`config-randomized.json`，同一20000步权重、demo_randomized+unseen、50官方任务×10计划回合、两会话共享同一串行推理服务。FP32/eager，关闭compile，无自动合批、集成或测试时参数更新。默认仅检查；已有历史运行不可重复启动覆盖。

实际原服务器已授权启动的命令：

```bash
python3 -B /home/zzj/competitions/lingbot/tools/randomized-unseen-rerun-20261024/rerun.py --start-approved --background
```

入口先完成匹配的两会话真实技术小测，通过后等待端口释放并继续50×10；两次小测纳入计划500，不重复计数。每回合即时追加JSONL并fsync，保存实际指令及unseen候选池SHA256；中断、专家阻塞、策略异常独立记录，失败不另选seed。

423有效策略回合、43成功；76专家阻塞、1场景初始化异常，不能说完成500有效回合。原clean+seen记录保留在results/step20000-clean，不由本入口重跑。`run_suite.py`、`config-clean.json`等作为来源代码快照保留；当前单场景操作使用rerun.py，不能将旧双场景帮助入口当作新机器一键安装器。

外部数据、仿真资产、模型、Python二进制环境和训练DCP不在源码仓库。换机器需部署依赖、适配路径和来源校验、重新做小测，不伪造DCP占位文件满足检查。官方模板及元数据阻塞差异见docs/REPRODUCTION.md。当前README为更正后说明，运行时的原始证据ZIP保持字节不变。
