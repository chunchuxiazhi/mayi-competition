# 20,000步模型 randomized＋unseen评测

本目录对应已完成的真实 unseen重测：同一份20,000步权重，`demo_randomized`＋`unseen`，官方50任务、每任务10个计划回合。最终423个有效策略回合、43次成功，成功率10.17%；76个专家阻塞槽位和1次场景初始化异常另列。

## 入口与执行流程

`rerun.py`只组织 `config-randomized.json`指定的单场景评测。默认只检查；历史后台启动命令为：

```bash
python3 -B /home/zzj/competitions/lingbot/tools/randomized-unseen-rerun-20261024/rerun.py --start-approved --background
```

该运行已结束，已有 supervisor标记会拒绝重复启动。新实验须使用独立配置与输出。原 clean＋seen记录保存在 `results/step20000-clean`，不由本入口重跑。`run_suite.py`、`config-clean.json`作为来源代码快照保留，当前操作入口为 `rerun.py`。

流程先执行两会话真实技术小测，检查 reset隔离、动作格式、实际 unseen指令和结果落盘，通过后等待端口释放并继续50任务。两次小测纳入计划500回合。两路 worker共享一个 FP32/eager服务，关闭 compile，请求串行处理，不自动合批。

## 指令与回合记录

配置校验、客户端传参和官方指令生成均按 setting选择指令类型；此入口固定 randomized使用 unseen。每个有效回合保存实际指令、候选数及候选池 SHA256；缺少 unseen任务指令时明确报告异常，不从 seen任务指令池替代选择。

策略与专家事件分别追加到 JSONL并同步落盘。正常失败计入有效回合，异常和中断独立记录；恢复按任务及回合索引去重，失败不另选 seed。35项 CPU回归检查与真实小测均有记录。

## 复现边界

原服务器路径、模型来源和代码校验用于追溯本次运行。新机器需要外部模型/数据/仿真资产、Python环境，并重新适配来源检查和小测。公开 HF不含 DCP状态，独立部署的 HF-only校验入口尚待验证。

专家元数据依赖仍造成部分槽位阻塞，与官方“采集失败后继续 policy测试”的要求存在流程差异。正式 JSON字段模板亦待核实。上述情况见 `docs/REPRODUCTION.md`。原导出证据 ZIP保持运行时字节，当前 README为实验结束后的说明。
