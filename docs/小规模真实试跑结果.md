# 20000步模型：双场景真实小规模试跑

执行日期：2026-10-23（北京时间）。使用既定global_step_20000权重，FP32、eager、关闭compile；每种场景一份模型服务、两个独立会话worker。未启动50任务完整快速评测或训练。

## 结果

|setting|任务|有效回合|成功数|动作步数|策略阶段耗时|
|---|---|---:|---:|---:|---:|
|demo_clean + seen|lift_pot|1|0|400|52.79秒|
|demo_clean + seen|hanging_mug|1|0|900|74.75秒|
|demo_randomized + seen|lift_pot|1|0|400|47.15秒|
|demo_randomized + seen|hanging_mug|1|0|900|75.89秒|

四个回合全部正常执行到官方步数上限，未完成任务。没有策略异常、专家阻塞或未收尾回合，失败记录保留。技术检查通过不等于任务成功；四次样本不能推断整体成功率。它们计入后续计划回合，正常失败不重跑、不换seed。

20000步模型CPU导出及严格加载验证通过，导出权重25,503,889,812B（约25.5GB），两个setting共用。两个setting的smoke均验证了首个动作、两个独立session，以及两路reset完成后再推理。

## 资源和保存

- clean采样最低可用统一内存67.105GiB；randomized为65.220GiB，均远高于16GiB保护下限。统一内存只算一份。
- 主机load采样约1—2.2，有效CPU20；未观察到CPU或内存资源保护触发。该结果不代表全部50任务的资源峰值。
- 缓存累计增长851,968B，约0.81MiB；结束后磁盘可用约1.63TB。
- 结果根目录实际文件总量在导出后约1,312,388B（约1.31MB，不包含25.5GB模型导出）；本报告和验证记录会增加少量元数据。
- 原始账本四个任务-setting各一个完整策略回合，无重复终态；汇总有效回合4、成功0。生成JSON、CSV和44个文件条目的校验证据包，ZIP逐文件SHA256等核验通过。
- 模型服务及仿真worker已全部退出，GPU资源释放；当前状态smoke_passed_paused。

## 发现的启动问题

第一轮clean完成后立即切换randomized，在相同端口19360的bind检查处报Address already in use。该次未启动randomized策略回合，所有子进程随后退出。稍后核查端口已空闲，符合服务退出后短暂端口残留的情况，未在错误发生瞬间抓到socket状态，因此不将TIME_WAIT作为已确证根因。

手动继续同一smoke入口后，已匹配的clean检查被跳过，randomized首次两个策略回合正常完成。模型、配置、seed、代码均未修改；未重跑任何正常失败回合。

**连续启动两个setting的端口切换逻辑仍需修复后再开始无人值守完整评测。**本次没有通过改写代码指纹、替换seed或覆盖clean结果绕过门槛。真实策略、渲染和保存路径已跑通，连续调度的这一问题必须单独处理。

## 文件位置

服务器结果根目录：
`/home/zzj/competitions/lingbot/runs/finetuned-step20000-clean-randomized-50x10-20261023`

逐场景证据：`clean/smoke-result.json`、`randomized/smoke-result.json`。

最终核查：`preparation/small-test-verification.json`。

导出证据目录：`exports/quick-dccb03f3610f1bb7`，包含`quick_results.json`、`per_task.csv`、`submission_readiness.json`、`manifest.json`、`quick_results_evidence.zip`。

本地机器可读副本：`small-test-evidence.json`。此处没有复制模型或整套数据。

该ZIP仅为4个有效回合的小规模试跑证据，不是50任务完成结果，更不是各100次的官方完整成绩或已验证可接收的正式提交包。
