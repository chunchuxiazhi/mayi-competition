# 环境部署与实验复现

本说明对应最终20,000步模型及其 clean＋seen、randomized＋unseen 评测。原始预训练和10,000步 clean 结果用于对照。实验配置、原始日志与统计文件分别保留，文字校订不改变历史记录。

## 环境与外部资源

原部署根目录为 `/home/zzj/competitions/lingbot`，运行于 ARM64 Linux 和 NVIDIA GB10，统一内存约119.7GiB。模型与仿真分别使用 `envs/vla`、`envs/sim` 两套 Python 环境；[environment](../environment)记录已安装包版本。这些清单用于环境核对，并非适用于任意平台的依赖锁文件。

另需按官方来源准备指定预训练模型、Qwen 依赖、训练所需公开教师模块、允许的 clean V3数据、RoboTwin 完整物体/机器人资产、纹理和GPU运行环境。训练使用的教师权重与路径见冻结配置；没有使用官方 RoboTwin 后训练模型作为初始化权重。数据、大型资产、环境二进制和缓存不包含在普通 Git 仓库或提交 ZIP 中。

## 目录映射

|仓库路径|原服务器对应位置|说明|
|---|---|---|
|`code/tools/`|`tools/`|本项目训练、评测与保存恢复适配|
|`code/configs/`|`configs/`|机器人映射、归一化与训练配置|
|`code/runtime-source/`|`repos/`|LingBot与RoboTwin文本源码快照|
|`training/step_10000`、`training/step_20000`|对应训练运行目录|冻结配置与恢复/导出核验证据|
|`results/`|对应评测运行目录|原始记录、计划及汇总|

源程序中的绝对路径指向原服务器。新机器需要重新部署依赖、调整路径并建立新运行目录，重新生成输入计划、来源检查和真实小测证据。旧记录中的路径与指纹用于追溯原实验，不应改写为新机器已经验证的证据。

## 训练与权重恢复

第一阶段工具为 `code/tools/finetune-clean-v3-gb10`，第二阶段为 `code/tools/finetune-clean-v3-gb10-continue20000-20261018`。原部署的配置检查命令为：

```bash
ROOT=/home/zzj/competitions/lingbot
$ROOT/envs/vla/bin/python -B $ROOT/tools/finetune-clean-v3-gb10/launch.py --phase formal10000
$ROOT/envs/vla/bin/python -B $ROOT/tools/finetune-clean-v3-gb10-continue20000-20261018/launch.py --phase formal20000
```

以上不带 `--start`，只执行相应检查。实际训练还需完整数据、教师资源及恢复证据；已有训练输出不能覆盖重跑。第二阶段从10,000步完整训练状态恢复，新增10,000次更新，累计达到20,000步。

公开 HF 导出用于推理，不包含 Adam 状态、数据加载器游标或 RNG 状态。10,000步完整续训状态已按保留策略清理；20,000步完整状态保留在原服务器，不在本次发布范围。因此，下载 HF 文件后可部署推理，但不能据此恢复完整训练状态。权重下载方法见[模型说明](../models/README.md)。

## 评测入口与结果来源

最终 clean 记录位于 `results/step20000-clean`；实际 unseen 重测位于 `results/step20000-randomized-unseen`。两组使用同一份20,000步 HF 权重，合并方式及来源校验值见 `results/aggregation-manifest.json`。

unseen 原部署入口默认只检查：

```bash
python3 -B /home/zzj/competitions/lingbot/tools/randomized-unseen-rerun-20261024/rerun.py
```

`rerun.py`仅组织 randomized＋unseen。显式开始后，先执行两会话技术小测，通过后等待端口释放，再继续50任务评测；小测回合计入计划500回合。已有 supervisor 标记会阻止重复启动。底层 `driver.py`保留单 worker 选项，但单路复现仍需匹配的新配置、输出与小测证据。

两路 worker 共享一份常驻模型，服务不自动合批；会话的 reset、动作缓存、计数和随机数状态分开保存。场景 seed、策略 seed、指令 seed和实际指令写入逐回合日志，策略失败不另选 seed覆盖结果。关闭视频、逐帧图片和完整观测保存，结果按回合追加并同步到磁盘。

原来源检查还引用训练 DCP 的恢复与导出证据。独立部署只有 HF 权重时，需单独适配并验证来源检查入口；本项目尚未完成新机器从零安装的一键复现实测。原始评测证据 ZIP保留运行时字节，其中准备阶段说明应结合冻结配置和事件日志解读。

## 统计与规则对应

有效策略回合指产生 `policy_completed` 终结记录的回合；任务成功或达到官方步数上限后的正常失败均计入分母。专家阻塞、初始化异常和中断另列。最终两组有效878回合、成功108次，尚有121个专家阻塞槽位和1个初始化异常。

官方 Q&A 明确：clean2clean 使用 seen，clean2random 使用 unseen；专家五次采集配额按发起次数计算，与 seed无关，采集失败不应影响 policy测试。现有预检查/元数据依赖造成的阻塞仍是流程差异，本项目没有把这些槽位记成已完成策略回合。clean归一化数据来源有技术证据，统计方法尚未得到赛事方单独确认。

每任务10个计划回合是算力与时间约束下的快速评测。官方允许测不足100次时按实测填写、以官方复测为准。正式 JSON模板仍需登录取得并转换；目前的 `quick_results_evidence.zip`是可校验的实验记录包，不能替代官方格式核对。
