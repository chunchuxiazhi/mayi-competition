# 复现范围和依赖

本仓库归档GB10单机上的50任务clean联合微调、四组快速仿真评测及实际适配源码。原始预训练来源为`robbyant/lingbot-vla-v2-6b`，未使用官方RoboTwin后训练权重。

## 文件用途

- `code/tools/finetune-clean-v3-gb10`：第一阶段训练、导出和检查工具；`finetune-clean-v3-gb10-continue20000-20261018`：10000至20000步续训、完整恢复与保存适配。
- `code/tools/clean-quick-baseline-20261016`、`clean-checkpoint-eval-20261017`：预训练与10000步clean评测。
- `code/tools/checkpoint-dual-setting-eval-20261023`：20000步clean/randomized评测、结果保存和导出。
- `code/tools/dual-setting-dispatch-20261024`：两个setting顺序执行前的端口释放等待，避免TIME_WAIT误报；默认仅检查。
- `code/runtime-source`：归档时实际安装的Lingbot与RoboTwin文本源码/配置快照，保留原许可；不包含仿真大型资源或训练数据。
- `code/configs`：实际训练、机器人和clean归一化配置；`training`：两个阶段的冻结配置、输入选择和恢复/加载证据。
- `environment`：VLA与仿真两个Python环境的安装包版本。它是环境记录，不是可直接运行的通用依赖锁文件。

## 环境与数据

原服务器是ARM64 Linux + NVIDIA GB10，统一内存约119.7GiB。模型服务FP32/eager、关闭compile；每个setting两个仿真worker共享一份服务，不自动合批。训练冻结Qwen骨干（视觉仍参与前向），约1.938B参数可训练；有效batch=8，微批1、累积8，50任务clean V3各50条演示。

完整复现仍需自行取得官方预训练基础模型、Qwen依赖、允许的clean V3数据、RoboTwin仿真资源和seen/unseen场景纹理。数据和模型缓存不在普通Git仓库中。源程序中保留的绝对路径指向原服务器`/home/zzj/competitions/lingbot`，不是下载者电脑上的路径。换服务器时需重新部署目录、适配配置并重新生成输入与smoke证据，不应直接改写历史回合记录的路径或指纹来冒充已验证。

原始项目目录结构中的`code/tools`和`code/configs`应对应服务器项目的`tools/`、`configs/`，两套`runtime-source`对应`repos/`。冻结配置和历史来源记录保留原样供审查，不能保证一次pip install便可跨平台运行。

## 训练与权重

10000步和20000步的HF推理导出各约25.52GB，模型配置、tokenizer和processor完整；大张量由Release分片分发，下载方式见`models/README.md`。这些导出不含Adam状态、数据加载器游标或RNG续训状态，不能冒称完整断点恢复checkpoint。10000步原始续训状态已按此前保留策略清理；20000步续训状态仍在服务器，但依用户要求不上传。

当前上传只含这两份推理模型；不上传19000步、20步试训、原始预训练权重、优化器、数据集、缓存或视频。

## 结果解释

每组均为每任务10个计划回合，而非100次标准评测。分母采用实际有效策略回合；专家阻塞、异常和历史中断分别记录，失败不换seed重跑。归一化clean来源有技术证据，统计方法与专家配额适用范围仍须官方确认；不自行声明已获官方逐项认可。

原始结果和SHA256应作为可审计记录保留。新服务器重新评测应新建运行目录。`quick_results_evidence.zip`是快速评测证据包，并不等于已符合官方附件schema的正式提交包。

最终正式提交仍需核对比赛官方次数、模板、匹配权重与复现说明要求。本GitHub归档不代表已在天池完成提交。
