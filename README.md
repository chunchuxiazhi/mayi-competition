# mayi-competition
蚂蚁灵波具身智能比赛的相关材料

归档比赛指定原始预训练模型`robbyant/lingbot-vla-v2-6b`的GB10适配、50任务clean联合微调，以及预训练、10000步和20000步模型的仿真快速评测。未使用官方RoboTwin后训练模型。

## 快速评测结果

每组覆盖50任务，每任务10个计划回合。clean使用seen；20000步randomized使用重新实测的unseen。成功率按实际有效策略回合计算；专家阻塞和异常单独记录。

|模型|场景|计划槽位|有效回合|成功|有效回合成功率|专家阻塞|策略异常栏|
|---|---|---:|---:|---:|---:|---:|---:|
|原始预训练|clean|500|459|43|9.37%|41|0|
|10000步|clean|500|458|43|9.39%|42|0|
|20000步|clean|500|455|65|14.29%|45|0|
|20000步|randomized + unseen|500|423|43|10.17%|76|1|

20000步两场景合计878有效回合、108成功。原randomized+seen实验已从当前材料中移除，替换为真实unseen重测，未把旧结果简单改标签。clean结果逐字节保留。唯一记入策略异常栏的情况实际发生于场景初始化时物体不稳定；不能等同于模型推理崩溃。预训练组保留1次历史中断记录，但无未收尾回合。不同组有效回合集合有差别，表格是汇总比较，不冒充严格配对结论。

这是每任务10次计划的快速评测，未在天池提交。官方最新Q&A说明受算力或时间限制时可以按实测次数提交，最终以官方复测为准。121个专家阻塞槽位及1个场景异常没有产生有效策略结果，不能把1000个已处理槽位称为1000个有效回合。最新Q&A要求专家采集失败不阻止策略测试，现有元数据依赖仍存在未解决的流程差异，见复现说明。

逐任务对比见[comparison-per-task.csv](results/comparison-per-task.csv)，总表见[summary.csv](results/summary.csv)。合并结果见[results.json](results/results.json)，其源文件SHA见[aggregation-manifest.json](results/aggregation-manifest.json)。clean与unseen原始逐回合、专家检查、实际指令、seed、步数和异常均保存在[results](results)。JSON仍是本项目的quick导出，正式官方字段模板尚待登录取得并转换。

## 模型权重

- [10000步Release](https://github.com/chunchuxiazhi/mayi-competition/releases/tag/lingbot-step-10000)：约25.52GB的完整HF推理导出。
- [20000步Release](https://github.com/chunchuxiazhi/mayi-competition/releases/tag/lingbot-step-20000)：约25.52GB的完整HF推理导出。

每份含完整模型、tokenizer、processor和配置，大张量分片均小于2GiB，附SHA256。Release只有在全部资产完成校验后才发布；如果链接尚不可见，表示模型归档还未发布，不能仅凭仓库清单视为已上传完成。

下载与恢复见[models/README.md](models/README.md)。本仓库只归档上述两份推理权重，不含优化器等完整续训状态，不含其他训练步数的权重。

## 复现文件

- [code](code)：训练、评测、会话隔离、保存和端口等待代码，及实际安装的上游文本源码快照。20000步评测入口已替换为[randomized-unseen-rerun-20261024](code/tools/randomized-unseen-rerun-20261024)。
- [training](training)：10000/20000步阶段冻结配置、选择的clean演示索引、恢复和加载验证记录。
- [environment](environment)：VLA和仿真两个环境的包版本。
- [docs/REPRODUCTION.md](docs/REPRODUCTION.md)：目录映射、资源需求和复现限制；[docs/server-audits](docs/server-audits)保留适配审查和原始检查报告。
- [archive-source-manifest.json](archive-source-manifest.json)：从服务器取回的原始文件路径、尺寸和SHA256。归档区禁用Git换行转换，服务器快照字节可直接核验；后续修改应另行记录，不能覆盖历史来源指纹。

原服务器为ARM64 Linux + NVIDIA GB10。模型服务采用FP32/eager、关闭compile；两路worker共享同一模型，推理串行，无自动合批。模型总规模约6.376B，训练冻结Qwen骨干、约1.938B可训练；冻结视觉权重不代表关闭视觉前向。训练数据为50任务clean V3各50条演示，有效batch=8（微批1、梯度累积8）。

不上传训练数据、仿真大型资产、依赖环境、缓存、帧图片或视频；在新机器上仍需按官方来源取得这些依赖。源记录保留原服务器绝对路径，需要重新部署和验证，不能视为开箱即用的跨平台安装包。

上游源码来源及许可见[licenses](licenses)。归一化统计方法尚未获官方单独确认；专家采集与策略测试分别计数，未解决的阻塞如实披露，不自行宣布已获赛事逐项确认。
