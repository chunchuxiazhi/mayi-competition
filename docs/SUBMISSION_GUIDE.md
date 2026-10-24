# 提交与复现材料说明

团队ID1385264，团队名“连仿真都做不到的话”。结果results/results.json已按[官方模板](official-template/初赛评测结果模板_JSON版.json)填写：clean有效455、成功65；randomized有效423、成功43。原始详细JSON在results/evidence，原始事件及异常未修改。每任务每场景计划10次，不是标准100次。

提交ZIP包含01结果、02代码、03完整2万步HF权重、04复现说明。未参加真机和小红书活动。平台截图显示ZIP上限100000M、剩余3次；约25.52GB完整包低于标示上限，审阅包不含模型张量不能替代完整版。尚未通过天池入口提交。

包内eval.sh调用portable_eval.py；仓库等价入口为code/submission-entry/portable_eval.py，支持--setting clean、randomized或both。默认只打印计划，--check检查实际文件/路径和可选权重SHA，仅--execute启动评测。使用方法见[独立入口说明](../code/submission-entry/README.md)。代码已补齐models/data，直接从提交HF验证，不强制依赖训练DCP和历史运行标记。

包内train.sh调用portable_train.py，仓库入口位于同目录；重定位历史两阶段配置，调用原训练/保存/续训适配。训练数据、官方资产、教师模块和环境外置，按说明准备；新输出与历史实验分离，资源通过路径/符号链接复用。

本次仅做CPU检查和材料修复，未重跑实验。121个专家阻塞槽位和1个初始化异常照实披露；专家阻塞流程差异没有被转换JSON消除。新入口没有动作推理或真实仿真端到端测试，不把静态通过当作新实验完成。最终规则认定和复测成绩由主办方决定。
