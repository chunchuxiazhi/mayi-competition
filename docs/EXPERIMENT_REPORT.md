# 复现与调优说明

## 1. 基础信息

- 团队名称：连仿真都做不到的话；team_id：1385264。
- 使用模型版本：robbyant/lingbot-vla-v2-6b原始预训练模型上，联合微调累计20,000次优化器更新。
- 代码版本：历史冻结版本及文件SHA见training、repository-source-manifest.json；本次新调用文件校验见final-materials-validation.json，最终打包来源Git提交另记于submission_status.json。
- 提交checkpoint：03_模型权重/checkpoint，18个完整HF推理文件、6个FP32权重分片，25,519,988,163字节；不含优化器等完整续训状态。model_manifest.json记录六分片SHA。

## 2. 训练复现

- 数据：RoboTwin2.0 Aloha-AgileX的50任务clean V3，每任务50条，共2500演示、548893帧；未使用randomized或额外rollout训练。
- 一键训练脚本：02/train.sh；使用方法与依赖安装见02/README.md。该脚本显式--execute后调用包内portable_train.py，再调用历史GB10 train_entry.py，不另写训练方法。
- 配置：02/configs/training_step_10000.json、training_step_20000.json；历史冻结配置位于04/training，重定位配置只生成到新workdir。
- robot config：02/configs/original/robots/clean-v3-gb10/robotwin.yaml；归一化：02/configs/original/normalization/robotwin_clean_v3.json，SHA256为0b5ed854065e9c64e1b8934c274c38d0efa062d743bfad071f1a089983125611，来源为允许clean数据，未使用randomized统计。
- 第一阶段10,000步、AdamW、基础lr2e-5；第二阶段完整恢复模型、Adam、loader、RNG，基础lr1e-5到5e-6余弦继续至20,000步。
- micro batch1、梯度累积8、data worker0、pin_memory=false；参数FP32，训练启用混合精度；eager注意力、关闭compile、梯度检查点；冻结Qwen骨干，视觉仍前向，总参数约6.376B，可训练1,938,090,551。
- 原始预训练、Qwen、MoGe-2、LingBot-Depth、DINO-Video来源和用途逐项列在02/README.md。深度、未来深度、视频损失权重各0.004。
- 每1000步同步保存DCP，跨两个阶段只留最新两份；不在训练时并发HF导出。阶段结束后单独CPU导出，不重复复制模型或数据到各运行目录。

## 3. 评测复现

- 一键评测脚本：02/eval.sh，调用02/src/portable_eval.py。默认只输出计划；--check检查路径和文件；仅显式--execute才启动未来复现。
- checkpoint加载：默认本包03/checkpoint；用六个权重分片SHA256直接校验，与训练DCP、原路径、mtime及历史smoke记录脱离。
- clean命令：02/README.md完整命令中的--setting clean，对应demo_clean+seen。
- randomized命令：同一命令中的--setting randomized，对应demo_randomized+unseen；--setting both按顺序运行两者。
- 外部参数：--workdir新运行目录、--vla-python、--sim-python、--qwen-root、已部署官方资产的--robotwin-root，必要时--mplib-lib；默认--test-num 10，允许按需显式设100；--workers 1或2。
- 原环境：ARM64 Linux、GB10，约119.7GiB统一内存；VLA/仿真两套环境，包版本见02/src/environment。内存只计算一份，不把系统与显存相加；clean采样最低可用61.69GiB，最终unseen重测54.61GiB。
- 一份模型服务、最多两worker，会话reset/动作缓存/计数/随机数独立，请求串行、无自动合批；同一checkpoint，无集成、路由、投票、参数更新。
- 环境seed=100000+i；策略seed=1000000+1000*t+i；指令seed=2000000+1000*t+i。失败不换seed；每回合立即追加JSONL并同步。关闭视频、逐帧图片与完整观测轨迹，文本日志有上限。

## 4. 代码改动说明

- LingBot-VLA官方代码及本地运行快照保存在02/src/runtime-source/lingbot-vla-v2；模型/data源码已从实际服务器补齐。训练入口、存储、视频尾帧、归一化、恢复及导出适配位于02/src/tools/finetune-clean-v3-gb10和第二阶段对应目录，实际配置/恢复证据在04/training。
- RoboTwin评测客户端script/eval_policy_client_lingbotvla.py修改原生库导入顺序、实际test_num解析、结果目录和视频开关。动作含义、任务成功判断、官方步数上限、两场景设置保持原版本。
- 评测循环evaluate_task.py、会话session_policy.py和资源/落盘代码来源于实际unseen评测工具，新增调用在portable_eval.py，副本在portable_runtime。新副本修改HF来源校验、路径和调用门槛；没有改写旧实验文件或伪造新smoke通过记录。
- 新训练入口只重定位原两阶段recipe和调用已有适配，保留同步DCP、流式写入和跨阶段保留策略。新接口未启动真实训练验证。

## 5. 调优说明与实测结果

调整重点为GB10统一内存约束下的batch、骨干冻结、数据worker、eager及保存内存峰值；模型规模没有因为冻结而从6B裁剪为2B。使用允许clean数据统计归一化参数，动作chunk50、图像224、三路相机；数据处理细节见冻结配置和训练适配，方法未经主办方逐项确认，不代替主办方判断。

|模型|场景/指令|计划|有效policy|成功|成功/有效|
|---|---|---:|---:|---:|---:|
|原始预训练对照|clean/seen|500|459|43|9.37%|
|10000步对照|clean/seen|500|458|43|9.39%|
|提交20000步|clean/seen|500|455|65|14.29%|
|提交20000步|randomized/unseen|500|423|43|10.17%|

两场景合计878个有效回合、108次成功。clean未执行45个专家阻塞槽位；randomized未执行76个专家阻塞槽位，另有put_bottles_dustbin、seed100000的场景初始化异常。没有把这些槽位补为已完成尝试、成功或普通策略失败。最新官方Q&A明确专家采集失败不应阻止policy，历史实现与其存在已披露流程差异；本次用户选择不重跑实验，差异没有被格式转换或文档修正消除。

本次按官方附件结构填写结果，team_id为1385264；attempts逐任务采用实际有效policy回合数。原详细JSON保留在01/evidence，原始日志在01/raw，官方模板在04/official-source/official-template。每任务每场景计划10次是快速评测，不是100次标准成绩；官方Q&A允许不足100次时按实测填写，以官方复测为准。

20000步clean的有效回合成功率高于两个对照，但有效样本集合不同、每任务次数较少，不据此作显著性或训练步数效果外推；108/878仅是描述，不冒充官方加权成绩。

## 6. 本次材料修正与验证范围

仅修正官方JSON、补齐源码、编写独立调用、更新说明与校验；没有补测121个阻塞槽位，没有训练或加载模型进行动作推理。新入口通过CPU静态和模拟子进程调用测试，未实测新入口的模型/仿真完整链路；既有结果和模型来自实际已完成实验。评委可按02/README在已部署依赖、数据/资产后使用新目录复现。

未参加真机可选任务和小红书活动，材料只含01—04。当前尚未通过天池入口提交；页面标示ZIP上限100000M、剩余提交3次，完整包约25.52GB。
