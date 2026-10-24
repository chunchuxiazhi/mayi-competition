# 20000步模型：50任务×clean/randomized各10回合

本轮只准备代码、配置和1000个确定的计划槽位，执行语法、配置、路径、计数、结果隔离与启动控制检查。2026-10-23代码与资源复查修复了动作块越过步数上限、日志写入故障、持久化与导出等问题，最终32项CPU测试通过，详见《代码与资源检查报告》。**尚未导出或加载20000步模型，尚未运行仿真smoke或评测。**静态检查通过不代表已经通过实际仿真验证。

## 代码与目录

- 服务器工具：`/home/zzj/competitions/lingbot/tools/checkpoint-dual-setting-eval-20261023`
- 结果根目录：`/home/zzj/competitions/lingbot/runs/finetuned-step20000-clean-randomized-50x10-20261023`
- `clean/`：`demo_clean + seen`，50任务×10计划回合。
- `randomized/`：`demo_randomized + seen`，50任务×10计划回合。
- 两个子目录各有固定的`episodes-plan.json`、配置、来源记录、每任务结果和汇总；根目录保存双场景汇总。
- 单份推理导出：`/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/clean50-20000-from10000-20261018/exports/global_step_20000/hf_ckpt`

两个setting依次运行，每种内部最多两路worker。一个模型服务共享一份FP32权重，推理请求串行处理，没有自动合批。保留已有的会话私有缓存、步数、FeatureTransform、Python/NumPy/Torch CPU/CUDA RNG隔离；原`session_policy.py`未改动。保留单路重放入口，不准备四路。

## 文件修改说明

|文件|作用|
|---|---|
|`config-clean.json`、`config-randomized.json`|固定20000步来源、50任务、每任务10次；分别使用官方clean/randomized配置，语言均为seen；关闭视频、图片和完整观测轨迹|
|`validation.py`、`static_check.py`|核对官方launcher中完整50任务列表、重复、checkpoint来源/hash、客户端可配置测试数、随机纹理编号、输出路径及配置一致性；未导出权重时明确报告runtime_ready=false|
|`prepare_inputs.py`|冻结每回合seed与setting；clean与原10000步clean测试使用可配对的指令输入；randomized独立生成并记录官方指令，不把clean指令强套给随机场景|
|`model_source.py`、`prepare_model.py`|为未来显式执行准备CPU导出及单独严格加载验证；共享一份导出，校验20000步checkpoint、训练配置、文件hash及实际部署加载证据|
|`evaluate_task.py`|场景、训练步数、策略标签改为配置驱动；实际回合成功率使用有效完成数；禁止原生尾部文本导出误用计划数量作有效回合分母；官方判据、动作、步数上限不变|
|`common.py`|逐回合fsync保存，保留成功、失败、异常、中断与专家阻塞；重复终态/混合setting立即报错；成功率用实际有效回合数；结构化账本每文件2MiB上限|
|`driver.py`|复用已验证的一份模型/两路worker调度；显式区分smoke与全套；新权重和每个setting都必须先有匹配smoke证据；子进程日志有上限|
|`run_suite.py`|默认只做静态检查；模型准备、smoke、完整快速评测分别需要显式参数，互不自动串联；顺序运行两种setting并汇总，可后台执行，不自动重试|
|`test_preparation.py`|15项纯CPU控制和计数测试，不导入模型或仿真环境|
|`runtime_resources.py`、`bounded_log.py`|统一内存、CPU、磁盘、inode及缓存增长保护；受限日志持续排空管道，并上报写入故障|
|`export_results.py`|生成JSON、CSV和带逐文件校验的证据ZIP，明确标注10回合和提交未就绪|
|`test_audit_guards.py`、`test_client_flow.py`|新增持久化、资源、导出及假环境客户端控制流测试，连同原测试共32项|

官方训练/仿真源码、原10000步结果、模型、数据集及其他项目未改动。服务器某些源码目录没有Git元数据，明确记录commit未知，并保留实际源文件SHA256，不伪造版本。

## 保存与恢复

每个任务的`episodes.jsonl`记录任务、setting、指令类型/实际指令、scene/policy/instruction seed、成功与否、执行步数、耗时和异常；`expert.jsonl`单独记录专家预检查。每次写入立即flush/fsync，再清理环境或开始下一回合。

已完成回合（包括正常失败）恢复时跳过，避免重复计数。中断单独标记；只有明确使用`--retry-interrupted`才允许同seed重试。策略失败不会挑新seed。专家检查异常不算成策略失败或成功，也不静默剔除；最终明确显示有效回合不足。专家检查/采集的五次配额适用范围、归一化方法的赛事认可状态仍未获得逐项官方确认。

先前500计划槽位只有458个有效回合；这次也不能保证1000个计划槽位一定得到1000个有效回合。所有结果标注10回合快速评测，正式100回合成绩与完整比赛总分字段不填写。

## 耗时：约十小时是估算

旧10000步clean测试：500计划槽位、458有效策略回合，后台实际耗时3小时32分40秒，已经包含两路并行；不能再除以2。

|阶段|预算|
|---|---:|
|50任务×10，clean|约3.5—5小时|
|50任务×10，randomized|约4—7小时，尚未实测|
|两个setting评测合计|约8—12小时，可暂按10小时安排|
|首次导出、严格加载验证和smoke|另留准备时间；整体建议约9—15小时，首次随机场景故障可能延长|

以上默认GB10连续供本项目使用。两万步模型成功率改变可能影响提前结束次数，randomized的场景准备和渲染也可能增加耗时，当前不能保证十小时准点结束。

## 存储：结果与权重分别计算

上次500计划回合结果目录只读实测：

|内容|实际文件字节数|
|---|---:|
|逐回合与专家JSONL账本|1,881,562 B，约1.88MB|
|JSON汇总/来源等|363,785 B，约0.36MB|
|文本日志|7,062,126 B，约7.06MB|
|其他小文件|11,018 B|
|目录合计|9,318,491 B，约9.32MB|
|磁盘实际分配|10,375,168 B，约10.38MB|

两种setting共1000计划回合若与历史相似，直接外推约18.6MB；考虑randomized的专家元数据、诊断与新增来源记录，原始结果预计20—100MB；加一份可移植证据ZIP可暂按100—200MB规划，**保守预留2GiB**。这些均为估算，不是按十小时录像计算，因为不保存视频、帧或完整观测。

限制措施：100个任务-setting目录，每目录两个JSONL账本各最多2MiB、一个任务文本日志最多2MiB；两份推理日志各最多16MiB；两个模型准备阶段日志各最多16MiB。这些主要日志的合计上限约664MiB，另有小型配置、摘要、启动记录和文件系统分配。账本达到上限会明确停止并保留已有记录，不静默截断结果。无限次人工重试不在本轮单次运行预算内。

**另需一份20000步推理导出，约25.5GB。**它不是仿真产生的数据，不随回合数增长；两种setting和两路worker共用，不能各复制一份。现有19000/20000步完整checkpoint约82GB已在盘，不作为此次新增存储再次相加。

只算此次仿真结果和证据打包，预留2GiB；连同首次推理导出、日志和缓存余量，**总共先预留40GiB（约43GB）**。未来若官方要求包含完整权重的提交ZIP，额外预留约30GB打包空间，整体可按约70GB新增余量规划。本次未创建该大包。沿用关闭compile的设置，不增加训练checkpoint；实际新增占用仍应在运行后核对。

## 内存

一份模型服务、最多两路仿真worker，不同时加载两个setting的模型。GPU/CPU使用同一物理内存池，不能把显存与系统内存相加。评测启动至少85GiB可用，运行保留16GiB下限；模型准备至少90GiB可用，CPU导出进程退出后才启动严格部署加载验证。使用前一轮验证过的FP32/eager/no compile推理配置，不做额外精度切换。

当前主计算线程预算为服务8、worker各4，有效CPU启动门槛16；原生渲染线程可能额外增加。Torch分配器上限72GiB，不等于整机总内存上限。启动空闲磁盘至少40GiB、运行至少8GiB，inode至少10000。所有缓存位于项目内，整轮缓存增长保护4GiB、CUDA磁盘缓存512MiB；主资源约2秒、缓存约60秒检查。阈值为保护措施，不能保证瞬间资源抢占绝不发生。

最终汇总从原始账本重建，并生成quick_results.json、per_task.csv、submission_readiness.json及quick_results_evidence.zip；ZIP核对成员、CRC、SHA256及尺寸。此包不复制模型，不冒充官方模板格式。官方要求各100次，本轮各10次不足；官方附件schema、上传大小及登录页面仍待核实，不能保证可直接上传获接收。

## 后续命令：本轮均未执行运行阶段

只读/静态检查：

```bash
python3 -B /home/zzj/competitions/lingbot/tools/checkpoint-dual-setting-eval-20261023/run_suite.py
```

未来明确批准后，分别执行，前一步不会自动开始后一步：

```bash
# 仅导出并严格加载检查，不跑仿真
python3 -B /home/zzj/competitions/lingbot/tools/checkpoint-dual-setting-eval-20261023/run_suite.py --mode prepare-model --prepare-approved

# 两种setting各2个技术验证回合，计入各自500计划槽位
python3 -B /home/zzj/competitions/lingbot/tools/checkpoint-dual-setting-eval-20261023/run_suite.py --mode smoke --start-approved

# 匹配smoke通过后，后台顺序跑完两个setting，跳过已完成回合
python3 -B /home/zzj/competitions/lingbot/tools/checkpoint-dual-setting-eval-20261023/run_suite.py --mode full --start-approved --background
```

完整快速评测可加`--workers 1`做单路复现。只有明确检查后需要重试中断回合时，才加`--retry-interrupted`；不用于重复正常失败回合。

当前保持暂停。运行准备尚缺20000步权重导出、严格实际加载验证和两种setting的真实smoke；本轮仅已完成代码与静态/CPU逻辑检查，不能宣称实际仿真已验证成功。
