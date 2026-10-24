# 指定预训练模型 clean 快速基线：准备记录

当前状态：**prepared_paused，未开始评测**。本次仅修改项目内代码、配置并做文本、AST、JSON/YAML、任务清单、路径、计数与输出冲突检查。没有加载模型、启动服务、启动仿真或训练，也没有计算归一化统计。

## 范围与路径

- 使用 `robbyant/lingbot-vla-v2-6b` 原始预训练六个权重分片，revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`。不使用 `-robotwin` 后训练权重，不创建优化器，不更新参数。
- 官方启动脚本的 50 个任务，顺序、成员完全一致，详见 `tasks50.json`；每任务 10 个策略回合，目标共 500 回合。
- `task_config=demo_clean`，`instruction_type=seen`；FP32、eager attention、fused MoE、compile=False、chunk_ret=True、use_length=50，保留已验证设置。
- 明确标为“每任务10回合的预训练模型 clean 快速基线”。不称标准 100 回合评测或完整比赛成绩。
- 工具：`/home/zzj/competitions/lingbot/tools/clean-quick-baseline-20261016/`。
- 新运行目录：`/home/zzj/competitions/lingbot/runs/pretrained-clean-quick-50x10-20261016/`。
- 旧 clean+unseen 的 22 个完成回合保留在原目录，不导入新账本。新目录仅复制小型配置，用符号链接引用原模型；没有复制模型或数据。

## 文件与作用

| 文件 | 作用 |
|---|---|
| config.json、tasks50.json、model-source.json | 固定评测范围、推理设置、种子、模型 revision 和文件来源 |
| driver.py | 一个服务、最多两路仿真进程；静态分配两队各 25 个任务；无自动重试，退出时仅终止自身创建的进程组；保留单路入口 |
| serve.py、session_policy.py | 只加载一次模型；每连接独立策略包装对象、动作缓存、计数和 RNG；共享权重，串行处理请求，不合批 |
| evaluate_task.py | 复用官方场景初始化、任务成功判据和动作执行；独立实现可审计的十回合循环、专家预检查和结果记录 |
| common.py | 追加写入并 fsync 的 JSONL 账本、种子规则、有效回合汇总、启动条件核查 |
| static_check.py | 只做语法、配置、路径、任务清单、计数、源码差异与状态隔离设计的静态检查 |
| normalization-plan.json | 归一化来源疑点及 clean-only 重算方案，不执行计算 |
| RoboTwin/script/eval_policy_client_lingbotvla.py | 将写死的 test_num=100 改为配置参数；正整数检查；客户端地址显式支持 loopback；默认仍兼容原 100 回合调用 |

原客户端补丁、修改前备份、代码版本、模型来源、环境差异和既有兼容补丁哈希保存在新运行目录 `preparation/`、`preparation-manifest.json`。任务源文件、官方 clean 场景、动作含义、成功判据和 `_eval_step_limit.yml` 不修改。本轮任务上限为原文件中的 400–1700 步，50 任务平均 620 步。

## 两路会话隔离

调度进程中的两个线程各维护一个队列，每队列同时只启动一个仿真子进程；单个子进程执行一个任务的十回合，随后退出并处理下一任务。因此是一个推理服务加最多两个仿真进程，不是两份模型或四路仿真。每任务有独立目录和文件锁；整个运行目录也有锁。

服务按 WebSocket 连接生成 session_id，限制最多两个活动连接。每连接浅拷贝轻量策略包装对象，共享同一个 VLA 权重对象，不复制模型。包装对象的 global_step、last_action_chunk、last_normalized_action_chunk、robot_config 和 action_key 独立。上游把 FeatureTransform 放在共享 VLA 上，因此只在互斥锁内切换本会话的 FeatureTransform，并在调用结束恢复。请求间没有 await；CUDA 完成后才释放状态，避免另一个会话提前进入。KV cache 在 sample_actions 的单次调用内以 past_key_values=None 创建，compile 关闭。

每连接保存和恢复 Python、NumPy、Torch CPU/CUDA RNG；每回合 reset 使用记录的 policy_seed。reset 只允许 robotwin，拒绝传入非空 path_to_pi_model，所以客户端不能换模型或重载权重。回复的 session_id 必须与本连接握手相符。无自动合批、模型集成、投票、平滑、置信度门控或测试时更新。

以上设计及上游可变字段已经静态核查；**双会话实际隔离、依赖兼容和运行内存尚未动态验证**。本次不以测试为由加载模型或连接客户端；不能将静态通过表述为双路运行已经验证。

## 种子、逐回合记录和恢复

任务索引 t 按 tasks50.json，回合索引 i=0..9。场景 seed=100000+i；policy_seed=1000000+1000*t+i；instruction_seed=2000000+1000*t+i，与 worker 调度顺序无关。指令只从官方生成器的 seen 列表选择，候选生成数量保留原默认 100，不因为把测试数量降到 10 而同步缩小指令池。保存实际文本，不用语言标签数替代任务数。

每任务写入 `tasks/<task>/episodes.jsonl` 和 `expert.jsonl`。策略记录包括任务、setting、指令类型与文本、三个 seed、回合索引、worker、attempt_id、成功与否、动作步数、官方步数上限、耗时、异常和时间。开始、指令就绪、完成/异常/中断分别追加，完成后立即 fsync，再清理环境、开始下一回合。专家尝试另记启动、结果、异常及用于指令生成的少量场景参数，不保存观测、图片或完整轨迹。

成功率=真实成功数/有效完成策略回合数；0 个有效回合时为 null。摘要同时报告计划数、有效完成数、异常、中断、专家阻塞和未完成数，只有全部 500 个有效回合完成才标 complete=true。异常不默认算成功，也不替换成新 seed；停止整轮以待核查。专家尝试不进入策略成功率分母。

恢复时用 task+episode_index 去重；已完成成功或失败均跳过，不能覆盖失败。异常终止回合需人工核查。进程非正常退出留下的 policy_started 会补记 interrupted，不冒充完成；显式 `--retry-interrupted` 才重做同 seed、同指令的中断回合，旧 attempt 仍保留。恢复复用已记录的通过专家检查的场景参数，避免重复专家调用。配置变更与既有运行不一致时拒绝续跑；账本损坏时停止并要求修复，不跳过坏行。

## 待解决事项：当前启动条件不满足

1. **归一化来源。** 官方 README 明确将示例描述为 clean 与 randomized 联合训练，默认 robotwin.txt 也同时列出两类路径，且有一处 clean 路径本体名称与 Aloha-AgileX 不一致。这使其无法直接充当本赛事 clean-only 输入清单，但不能据此断言 robotwin.json 的确切来源或宣布违规。新配置 normalization.path=null、status=pending_clean_only_provenance，没有默认沿用旧统计；模型构造配置的备用统计路径也已指向尚未生成的 clean-only 文件。

   方案：先验证 V3 包的 2500 条轨迹确实覆盖官方 50 任务、每任务 50 条 Aloha-AgileX clean，保存任务映射与输入哈希；再用官方 compute_norm_stats.py 对这些允许数据的状态和动作计算全量统计，禁用图像读取，保持官方 feature mapping、chunk、padding 和 absolute/delta 含义；保存输入、参数、代码和输出哈希。本次没有执行这一步。应另行确认这种统计处理的赛事规则口径。

2. **专家采集和预检查。** 原客户端每候选 seed 都执行 setup_demo+play_once，再判断 plan_success 与 check_success；失败、异常或不稳定会 now_seed+=1，继续找可通过专家的场景。策略失败后本来也不重跑该 seed。新方案保留相同专家成功检查和场景流程，但对固定 seed 检查失败明确记录并停止，不静默换 seed 或筛除困难场景。一个正常完成的目标方案可能包含 500 次专家预检查加 500 个策略回合；前者是否属于规则的专家采集配额、配额的“场景”按任务还是按 seed 定义，以及旧测试已消耗多少次数，仍需确认。不能把本地每 task/seed 的五次尝试保护当作官方配额已核实。若专家失败，目标 500 回合会保持未完成，等待核查，不擅自关闭检查强行凑数。

3. **工程调度与动态验证。** 两路共享同一权重、隔离状态并保留单路入口，是拟采用的工程调度方案，不宣称官方逐项确认。只能在后续获得授权后验证实际双会话行为；当前不宣称已运行可靠。规则未明确的训练方法和归一化来源均列为待确认，不自行宣布合规或违规。

## 资源估算

- 实测依据：此前同一 FP32 权重在真实三路图像输入下的 Torch GPU 张量峰值为 25.87 GB；六分片文件共约 25.50 GB。这个 25.87 GB 已属于 GB10 的统一物理内存使用，不能再与“显存”重复相加。机器 MemTotal 约 128.50 GB（119.68 GiB）。本次没有加载模型测内存。
- 两路方案估计：常驻总内存约 50–80 GB，包括一份模型及中间量、推理进程主机开销、两套仿真/规划/渲染资源和系统余量。两套仿真各约 4–10 GB、推理主机额外约 8–15 GB 均为未实测的预算假设。加载期可能短暂到 60–90 GB，服务健康后才启动仿真，避免同时出现两份模型加载峰值。正式启动前仍需检查其他项目占用。
- 输出：正常 500 回合的 JSONL、摘要、计划和元信息估计 5–15 MB；文本日志每任务最多 2 MiB、推理日志最多 16 MiB，整轮文本上限约 122 MB。合计为结果目录预留 200 MB，关闭视频、逐帧图片、完整观测及轨迹输出。账本不为省空间删除失败记录；异常堆栈限长。反复显式续跑的账本仍会增长，需要保留审计。
- 模型/数据复制新增量为 0。Torch/CuRobo/Triton/CUDA 缓存单独计，均定向项目 cache 目录；compile=False 不代表这些底层库完全不产生缓存。未来首次运行预留 1–5 GB 缓存余量，这是估算，本次未生成新编译产物。
- 耗时：旧 lift_pot 单路约 69 秒/400 步回合（含原流程开销）；全任务平均上限 620 步。按比例，单路约 15 小时量级；两路理想值约 7.5 小时，但同一推理服务串行，仿真和规划任务复杂度不同，暂预算 8–16 小时。双路没有实测，不承诺两倍加速；专家阻塞或异常会提前停止，不能视为完成。

## 后续命令：仅供审核，未执行

需先解决上述待确认项、填写统计文件及证据与 SHA256、确认专家规则和历史计数，再由用户明确批准开始。当前 execution_approval.approved=false，脚本还要求显式 --start-approved；条件不满足会在模型/仿真导入之前拒绝运行。

```bash
python3 /home/zzj/competitions/lingbot/tools/clean-quick-baseline-20261016/driver.py --workers 2 --start-approved
```

单路复现只改调度数量，权重、任务、seed、指令和推理参数保持一致：

```bash
python3 /home/zzj/competitions/lingbot/tools/clean-quick-baseline-20261016/driver.py --workers 1 --start-approved
```

中断后经明确批准重做同 seed 的未完成回合时才增加 --retry-interrupted。若更换归一化或策略配置，需使用新运行目录；不能把改变后的策略结果续写进已有运行。

最终参赛模型须在规定的 50 个任务 clean 数据上联合训练；当前原始预训练权重仅用于建立基线。**当前仍保持暂停，未开始本轮评测。**

参考：天池赛题与数据 https://tianchi.aliyun.com/competition/entrance/532514/information ；官方源码固定 revision be969b8fd117fb70550c5d4bf4bc328211b5b1b6 的 README、experiment/robotwin/start_robotwin_infer_and_eval.sh、scripts/compute_norm_stats.py；RoboTwin revision 13c3c47ff4312dd62484bcd51be034af55c062d1。论坛规则全文仍未由本对话直接抓取；专家五次配额和 clean/seen 对应关系按用户要求及转来的核查信息处理，不冒充已完整取得官方附件。
