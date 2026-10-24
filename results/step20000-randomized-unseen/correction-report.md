# randomized指令设置更正

2026-10-24，用户明确要求重跑randomized+unseen。此前检查没有把官方Q&A的setting/指令对应关系纳入断言，是检查遗漏，不归因于用户。

## 原因和修正

旧配置、评测客户端参数、指令生成函数和配置校验均固定seen。只改JSON标签会出现执行指令仍是seen的问题。因此这次在独立工具目录同时修正四个位置：randomized强制unseen，客户端传入配置的instruction_type，生成器选择官方unseen池，结果保存实际指令和候选池SHA256。缺少unseen池时明确报错，不回退到seen任务指令。

新工具：`/home/zzj/competitions/lingbot/tools/randomized-unseen-rerun-20261024`

新输出：`/home/zzj/competitions/lingbot/runs/finetuned-step20000-randomized-unseen-50x10-20261024/randomized`

同一20000步完整HF模型，FP32/eager、关闭compile，两个隔离仿真会话共享一份推理服务；沿用原场景/策略/指令seed规则。50个官方任务，每任务10个计划槽位，demo_randomized+unseen。原clean结果、原randomized+seen记录及原工具目录保留。

35项CPU回归检查通过，新增了三个关键检查：错误的randomized+seen配置被拒绝；官方unseen文本通过真实客户端控制流进入策略observation并保存终结记录；缺少unseen候选时不从seen池选取。逐回合结果仍为追加与fsync、实际有效策略回合分母、异常独立记录，断点不重复计数。

启动后先运行两个真实技术验证回合，核验两会话独立reset、实际动作、unseen候选池及输入；通过后自动继续50×10。这两个验证回合纳入计划500槽位，不额外增加计数。验证技术完成不要求模型必须成功完成任务。小测与正式阶段间保留端口释放等待和资源检查。

CPU指令来源复核：50任务均有官方unseen模板；424份历史完整场景元数据全部可生成unseen指令、无生成错误。其余76个历史槽位缺少完整专家元数据，本次没有凭空制造对应指令，也没有把这次CPU检查当作新仿真结果。

真实技术小测已通过：lift_pot与hanging_mug各完成1回合，实际指令均来自官方unseen候选池，两个session不同；动作shape为50×14，分别执行400和900步。两回合任务均未成功，这是保留的真实策略失败，不是技术验证失败。最低可用统一内存65.58GiB，两个子worker与服务均正常退出，小测结果纳入计划500回合。supervisor已进入正式阶段前的端口释放等待。

## 保留的问题

既有专家可行性预检查/元数据异常逻辑仍可能使部分槽位无法开始有效策略回合。失败、专家阻塞和场景异常继续逐项保留，不改seed筛掉它们、不声称必能完成500个有效策略回合。最新官方Q&A说明专家采集失败不影响策略测试；这一规则与元数据依赖的处理仍须单独解决，不能把本轮指令修正说成全部提交条件均已解决。

没有训练或更换checkpoint，没有关闭官方判分、降低步数上限、保存视频/逐帧图/完整观测。参考此前同机随机场景两路评测耗时，预计数小时；实际按运行日志更新，不用两回合小测估计最终成功率。

## 压缩包状态

此前生成的25.52GB审阅包含旧randomized+seen结果，不作为此次更正后的最终提交包。等待本轮真实结果后，再把保留的clean+seen与新randomized+unseen结果整理成新版本；不能仅修改旧结果中的instruction_type来顶替重测。官方模板仍需登录后下载，当前不向天池提交。

## 启动入口

已经通过用户明确授权启动。服务器脱离本机SSH会话的supervisor PID为2319562（父PID=1，独立session）。原服务器单次入口为：

```bash
python3 -B /home/zzj/competitions/lingbot/tools/randomized-unseen-rerun-20261024/rerun.py --start-approved --background
```

不要重复执行此命令；已有supervisor标记会拒绝重复启动。状态查看使用本目录`snapshot_remote.py`经`remote.py`读取，只读、不启动评测。
