# clean / randomized 顺序调度：端口切换修复

此目录位于比赛项目`tools/dual-setting-dispatch-20261024`，不修改已经真实验证的`tools/checkpoint-dual-setting-eval-20261023`内Python源码。继续使用同一20000步权重、配置、50任务清单、seed计划、原始回合记录和严格smoke门槛。

问题：原调度在clean服务退出后立即检查相同TCP端口，普通bind可能被关闭中的连接阻挡，导致randomized在策略回合开始之前停止。

修复：每个setting启动前使用与原driver相同的普通bind确认端口真正空闲。仅当`/proc/net/tcp{,6}`表明占用来自TIME_WAIT等关闭状态时限时等待，最多90秒，每0.5秒重查；其他监听或无法识别的占用立即报错。等待可中断，并检查内存、磁盘、inode及缓存，不换端口、不接管其他服务、不停止其他项目、不重跑已完成回合。

源文件：dispatch.py（顺序调度、锁、启动门槛、汇总及导出）、port_release.py（端口释放检查）、test_dispatch.py（12项CPU回归）、test_real_transition.py（真实同端口双TCP服务切换，不含模型/仿真）。install-manifest.json固定新调度及原Python源码SHA256；启动时重新核对。

默认只检查，不启动服务：

```bash
python3 -B /home/zzj/competitions/lingbot/tools/dual-setting-dispatch-20261024/dispatch.py
```

未来明确授权开始完整快速评测后：

```bash
python3 -B /home/zzj/competitions/lingbot/tools/dual-setting-dispatch-20261024/dispatch.py --mode full --start-approved --background
```

该命令按clean、randomized顺序运行，各50任务×10计划回合；每种最多两个worker共享同一模型服务。保留--workers 1单路入口。四个已完成的真实试跑回合保留并跳过，仍计入原计划；失败不换seed重跑。某场景技术异常则停止，不自动重启另一场景。

正式启动时，将本目录Python源码与安装指纹保存到结果根目录preparation/dispatch-source-<invocation>.json；端口检查和切换记录保存到dispatch-events.jsonl。最终正式复现包需要包含本目录，不能仅打包旧入口。原有证据ZIP保持原样，不悄悄修改历史启动代码指纹。

此次只部署修复和进行CPU/回环TCP验证，未启动完整评测、加载模型或运行新的仿真回合。10次快速评测仍不满足官方每任务各100次要求，不宣称是完整比赛成绩。
