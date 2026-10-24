# 提交调用入口

本目录与提交ZIP中02/src对应，两个入口都能识别仓库/提交包布局。完整资源来源与调用说明见docs/REPRODUCTION.md；官方格式、计数及提交范围见docs/SUBMISSION_GUIDE.md。

仓库运行示例：

```bash
python3 -B code/submission-entry/portable_eval.py --workdir /path/to/fresh-replay --setting both
```

默认只显示计划。实际文件检查增加--check，并指定--model（已下载/合并的2万步HF）、--vla-python、--sim-python、--qwen-root和含官方资产的--robotwin-root，必要时--mplib-lib。--hash-model检查完整文件；显式--execute才会启动。--workers为1或2，--test-num默认10。两个场景各有独立输出，使用同一份模型。

训练计划用portable_train.py --help查看，需显式指定原始预训练、clean V3、Qwen、MoGe和VLA环境路径；--execute才训练。代码复用原GB10适配，不改变保存策略。8项CPU模拟调用测试位于tests/test_portable_entries.py，没有真实模型或仿真动作测试。
