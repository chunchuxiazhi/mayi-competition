# 复现部署与独立调用

以下详细说明中的目录按提交ZIP表述。仓库对应关系：02/src/runtime-source→code/runtime-source，02/src/tools→code/tools，02/src/portable_runtime及portable_*.py→code/submission-entry，02/configs→code/submission-configs（原始配置位于code/configs），03/checkpoint→另行下载的完整HF目录。可直接调用仓库portable_eval.py/portable_train.py，不需要复制成中文目录结构。

# 代码运行与依赖

本目录对应团队1385264、累计20,000步模型的50任务clean+seen和randomized+unseen评测。src/tools保留实际历史运行代码；src/portable_eval.py与src/portable_train.py是本次补充的路径适配调用入口；src/portable_runtime复用历史推理服务、会话隔离、任务循环、资源保护和落盘逻辑。src/runtime-source为实际使用的源码快照，现已补齐lingbotvla/models和lingbotvla/data。

## 环境安装与资源准备

原实验使用ARM64 Linux、NVIDIA GB10、Python3.12、Torch2.9.1+cu130，系统/GPU共享约119.7GiB统一内存；VLA和仿真使用两套环境。精确包版本见src/environment，版本约束和MPLib显式环境记录见configs/original。新入口要求Python>=3.11；运行平台为Linux+NVIDIA。

现有实验服务器可直接通过--vla-python和--sim-python指定已验证环境，不需要重新安装或变更其他项目。新机器需先部署CUDA/Vulkan、PyTorch、Sapien、MPLib及资产；官方LingBot安装脚本为src/runtime-source/lingbot-vla-v2/tools/create_train_env.sh，RoboTwin安装说明参考上游https://github.com/RoboTwin-Platform/RoboTwin。ARM64环境需对照本包实际安装版本适配；官方安装脚本默认尝试FlashAttention，而本实验采用eager，不要求安装FlashAttention。

已有对应依赖后，安装本包模型源码的命令是：

```bash
/path/to/vla/bin/python -m pip install --no-deps -e "$BUNDLE/02_代码材料/src/runtime-source/lingbot-vla-v2"
```

上述--no-deps只用于保留已经配好的运行栈，不是完整依赖安装命令。包内已安装清单是复现依据；不宣称任意平台照搬pip清单都能一次安装成功。仿真程序实际使用的RoboTwin代码需与本包匹配，并准备官方资产、机器人和unseen纹理；数据、资产、教师权重、环境二进制不随ZIP重复分发。

## 权重和辅助模块来源

- 初始化：https://huggingface.co/robbyant/lingbot-vla-v2-6b，使用原始预训练权重，没有用官方RoboTwin后训练模型。
- Qwen3-VL骨干与tokenizer依赖：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct；骨干冻结，视觉仍参与前向。
- MoGe-2：https://huggingface.co/Ruicheng/moge-2-vitb-normal，model.pt；与LingBot-Depth共同提供冻结深度监督。实际build_depth_model在MoRGBD模式下加载这两份教师。
- LingBot-Depth：指定预训练仓库的depth/model.pt；冻结深度教师。
- DINO-Video：指定预训练仓库的dino_video/teacher_step_10000.pth及config.yaml；冻结视频教师。深度、未来深度、未来视频损失权重各0.004。相关上游许可证见资源原仓库与本包licenses，教师权重不另行分发；DINO-Video/DINOv3用于训练监督，推理不额外加载教师。
- clean数据：https://huggingface.co/datasets/TianxingChen/RoboTwin2.0；仅50任务clean V3、每任务50演示。归一化统计来自该允许数据，详见04报告和冻结配置。

## 两场景评测调用

先设置BUNDLE为解压后的团队目录，RESOURCES为已部署依赖和资产的目录。以下示例中--robotwin-root目录需含本包匹配的RoboTwin源码和官方完整资产。

```bash
BUNDLE=/path/to/连仿真都做不到的话_初赛提交材料
RESOURCES=/path/to/existing/resources
cd "$BUNDLE/02_代码材料"
PYTHON_BIN="$RESOURCES/envs/vla/bin/python" bash eval.sh \
  --setting both --workdir /path/to/fresh-evaluation \
  --vla-python "$RESOURCES/envs/vla/bin/python" \
  --sim-python "$RESOURCES/envs/sim/bin/python" \
  --qwen-root "$RESOURCES/models/Qwen3-VL-4B-Instruct" \
  --robotwin-root "$RESOURCES/repos/RoboTwin" \
  --mplib-lib "$RESOURCES/envs/mplib-build/lib" --check
```

默认只打印计划；--check只检查文件、任务和路径；--hash-model可额外逐块校验完整权重，均不加载模型、不执行动作。评委实际运行时将--check替换为--execute；使用同一参数把--setting改为clean或randomized可分别运行，--workers 1保留单路入口，默认2。--test-num默认10，可明确改为100；实际次数会写入新配置和结果。默认模型路径是本包03/checkpoint，也可通过--model指定同一完整HF导出。

两个场景顺序运行，各自启动一份模型服务及最多两个worker；场景结束清理本次进程，再等待端口释放。单份权重、FP32、关闭compile、串行推理、不自动合批；reset、动作缓存、步数和RNG按会话隔离。不集成、不按任务路由、不更新参数。

新输出位于workdir/runs/replay/clean与randomized；配置、seed、指令、逐回合事件和异常均独立保存；results.json和accounting.json位于workdir/runs/replay。完成回合立即追加并同步，断点按任务/回合识别；中断不自动重试、失败不换seed。禁止把新输出写进历史实验或提交目录。

专家预检查/元数据处理保留历史行为，包括专家异常时阻塞该槽位。这与最新Q&A“专家失败仍继续policy”的要求存在披露差异，本次用户明确不补跑，调用适配没有暗中修改该评测行为。

## 训练调用

```bash
PYTHON_BIN="$RESOURCES/envs/vla/bin/python" bash train.sh \
  --workdir /path/to/fresh-training --stage both \
  --pretrained "$RESOURCES/models/lingbot-vla-v2-6b" \
  --data "$RESOURCES/data/RoboTwin_lerobot_v30" \
  --qwen-root "$RESOURCES/models/Qwen3-VL-4B-Instruct" \
  --moge-root "$RESOURCES/models/moge-2-vitb-normal" \
  --vla-python "$RESOURCES/envs/vla/bin/python"
```

默认只输出计划，显式加--execute才会训练。入口重定位原配置和控制文件，调用原train_entry.py及保存/续训适配，不另写模型或损失。第一阶段10,000步、lr2e-5；第二阶段从第一阶段完整DCP续训至20,000步，lr1e-5到5e-6余弦；每1000步同步保存，跨阶段只保留最新两份。阶段之间单独CPU导出HF，训练与导出不并发。模型/数据通过路径或符号链接复用，120GiB磁盘及90GiB统一内存启动余量保护仍保留。完整HF推理模型不含优化器，不能冒充完整DCP来恢复第二阶段。

## 代码改动与验证边界

原始RoboTwin改动包括script/eval_policy_client_lingbotvla.py中的原生库导入顺序、可配置test_num、输出目录和视频控制；原LingBot/GB10适配及来源记录见04、src/tools和补全源码manifest。任务envs的成功判据、动作含义、步数上限、clean/randomized配置未改变。新调用只改路径、HF来源校验、调度入口和独立落盘，不重新计算历史分数。

新调用入口没有复用或伪造历史smoke证据，也未进行新的真实推理/仿真验证。本次通过CPU语法和模拟调用测试，不能据此宣称新机器环境已端到端实测。运行前应按平台核验外部环境；本次提交准备没有启动上述任何--execute命令。
