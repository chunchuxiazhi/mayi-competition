# LingBot-VLA比赛项目：改动汇总与10000步训练方案审核

日期：2026-10-16。用途：交给同伴检查工程改动、训练方法和评测流程是否符合比赛要求。本文是技术审计，不代表主办方确认合规。

## 1. 当前状态与审核范围

1. 已完成指定原始预训练模型的本地快速基线，随后完成20次optimizer step的真实训练试验、一次checkpoint保存、HF导出及严格模型重载。
2. **10000步正式训练尚未配置并启动。** 当前正式首轮配置仍是旧的2000步预算和自动校准保存间隔，不能直接拿来当作本文的10000步方案执行。
3. 本文覆盖本比赛的环境适配、历史源码补丁、clean统计重算、自定义快速评测客户端，以及训练适配代码和配置。下载、检查、启动等一次性辅助脚本按用途归组，生产入口逐个列出。源文件和patch哈希附在文末。
4. 修改范围是`/home/zzj/competitions/lingbot`及其中的项目独立环境。ELF、LLaDA等其他项目没有作为本次修改目标；没有改系统库文件。链接到系统CUDA、librt等不等于修改它们。
5. **措辞纠正：不能笼统说“官方源码完全没改”。** 训练准备阶段保留了官方训练主循环、损失定义和模型参数布局，但更早搭建环境时确实修改过少量模型源码、RoboTwin非任务逻辑文件和项目环境内的SAPIEN加载器，具体如下。

## 2. 比赛主线与当前口径

1. 初始化为`robbyant/lingbot-vla-v2-6b`指定原始预训练权重，不使用`lingbot-vla-v2-6b-robotwin`等现成RoboTwin后训练权重。
2. 原始权重revision记录为`11c703bf6a5c1f45b3b69168482da11fdbba53d7`；原始六个safetensors分片保留，训练另写输出目录。
3. 已记录的LingBot参考源码revision为`be969b8fd117fb70550c5d4bf4bc328211b5b1b6`；RoboTwin参考revision为`13c3c47ff4312dd62484bcd51be034af55c062d1`，叠加下列本地补丁。不能仅报commit而不附补丁。
4. 拟参赛模型在指定50任务的clean示范上联合训练同一个策略。数据为LeRobot V3 clean：2500段、548893帧；每任务50段。不加入randomized示范。
5. 训练配置里的`post_training: true`表示“我们进行微调”的代码模式，不代表载入官方后训练权重。
6. 官方示例本身包含clean与randomized联合训练，不能原样照用它的数据路径列表或据此认定其归一化来源符合本赛事clean-only要求。
7. 已取得赛事介绍/赛题与数据的部分公开内容；论坛规则解读、常见Q&A和提交附件全文仍不完整。训练方法、注意力后端、归一化及专家配额等未获赛事逐项正式确认，保留待审核状态。

## 3. 目录、模型和数据管理

| 路径，相对于比赛根目录 | 内容及处理 |
|---|---|
| `models/lingbot-vla-v2-6b` | 原始模型、附带depth和video教师文件，保留原文件 |
| `models/Qwen3-VL-4B-Instruct` | 必需的配置、tokenizer和处理器；不另复制整套Qwen权重 |
| `models/moge-2-vitb-normal/model.pt` | MoGe教师权重 |
| `data/RoboTwin_lerobot_v30` | 原始V3 clean数据，未就地改视频或Parquet |
| `configs/normalization/robotwin_clean_v3.json` | 从现有clean状态/动作重算的统计 |
| `configs/robots/clean-v3-gb10/robotwin.yaml` | 派生机器人映射，主要更换统计路径 |
| `configs/training/clean-v3-gb10/{pilot,stage1}.yaml` | GB10派生配置，原官方示例保留 |
| `tools/clean-quick-baseline-20261016` | 快速基线调度、会话隔离、专家/策略记录 |
| `tools/finetune-clean-v3-gb10` | 训练适配器、资源保护、导出和检查 |
| `runs/pretrained-zero-shot-20261015` | 更早试验及clean+unseen记录，单独保留 |
| `runs/pretrained-clean-quick-50x10-20261016` | clean+seen快速基线及归一化来源证据 |
| `runs/finetune-clean-v3-gb10/pilot` | 已完成的20步试训、唯一checkpoint和HF导出 |
| `docs/setup-baseline-20261015` | 环境安装记录、源码原件、兼容补丁 |
| `docs/training-prep-20261016` | 依赖前后清单、测试、训练准备及实测报告 |
| `cache`、`downloads`、`logs` | 项目缓存、安装资源和日志，不向其他项目放置副本 |

新运行目录只保存权重输出及小型配置/来源记录，不复制原始模型和整套数据。更早零样本目录曾临时引用官方统计，后续clean快速基线和本次训练均明确改用clean重算统计，两个时期不能混为一谈。

## 4. 环境与依赖改动

### 4.1 GB10平台差异

1. 服务器是Ubuntu 24.04 ARM64、GB10 sm12.1、约119.7 GiB统一物理内存。CPU内存与GPU张量占用属于同一池，不能重复相加。
2. 使用项目环境`envs/vla`和`envs/sim`。Torch采用已可用的`2.9.1+cu130`、torchvision `0.24.1`、Triton `3.5.1`，与官方安装脚本的部分固定版本不同。
3. `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`，使用CUDA 13工具链。设置`TORCH_CUDA_ARCH_LIST=12.0`和扩展编译`MAX_JOBS=2`，是本机已验证的构建路径；GPU实际仍是sm12.1，没有改变硬件。
4. Torch仍会提示sm12.1超出该版本声明的最高sm12.0。当前推理和20步真实反向已通过，但不等于所有算子与任意长训均已验证。
5. 关闭的是`torch.compile`，底层Triton/CuRobo仍可能首次编译内核并增长缓存。

### 4.2 训练依赖

1. 已有核心版本保留：transformers 4.57.3、NumPy 1.26.4、LeRobot 0.4.2、PyAV 15.0.0、torchdata 0.11.0、pyarrow 21.0.0、datasets 3.6.0、accelerate 1.7.0。
2. 补装：matplotlib 3.10.3及contourpy、cycler、fonttools、kiwisolver、pyparsing；lxml 5.4.0、pycryptodomex 3.23.0、optree 0.14.1、numpydantic 1.9.0。
3. utils3d 1.3使用MoGe指定提交`3fab839f0be9931dac7c8488eb0e1600c236e183`，源码、提交和安装记录留在项目目录。安装时使用`--no-deps`保留现有固定栈。
4. 未为消除依赖声明而强行安装全部机器人采集SDK、升级NumPy或降级Torch。未安装FlashAttention，未强编译当前路径不需要且无适用ARM wheel的moderngl。
5. **pip check没有全通过。** LeRobot声明的Torch/torchvision、datasets、accelerate、setuptools版本范围与当前栈不符，并缺部分采集/控制包；utils3d缺moderngl；numpydantic声明NumPy>=2而当前是1.26.4。当前已用训练实际路径验证，但不宣称整个环境所有功能都兼容。
6. 本次补依赖前后版本清单已保存，补装阶段没有替换已有包版本。这个结论限定于本次补装，不是说从最早安装开始从未改变环境。

### 4.3 仿真依赖与二进制适配

1. SAPIEN实际为3.0.3，与参考requirements的3.0.0b1不同；这是重要复现差异。
2. mplib来自官方v0.2.1标签提交`86376ca87d2c055b792d8cea4d15a38e861d866f`的ARM源码构建；该快照VERSION标为0.2.0，所以包元数据报告0.2.0。不能简单把元数据版本误报为源码来自其他标签。
3. CuRobo 0.7.8在项目内构建，使用CUDA工具链及最多两个编译任务；相关机器人资源路径通过项目内配置更新。
4. SAPIEN ARM wheel自带旧librt与Ubuntu的GLIBC_PRIVATE符号冲突：备份项目venv中的`librt-2-4cc790af.28.so`，替换为指向现有系统`/usr/lib/aarch64-linux-gnu/librt.so.1`的符号链接。系统库本身未改。
5. 原SAPIEN捆绑OIDN 2.0.1不支持本机CUDA设备，实拍图像出现去噪问题；在项目内构建OIDN 2.5.1，并修改项目venv的`sapien/_oidn_tricks.py`加载它，新增库链接，旧库及源码备份保留。
6. 已实际检查lift_pot clean场景三路RGB有限且非恒定，图像目视检查及新OIDN设备检查通过。**去噪库/仿真版本变化可能影响图像分布，不能据此声称与官方基准图像完全一致，须让同伴重点审核。**

## 5. 历史源码补丁：逐项说明

1. **LingBot `modeling_lingbot_vla_v2.py`：** 三处写死的`flash_attention_2`改成读取`self.config.vit_attn_implementation`，涉及Qwen整体/text配置和动作专家配置；vision配置原本已读此字段。目的为使官方配置选择真正生效，没有增删网络层或更换参数形状。
2. **LingBot `qwen2_action_expert.py`：** 增加`logger = logging.get_logger(__name__)`，避免已有回退/日志路径引用未定义logger。没有改MoE路由、专家数量或前向公式。实际采用fused MoE，保持原始checkpoint参数布局。
3. **RoboTwin `envs/camera/camera.py`：** 移除模块顶层未用于本轮RGB路径的Open3D导入。
4. **RoboTwin `envs/utils/save_file.py`：** Open3D由顶层导入改为`save_pcd()`内按需导入。未启用点云保存，未改变本轮RGB或任务成功判据。
5. **RoboTwin评测客户端：** 先导入torch、sapien、mplib，解决原生库加载顺序问题。
6. **同一客户端：** 去掉实际通过WebSocket服务推理时无用的`get_model = eval_function_decorator(...)`调用，避免导入无关策略模块。
7. **同一客户端：** `test_num=100`改为可配置正整数，默认兼容100；循环参数传递不再写死100。WebSocket host显式可配置，默认127.0.0.1。
8. **SAPIEN `_oidn_tricks.py`和项目venv动态库：** 见上一节，不应遗漏在“代码未改”的声明之外。
9. 上述原文件备份和unified diff保留在`docs/setup-baseline-20261015`及快速基线的`preparation`。本文末尾列出8份已找到的patch；它们是本地记录集合，不是对所有上游文件逐字穷举比较后得出的“除此之外绝无差异”证明。
10. 官方训练主循环`tasks/vla/train_lingbotvla.py`未在本次训练准备中就地修改；资源管理与数据适配通过新入口在运行时包装。官方任务类、clean/randomized场景配置、动作含义、任务成功判据和步数上限不在本项目的修改范围内。

## 6. clean-only归一化与任务清单

1. 新增`compute_clean_norm.py`，只读本地V3 clean的状态、动作和元数据，不读训练图像、不执行仿真、不更新模型。
2. 使用官方`RunningStats`数值类，5000个直方图bin，以64个episode为确定性处理批次。状态每帧一次；动作采用0..49的绝对动作窗口，末端在同一episode最后动作处padding，不跨episode、不减当前state。
3. 保持官方12个臂关节和2个夹爪原始切片及映射，再按模型统一55维表示处理；不因微调修改动作语义。
4. 输出`configs/normalization/robotwin_clean_v3.json`，SHA256：
   `0b5ed854065e9c64e1b8934c274c38d0efa062d743bfad071f1a089983125611`。
5. 记录输入Parquet/元数据哈希、官方数值类和机器人映射哈希、脚本哈希、统计计数及任务↔episode映射。
6. 任务映射不是把语言标签数量当任务数：按50段episode块，用官方seen指令模板匹配识别50个唯一任务，每块至少40/50匹配且领先其他任务。元数据与真实数据的索引、长度和帧范围一致。
7. 有12段指令没有逐字匹配预期模板，原文及episode ID均保留；未替换这些训练指令、未删样本。块级映射通过不等于每句模板完全一致，需审核该证据是否充分。
8. 可以确认本地统计只用了clean来源；**主办方是否接受此统计计算口径仍未正式确认**，不能仅凭“用了官方RunningStats”宣布合规。

## 7. 快速基线评测改动与已有结果

### 7.1 调度、计数、输出

1. 自建`driver.py`、`serve.py`、`session_policy.py`：一份FP32预训练权重、最多两路仿真；推理锁内串行，不自动合批。保留单路入口。
2. 每个WebSocket连接独立session_id、reset、动作缓存、步数、机器人配置和FeatureTransform；Python/NumPy/Torch CPU/CUDA随机数状态独立保存恢复。模型权重共享，不做集成、投票或测试时更新。
3. 自建`evaluate_task.py`使用官方场景初始化、动作执行、任务成功判据和步数上限，但重写可审计的固定回合循环、专家处理及输出；它不是未经修改的官方评测程序。
4. 计划50任务×10个策略回合，`demo_clean`+`seen`。场景seed=`100000+i`；policy seed=`1000000+1000*t+i`；指令seed=`2000000+1000*t+i`，t为固定任务索引，i为0..9。不因策略失败换seed重跑。
5. 指令用官方生成器的seen池，候选数量保留100；保存实际指令文本，不临时编造兜底指令。
6. `common.py`、`status.py`记录JSONL和任务/整体汇总；逐回合立即追加并fsync。包含任务、setting、指令、seed、成功、执行步数、耗时、异常、会话证据等。
7. 已完成成功或失败均去重保留；中断单独标记，只能显式按同seed/同指令恢复。全局模型/内存/磁盘错误停止运行。
8. 关闭视频、逐帧图片、完整观测轨迹和专家训练数据保存；之前clean+unseen结果独立保留，不混进新clean+seen基线。

### 7.2 专家预检查的实质性控制流变化：重点审核

1. 官方原流程会对候选seed先执行`setup_demo + play_once`，专家失败/异常可能换seed，寻找可行场景。
2. 本地固定seed；专家正常返回失败但场景参数和官方seen指令有效时，重新初始化**同seed**场景，仍执行并记录策略回合，不因专家失败筛掉这一回合。
3. 若专家异常、物体不稳定或参数/指令缺失，保留未完成槽位，继续其他固定槽位；不静默换seed凑齐500回合。
4. `recovery.py`校验参数、对象描述、记录来源与配置修订；复用已保存的专家结果，避免恢复时无谓再次调用专家。专家事件和策略回合分账计数。
5. 本地每task/seed的尝试保护不代表已确认官方“五次专家采集配额”的作用域。可行性预检查是否属于采集、历史调用次数、失败场景应如何评测，都须按赛事文件/答疑确认。
6. 8项控制回归、会话隔离检查和真实双路技术smoke已通过。技术smoke的策略失败仍计为任务失败，不把“任务完成成功”当作程序可运行的唯一条件。

### 7.3 已有分数的正确表述

计划500个固定槽位，实际有效策略回合459；成功43、普通失败416，另41个槽位因专家预检查异常没有进入有效策略回合。35任务有10个有效回合，15任务不足10个。

- 43/459约9.37%是**已执行有效回合合计成功率**。
- 不能称为完整500回合成绩，不能写成“500个有效回合成功43个”。
- 后续与微调模型比较要保留异常槽位、各任务分母及共同有效seed，不能只比较筛掉异常后的数字。
- 这只是“每任务10回合的预训练模型clean快速基线”，不是标准100回合评测或正式比赛成绩。

## 8. 训练配置：相对官方示例的变化

1. 从官方`configs/vla/robotwin/robotwin.yaml`派生新配置，不覆盖原示例。完整逐字段差异自动附在本文附录A。
2. 官方示例micro batch 32/global batch 1024，注释对应32GPU；本机1rank、micro batch 1，pilot累积4，现有stage1累积8。参数每4/8个微批更新一次，保存文件是另一种频率。
3. `train_expert_only=true`冻结Qwen视觉语言骨干；`freeze_vision_encoder=true`。实际可训练参数1,938,090,551，包含动作MoE专家及相关投影/对齐模块，不是仅训练一个小线性头。不是按任务选用不同专家或不同权重。
4. `freeze_vit=false`保留该字段，因为当前根模型不按这个旧字段定位视觉模块；实际冻结范围由`train_expert_only`及可训练参数检查确定，不能只看字段名字。
5. 保持官方MoE结构：36层、32专家/top-k 4及已有辅助路由损失等。`moe_implementation=fused`保持参数布局，不切换为参数名不匹配的另一套MoE实现。
6. 优化器从Muon改AdamW；关闭MoE专家单独LR放大。峰值LR从1e-4改2e-5，最低LR从5e-5改1e-5，保留cosine。pilot warmup比例0.1=2步；现有2000步stage1比例0.03=60步。
7. `enable_fp32=true`、`enable_mixed_precision=true`组合在当前官方入口中构建FP32模型；实际导出6,375,907,511个元素全部F32。教师目标计算仍保留官方BF16 autocast，**不能把整套训练叫作所有算子均纯FP32**。
8. `use_compile=false`；VLA注意力用`train.attention_implementation=eager`，Qwen/视觉相关后端显式用`model.vit_attn_implementation=eager`。没有删掉注意力层，也不保证与FlashAttention逐位一致。
9. 首次真实启动因只设置前一个字段，后一个仍默认FlashAttention2而失败，更新次数0。补齐model字段后重试完成；失败日志及旧配置保留。此前使该字段生效的源码补丁见第5节。
10. 开启梯度检查点以降低激活内存；未开启CPU或activation offload，未引入LoRA、平滑、置信度门控或新的训练目标。
11. 保留depth/video教师与原损失权重、未来图像监督和动作窗口50。video教师`flex_block_causal`改为仓库已有的`sdpa_block_causal`，目的是避免隐式compile；设计采用同类分块因果掩码，但尚未做两条路径逐项数值对照。
12. 固定图像尺寸224、关闭图像增强。数据仅指向合并V3 clean目录，机器人映射指向clean重算统计。
13. DataLoader `num_workers=0`，`pin_memory=false`，无异步预取；配置中的prefetch_factor=1在worker=0时实际传None。初期两spawn worker原始读取通过，但完整归一化StatefulDataLoader退出时复现SIGABRT；未吞错，最终用同步路径并验证正常退出。
14. DataLoader的一个外层batch包含累积所需的多个微批；模型一次前向micro仍为1。梯度累积没有并行启动8份模型。
15. 训练seed固定42。关闭自动评测、W&B及训练中HF导出；辅助图像可视化间隔设100万步，本次20步及拟10000步不会触发。
16. 现有stage1仍为2000步/save_steps=250最大值；另有约30分钟自动校准逻辑，见第12节。本文比较的固定1000/5000保存尚未落到执行入口。

## 9. 新增训练代码：逐文件说明

| 文件，位于`tools/finetune-clean-v3-gb10` | 作用与边界 |
|---|---|
| `launch.py` | 默认只校验，显式`--start`才运行；1rank互斥锁、GPU占用/统一内存/磁盘检查，冻结配置及代码哈希；串行训练→CPU导出→GPU严格重载，不自动启动stage1或仿真 |
| `train_entry.py` | 导入官方训练主循环并安装运行时适配；恢复失败强制退出，不接受上游捕获错误后回退成新优化器 |
| `gb10_runtime.py` | 包装模型/优化器/数据构造和DCP；检查冻结范围；记录loss、梯度范数、LR、步数、耗时与内存；严格恢复与同步保存；每次启动有attempt标识 |
| `safe_video.py` | 训练专用LeRobot子类，逐episode/相机读取有效视频范围；有限尾帧补齐及审计，不改原始视频或评测环境 |
| `storage_policy.py` | 原子JSON、checkpoint完整性标记、保留清理、空间预检查、保存间隔校准；限制删除到本run合法路径 |
| `export_cpu.py` | 训练退出后只读DCP模型状态，CPU导出完整FP32 HF权重；检查分片和哈希、原子重命名，验证动作投影相对原预训练权重确有变化 |
| `verify_reload.py` | 用实际部署模型类严格加载HF，核对动作投影与导出权重完全相同；不执行动作推理或仿真 |
| `validate_data.py` | 7500个相机/episode区间和实际样本读取检查，无模型训练 |
| `validate_native_loader.py` | 同步原生StatefulDataLoader的微批、状态序列化及正常退出验证，使用CPU单rank布局桩 |
| `test_policies.py` | 12项CPU测试，覆盖视频边界、checkpoint路径/完整性/保留、保存间隔计时规则等 |
| `pilot-control.json`、`stage1-control.json` | 项目路径、任务清单、统计及上游trainer哈希、资源阈值、保留数量和待确认标记 |

其他一次性脚本负责依赖安装、源码/元数据检查、配置生成与校验、SSH后台启动、失败归档、状态读取和报告发布。它们不定义新的训练损失或任务判分；不是正式训练时全部同时执行。

### 9.1 视频缺尾帧处理：重点审核

1. V3中部分相机视频较该episode的状态/动作长度少1–2帧；原始读取在某些尾部样本会越界。
2. 训练专用适配器按double时间及真实区间定位，严格限制缺帧数量为0、1或2；只对缺失尾部显式重复同episode最后一张有效图像，保留对应状态/动作标签。
3. 其他长度错误、时间不对齐或无有效帧直接报错，不随机重抽、删样本、扩大原始匹配容差，也不读取相邻episode图像。
4. PyAV解码器每路1线程，用完关闭；模块级子类可被spawn序列化，但最终训练仍用0 worker。
5. 记录发生补齐的episode、相机、原索引和选用索引；按每进程每episode/相机首次事件审计，不是每次重复索引都追加一条。日志上限16 MiB/文件。
6. 原始视频和Parquet未修改。**这属于会改变少量训练输入的预处理，并非纯日志改动；赛事是否接受须明确审核。**

### 9.2 内存与存储保护

1. 启动要求统一内存MemAvailable>=90 GiB，拒绝与已有GPU计算进程并跑。
2. CUDA分配器上限72 GiB；它不是整个进程RAM硬配额，也不是额外的独立显存。
3. 训练父进程每2秒检查MemAvailable；低于16 GiB停止，20秒未退出则终止自己创建的训练进程组。优化器更新前也检查，保存前要求可用>=28 GiB。
4. `clip_grad_norm_`启用非有限报错，loss非有限停止，微批数量与更新步数不符停止。
5. 训练root共享250 GiB预算；DCP保存前预留64 GiB，并另留磁盘120 GiB。CPU导出前要求56 GiB可用统一内存、预留32 GiB磁盘；重载前要求80 GiB可用。**这些是预检查和轮询保护，不是文件系统硬配额或无OOM保证。**
6. 关闭core dump。本轮DataLoader并行诊断曾产生约643 MB的系统崩溃转储，核实为本轮测试文件后已清理，安全诊断摘要保存在项目内；未清理其他人的文件。
7. 单GPU训练进程、同步读取、DCP一个写入线程；训练模型、导出模型和重载模型不同时常驻。
8. 文本训练日志上限50 MiB/阶段，结构化训练日志有单文件上限，保留小型TensorBoard和尾帧审计；不保存训练视频、逐帧图像或完整观测。TensorBoard和编译缓存仍会增长，不能把文本日志上限当成全部输出硬上限。

### 9.3 checkpoint及恢复设计

1. 模型、优化器分别DCP保存，附步数、LR scheduler、DataLoader状态及Python/NumPy/Torch CPU/CUDA随机数状态。
2. 同步写入和fsync；检查metadata、分片及extra state后写`complete.json`，记录文件大小与关键metadata摘要。
3. 新恢复点提交成功后才删除超出保留数的旧完整点。只允许本run/checkpoints直接子目录、合法global_step名称和匹配配置签名，拒绝外部路径及符号链接目标。
4. 不自动删除未知、未完成或损坏目录。恢复严格要求模型、优化器和配置一致，失败停止；到达计划最大步数后拒绝再次多更新一步。
5. 每个恢复点实测约41.02 GB，其中模型约25.51 GB、优化器约15.51 GB；最终HF为额外约25.52 GB。不是把每个checkpoint再加一套独立原始模型。
6. 本轮实际验证了保存、DCP模型状态读取导出及部署模型严格重载；**优化器/DataLoader中断后恢复及长训保留清理尚未做真实大模型验证**，CPU策略测试不能替代该验证。

## 10. 20步真实试训实测

| 项目 | 结果 |
|---|---:|
| optimizer step / micro batch | 20 / 80，累积4 |
| 稳定更新耗时中位数 | 4.871秒，剔除含初始化的首步 |
| 稳定耗时范围 | 4.785–5.384秒 |
| 训练循环 | 约106秒，含首步冷启动，不含模型读取和保存 |
| DCP保存 | 约38.16秒 |
| 成功启动到导出及重载结束 | 约6分51秒，不含更早失败诊断 |
| CUDA allocated峰值 | 54.594 GiB |
| CUDA reserved最大记录 | 54.805 GiB，和allocated重叠，不能相加 |
| 更新时采样的最小MemAvailable | 约49.00 GiB |
| checkpoint写入时一次MemAvailable采样 | 约25.95 GiB，并非全程最小值的证明 |
| 首步 / 末步loss | 0.34209 / 0.29512，全程有限 |
| 唯一DCP | 41,017,739,148 bytes |
| 最终HF权重及配置等 | 25,519,988,108 bytes |
| pilot目录磁盘分配 | 约66.55 GB |
| 整个比赛目录磁盘分配 | 约156.56 GB |
| 磁盘剩余 | 约2.80 TB |

1. 导出全部F32，动作投影相对原预训练最大绝对变化0.00025343894958496094，实际部署模型严格重载并精确核对此投影通过。
2. 首个optimizer step处于LR=0的官方warmup起点；20步是官方循环/优化器调用计数，后续非零LR已让权重变化。
3. 不同step使用不同样本，末步loss较低不能推断仿真成功率提高。本轮没有做动作推理或仿真，也没验证10000步收敛。
4. 所有试训/导出/重载进程已退出。当前20步权重是功能试训产物，未冒充最终参赛模型。

## 11. 待同伴和官方审核的问题

1. **指定初始化与训练范围：** 原始预训练起点、50任务clean联合训练和一个最终策略是否符合当前赛事完整规则；完整规则仍需取得。
2. **冻结与训练方法：** 冻结Qwen、训练动作专家及对齐模块、AdamW/LR/梯度累积等是否存在额外约束；未自行宣布合规或违规。
3. **注意力实现：** 允许使用eager及video教师SDPA后端吗；源码配置化补丁是否需提交说明。官方代码提供选项不等于赛事批准，数值逐位等价尚未验证。
4. **clean统计：** 是否接受只从clean数据重算统计、50步窗口加权和该归一化映射；来源清楚不等于方法获批。
5. **缺尾帧：** 是否接受训练时重复最后有效图像，而保持状态/动作标签不变；是否有主办方推荐数据修复版本。
6. **仿真版本：** SAPIEN 3.0.3、ARM mplib构建及OIDN 2.5.1与官方参考栈差异，尤其图像去噪变化是否影响正式评测可比性。
7. **评测流程与专家配额：** 固定seed、专家失败后同seed运行策略、41个异常槽位的处理是否满足正式规则；五次专家采集限制是否包括预检查，计数作用域是什么。
8. **并行评测：** 同权重、隔离会话的两路调度是否可用于正式提交；保留单路入口，但不宣称已获逐项确认。
9. **训练恢复：** 正式长训前宜验证真实优化器/DataLoader恢复，不额外增加未经授权的参数更新；目前这一项未做。
10. 以上审核项已经分别标明“实施事实”和“规则待确认”，不能因为20步通过就整体标注完全合规。

## 12. 10000步：两个保存方案比较（尚未执行）

### 12.1 统一比较条件

1. 两方案都从同一指定原始预训练权重开始，单rank、micro batch 1、累积8、同样的LR/损失/50任务clean数据；10000是optimizer更新次数，不是微批次数或仿真回合数。
2. 10000步采样80000个帧起点，约相当于548893帧的14.6%。每个起点包含50步动作监督，因此不能把14.6%直接解释为只学了14.6%的动作，亦不能视为已充分遍历全部示范。
3. 20步试训累积4，每次更新中位4.871秒；按累积8近似线性折算9.742秒/更新。**正式累积8尚未实测；这是外推，不是10000步测速。**
4. 10000步纯训练约27.06小时，建议安排约28–32小时供训练和保存/导出，并另留评测、故障恢复时间。该范围是排期余量，不是统计置信区间或保证。
5. checkpoint保存一次暂按38.16秒；当时1线程同步写入真实41.02 GB。长训时文件系统负载、内存回收与日志等可能让它变化。

### 12.2 方案对照

| 项目 | A：5000步保存一次，保留两份 | B：1000步保存一次，只留最新和次新 |
|---|---|---|
| 保存节点 | 5000、10000 | 1000、2000……10000 |
| 总保存次数 | 2 | 10 |
| 最终保留恢复点 | 5000、10000 | 9000、10000 |
| 最终DCP占用 | 约82.04 GB | 约82.04 GB |
| 最终再导出一次HF后 | 正式运行约107.56 GB | 正式运行约107.56 GB |
| 保存时瞬间DCP数量 | 本轮最多2份 | 新点提交后再删旧点，最多短暂3份 |
| DCP瞬间占用 | 约82.04 GB | 约123.05 GB |
| 本轮新目录最大文件量估计 | 约107.56 GB（含最终HF） | 约123.05 GB（训练时3份DCP；HF在训练结束后导出） |
| 写checkpoint的累计数据量 | 约82 GB，不等于最终额外空间 | 约410 GB，不等于同时存410 GB |
| 保存耗时合计估算 | 约76秒 | 约382秒 |
| 两方案保存耗时差 | 基准 | 多约305秒，即5.1分钟 |
| 正常完整运行训练计算量 | 相同 | 相同 |
| 一段无恢复点的最长训练量 | 5000步，约13.53小时 | 1000步，约2.71小时 |
| 故障损失示例 | 第4999步失败，首次恢复点尚未产生，可能重做几乎13.5小时 | 第4999步失败，可从4000步恢复，约重做2.7小时以内 |

说明：最大损失指模型/优化器有效恢复已验证、上一个checkpoint完好且未越过下一个提交点的普通中断场景；不是任意磁盘损坏的保证。若假定故障在间隔内均匀发生，平均丢失时间约为一半，即A约6.8小时、B约1.35小时；这是情景计算，不是对故障概率的预测。

### 12.3 推荐与保留用途

1. **推荐B：每1000步保存、只留最新和次新。** 多出的约5分钟不足纯训练时间的0.4%，最终恢复点空间相同，遇到中断时损失显著更小。
2. 两方案不会仅因保存频率不同就必然产生学习效果差异。应保持所有训练超参数一致；保存不是一次额外梯度更新。
3. B的9000/10000两个点适合恢复及末期比较，但不会保留5000步供研究。如果需要固定5000里程碑，应额外“固定保留”它，不能还宣称始终只有最新两份；这不属于本文两个原方案。
4. 若你只关心5000与10000的学习差异，A保留节点更符合这个目的，但要接受首次保存前约半天的进度风险。两方案都不通过训练loss自动选“最好模型”，因为当前没有独立验证指标。
5. 现有比赛目录已约156.56 GB（包含pilot）。正式运行再加107.56 GB后，总体约264.11 GB；B在三份DCP短暂共存时可能约279.61 GB。这里没有把pilot重复计算。另给缓存、小型日志和测量误差留余量。
6. 从训练root单独看，pilot约66.55 GB加正式输出，稳态约174.11 GB，B瞬间约189.61 GB，仍低于现有250 GiB（约268 GB）的预算检查值。实际大小依赖同一冻结范围和精度；改变训练方法需重算。
7. 三天连续纯训练的理论上限约26600步，但会被保存、初始化、导出、评测、故障和长训速度变化减少；本次讨论只计划10000步，不据此擅自延长训练。

### 12.4 正式启动前必须落实的代码和配置事项

1. 新建独立10000步run和派生配置，保留pilot、原2000步配置和旧基线。明确max_steps=10000，累积8，keep_checkpoints=2及chosen save_steps=1000或5000。
2. **启动器新增/启用固定保存间隔模式，关闭该运行的约30分钟自动校准。** 当前`launch.py`调用`checkpoint_schedule(...target_seconds=1800)`会将间隔缩短，单改YAML数字不会执行预期固定1000/5000。
3. 保留“最后一步必须保存”的官方逻辑，确保10000点产生；保存只在optimizer更新后发生，验证节点没有额外重存或越界更新。
4. 明确warmup：若沿用0.03比例，10000步对应300步warmup，而不是现有2000配置的60步。冻结最终LR、批量、步数、数据和统计后再启动。
5. 不直接把20步pilot或不同max_steps的旧配置当作同一实验接着训练；当前恢复签名保护也会拒绝。正式初始化和是否另作恢复验证需在启动前写清楚。
6. 用静态计数检查A恰好2次保存、B恰好10次保存，且清理只在新完整点提交后发生；评估B三份共存空间。不得用启动真实训练来替代这项检查。
7. 保持资源保护、FP32、eager、同一权重、1rank及0数据worker；未另行决定前，不临时加并行进程、改精度或切训练方法。
8. 本文交同伴审核；**当前没有开始10000步训练，也没有把建议方案自动写成正在运行的配置。**

## 13. 审核证据与参考入口

1. 试训证据：`runs/finetune-clean-v3-gb10/pilot/pilot-measurements.json`、`reload-verification.json`、`logs/updates.jsonl`、冻结配置和启动清单。
2. 环境与源码补丁：`docs/setup-baseline-20261015/*.patch`、原件备份和`docs/training-prep-20261016/{environment-before.txt,environment-after.txt,pip-check.txt,preparation-manifest.json,teammate-change-audit.json}`。
3. 归一化来源：快速基线`preparation/clean-normalization-provenance.json`和`clean-episode-task-map.json`。
4. 赛事入口：[赛题与数据](https://tianchi.aliyun.com/competition/entrance/532514/information)、[规则解读](https://tianchi.aliyun.com/forum/post/1081761)。这些是待完整核对的规则入口，不声称本文已取得全部正文。
5. 技术实现参考：[LingBot官方训练配置文档](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/vla/Training_Config.md)、[V3 clean更新说明](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/discussions/95)。技术代码支持某功能不等于赛事批准某方法。


## 附录A：派生配置的完整显式字段差异

对比的是项目留存的官方RoboTwin示例YAML解析结果和当前两份派生配置；“未指定”不等于运行时没有默认值。与原文件相同的字段不重复列出。stage1是现有2000步配置，不是拟议10000步配置。

| 字段 | 官方示例显式值 | 已实测pilot | 当前stage1 |
|---|---|---|---|
| `data.data_name` | `"multi"` | `"robotwin"` | `"robotwin"` |
| `data.image_augment` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `data.img_size` | 原文件未显式指定／该配置未设置 | `224` | `224` |
| `data.norm_stats_file` | 原文件未显式指定／该配置未设置 | `"/home/zzj/competitions/lingbot/configs/normalization/robotwin_clean_v3.json"` | `"/home/zzj/competitions/lingbot/configs/normalization/robotwin_clean_v3.json"` |
| `data.num_workers` | `8` | `0` | `0` |
| `data.pin_memory` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `data.prefetch_factor` | 原文件未显式指定／该配置未设置 | `1` | `1` |
| `data.robot_config_root` | `"./configs/robot_configs"` | `"/home/zzj/competitions/lingbot/configs/robots/clean-v3-gb10"` | `"/home/zzj/competitions/lingbot/configs/robots/clean-v3-gb10"` |
| `data.train_path` | `"assets/training_data/robotwin.txt"` | `"/home/zzj/competitions/lingbot/data/RoboTwin_lerobot_v30"` | `"/home/zzj/competitions/lingbot/data/RoboTwin_lerobot_v30"` |
| `eval.enabled` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `model.model_path` | `"/path/to/pretain_ckpt/hf_ckpt"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b"` |
| `model.tokenizer_path` | `"/path/to/Qwen3-VL-4B-Instruct"` | `"/home/zzj/competitions/lingbot/models/Qwen3-VL-4B-Instruct"` | `"/home/zzj/competitions/lingbot/models/Qwen3-VL-4B-Instruct"` |
| `model.vit_attn_implementation` | 原文件未显式指定／该配置未设置 | `"eager"` | `"eager"` |
| `train.align_params.depth.moge_path` | `"/path/to/depth/moge2-vitb-normal.pt"` | `"/home/zzj/competitions/lingbot/models/moge-2-vitb-normal/model.pt"` | `"/home/zzj/competitions/lingbot/models/moge-2-vitb-normal/model.pt"` |
| `train.align_params.depth.morgbd_path` | `"/path/to/depth/model.pt"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b/depth/model.pt"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b/depth/model.pt"` |
| `train.align_params.video.attention_mode` | `"flex_block_causal"` | `"sdpa_block_causal"` | `"sdpa_block_causal"` |
| `train.align_params.video.ckpt_path` | `"/path/to/dino_video/teacher_step_10000.pth"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"` |
| `train.align_params.video.config_path` | `"/path/to/dino_video/config.yaml"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b/dino_video/config.yaml"` | `"/home/zzj/competitions/lingbot/models/lingbot-vla-v2-6b/dino_video/config.yaml"` |
| `train.align_params.visual_steps` | `5000` | `1000000` | `1000000` |
| `train.allow_partial_checkpoint` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `train.async_save_hf_weights` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `train.attention_implementation` | `"flex_cached"` | `"eager"` | `"eager"` |
| `train.enable_activation_offload` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `train.enable_fsdp_offload` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `train.enable_gradient_checkpointing` | `false` | `true` | `true` |
| `train.enable_mixed_precision` | 原文件未显式指定／该配置未设置 | `true` | `true` |
| `train.freeze_vision_encoder` | `false` | `true` | `true` |
| `train.freeze_vit` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `train.global_batch_size` | `1024` | `4` | `8` |
| `train.gradient_accumulation_steps` | 原文件未显式指定／该配置未设置 | `4` | `8` |
| `train.lr` | `0.0001` | `2e-05` | `2e-05` |
| `train.lr_min` | `5e-05` | `1e-05` | `1e-05` |
| `train.lr_warmup_ratio` | 原文件未显式指定／该配置未设置 | `0.1` | `0.03` |
| `train.max_steps` | `50000` | `20` | `2000` |
| `train.micro_batch_size` | `32` | `1` | `1` |
| `train.moe_monitor_interval` | `1000` | `100` | `100` |
| `train.num_train_epochs` | `29000` | 原文件未显式指定／该配置未设置 | 原文件未显式指定／该配置未设置 |
| `train.optimizer` | `"muon"` | `"adamw"` | `"adamw"` |
| `train.output_dir` | `"/path/to/save_ckpt"` | `"/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/pilot"` | `"/home/zzj/competitions/lingbot/runs/finetune-clean-v3-gb10/stage1"` |
| `train.save_epochs` | `29000` | `0` | `0` |
| `train.save_hf_weights` | 原文件未显式指定／该配置未设置 | `false` | `false` |
| `train.save_steps` | `10000` | `20` | `250` |
| `train.seed` | 原文件未显式指定／该配置未设置 | `42` | `42` |
| `train.train_expert_only` | 原文件未显式指定／该配置未设置 | `true` | `true` |
| `train.use_compile` | `true` | `false` | `false` |
| `train.use_moe_expert_lr` | `true` | `false` | `false` |

共46项显式字段差异。运行时参数解析器还会填充默认字段；上述比较不替代最终冻结的CLI配置。

## 附录B：补丁与当前文件校验摘要

以下8份patch为服务器现存记录。完整文本保存在对应文件及`teammate-change-audit.json`；R2为自建控制代码的修订，不是官方任务判分补丁。

| 补丁（比赛根目录相对路径） | bytes | SHA256 |
|---|---:|---|
| `docs/setup-baseline-20261015/eval-client-unused-act-import.patch` | 461 | `ec452787c80ac8acbfe43e93e13463f430fe5e9d72a54b542ae646e5168e4b70` |
| `docs/setup-baseline-20261015/sapien-oidn-upgrade.patch` | 668 | `a85380242f49678ea96e38560e086d96067c0c839e87c586fed7db1eb63ccae0` |
| `docs/setup-baseline-20261015/configured-attention.patch` | 1204 | `3100beee6624bcce93922439774bda05f9cb812375b9ed802cfbd2a00cb31b4a` |
| `docs/setup-baseline-20261015/rgb-optional-open3d.patch` | 594 | `2a4335a0573079d8377dbbc3a2178d00914fa2c2f38a26da811bd859340f7a96` |
| `docs/setup-baseline-20261015/sim-native-import-order.patch` | 227 | `eab3ddb35cadb9f8939fe05c112171161f56ff5c18f441d4a66bbf868c4e8a99` |
| `docs/setup-baseline-20261015/moe-fallback-logger.patch` | 400 | `e9eefd5ca3eee5396398bafbe84b4aff0f9b02e88cfc733bf0c88a46617927b4` |
| `runs/pretrained-clean-quick-50x10-20261016/preparation/control-r2.patch` | 37571 | `d7368dc416a533215a25ad351ada28a46c7dca450761ad69083c74d397712c41` |
| `runs/pretrained-clean-quick-50x10-20261016/preparation/configurable-count.patch` | 1291 | `55f3c404f5ac33af894ef8da128a20d571059894254433360acb13ddd3f2209d` |

注意力源码补丁原文（可直接核对三处更改）：

```diff
--- upstream/modeling_lingbot_vla_v2.py
+++ local/modeling_lingbot_vla_v2.py
@@ -127,14 +127,14 @@
         vlm_config = AutoConfig.from_pretrained(self.config.tokenizer_path)
         if self.config.vocab_size not in (0, 257152):
             vlm_config.text_config.vocab_size = self.config.vocab_size
-        vlm_config._attn_implementation = "flash_attention_2"
-        vlm_config.text_config._attn_implementation = "flash_attention_2"
+        vlm_config._attn_implementation = self.config.vit_attn_implementation
+        vlm_config.text_config._attn_implementation = self.config.vit_attn_implementation
         vlm_config.vision_config._attn_implementation = self.config.vit_attn_implementation
         self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
         if self.config.use_lm_head:
             self.qwenvl.tie_weights()
 
-        self.config.qwen_expert_config._attn_implementation = "flash_attention_2"
+        self.config.qwen_expert_config._attn_implementation = self.config.vit_attn_implementation
         self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)
 
         if getattr(self.config, "adanorm_time", False):
```

当前关键文件SHA256：

| 文件 | SHA256 |
|---|---|
| `configs/training/clean-v3-gb10/pilot.yaml` | `ae421829a3dafc5711ef89e53e80847bbc25829d48636d0885ab2a05a9343c90` |
| `configs/training/clean-v3-gb10/stage1.yaml` | `b4f1fcb345d0e764e3cf14f123ff48c47f27d880e86ffdf4106d7843deab002d` |
| `configs/robots/clean-v3-gb10/robotwin.yaml` | `af23fd2e289af1edf5dc1671c79e0a2bb9172dc796af938db267760110d03cbf` |
| `configs/normalization/robotwin_clean_v3.json` | `0b5ed854065e9c64e1b8934c274c38d0efa062d743bfad071f1a089983125611` |
| `repos/lingbot-vla-v2/tasks/vla/train_lingbotvla.py` | `7896927842eea89793da4550cc7096e3c5f315e749f0bf8812cc51e1bae711d7` |
| `repos/lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml` | `ee915e4ad34589a6def35c9e3c12bb8ff7d9f768dfc7ad44ce789fa7aaea57e2` |
| `repos/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` | `b009a4a20fb4d345ecbbf4b9727e80f20884e6f468a1dcfefe7b3293d804fc80` |
| `repos/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py` | `67c1de355b3bac4d7ff940b9c43135e7897cfeaf1ef88224e1766bc1d1f28db9` |
| `repos/RoboTwin/script/eval_policy_client_lingbotvla.py` | `4212720b01ce9ee7ac0b846996e10ccfb99808e1cda1929e8089429870f4269e` |
| `repos/RoboTwin/envs/camera/camera.py` | `cc71bb922d63379840f3f33e62ecd0946dac74b3b49e11a4d60bd6b95abe8920` |
| `repos/RoboTwin/envs/utils/save_file.py` | `d615723e6b00949ee8d8fc2150f4566b57b4fa73539610054771029d7f725ce9` |

当前新增训练适配器SHA256：

| 文件 | SHA256 |
|---|---|
| `export_cpu.py` | `97319921aa60be0add0effd66420f49cb9b6246f89598b1f0bb6d45690966c2b` |
| `gb10_runtime.py` | `d68f5fdcc069a8e42aa1673039a4c432e49bc119946bfc5321b42335f2ba8948` |
| `launch.py` | `7e0a1bea7a0a8d39494251236187c68e143f8951b0567af894b8cf9dd0ef8f12` |
| `safe_video.py` | `4ee9b3894c34c974d4e3bc9c7eeab66a2de866333a1ef8f010d12c966f4e8ee1` |
| `storage_policy.py` | `112bd826e0b6266847c4cd0b8314bc776861b858e9343158d9d7107f755d7f6f` |
| `test_policies.py` | `753a14b07603c8ce22e629196eb1e00ad9c2548fb14444c9336270c2b0ae8c19` |
| `train_entry.py` | `a3a9f85f98ca3d488b9264dbcae8ebadb6cf6cea772cc6c72aa946c4d2dd1ac4` |
| `validate_data.py` | `e14d7af9887c5f2c74dcaca69421aa8e4f7c2771c1d144ea39ae0a400314e134` |
| `validate_native_loader.py` | `5922da3a0b4473b4cec421c23e282992a00fb59f45f877afe836fd706644098c` |
| `verify_reload.py` | `97ab106331379f4b7905f3b0f32e465995bcaf85137dce3bcb18890ca2333f2d` |

## 附录C：已安装关键包与剩余依赖声明问题

| 环境 | 包 | 实际版本 |
|---|---|---|
| `vla` | `accelerate` | `1.7.0` |
| `vla` | `av` | `15.0.0` |
| `vla` | `datasets` | `3.6.0` |
| `vla` | `lerobot` | `0.4.2` |
| `vla` | `lxml` | `5.4.0` |
| `vla` | `matplotlib` | `3.10.3` |
| `vla` | `numpy` | `1.26.4` |
| `vla` | `numpydantic` | `1.9.0` |
| `vla` | `optree` | `0.14.1` |
| `vla` | `peft` | `0.15.2` |
| `vla` | `pyarrow` | `21.0.0` |
| `vla` | `pycryptodomex` | `3.23.0` |
| `vla` | `torch` | `2.9.1+cu130` |
| `vla` | `torchdata` | `0.11.0` |
| `vla` | `torchvision` | `0.24.1` |
| `vla` | `transformers` | `4.57.3` |
| `vla` | `triton` | `3.5.1` |
| `vla` | `utils3d` | `1.3` |
| `sim` | `lxml` | `6.1.3` |
| `sim` | `matplotlib` | `3.11.2` |
| `sim` | `mplib` | `0.2.0` |
| `sim` | `numpy` | `1.26.4` |
| `sim` | `nvidia_curobo` | `0.7.8` |
| `sim` | `sapien` | `3.0.3` |
| `sim` | `torch` | `2.9.1+cu130` |
| `sim` | `triton` | `3.5.1` |
| `sim` | `warp-lang` | `1.12.0` |

最近保存的pip check输出（补依赖后的记录，说明尚有声明冲突；20步成功不等于全部包功能通过）：

```text
lerobot 0.4.2 requires cmake, which is not installed.
lerobot 0.4.2 requires deepdiff, which is not installed.
lerobot 0.4.2 requires diffusers, which is not installed.
lerobot 0.4.2 requires draccus, which is not installed.
lerobot 0.4.2 requires gymnasium, which is not installed.
lerobot 0.4.2 requires pynput, which is not installed.
lerobot 0.4.2 requires pyserial, which is not installed.
lerobot 0.4.2 requires rerun-sdk, which is not installed.
lerobot 0.4.2 requires termcolor, which is not installed.
utils3d 1.3 requires moderngl, which is not installed.
lerobot 0.4.2 has requirement accelerate<2.0.0,>=1.10.0, but you have accelerate 1.7.0.
lerobot 0.4.2 has requirement datasets<4.2.0,>=4.0.0, but you have datasets 3.6.0.
lerobot 0.4.2 has requirement setuptools<81.0.0,>=71.0.0, but you have setuptools 84.0.0.
lerobot 0.4.2 has requirement torch<2.8.0,>=2.2.1, but you have torch 2.9.1+cu130.
lerobot 0.4.2 has requirement torchvision<0.23.0,>=0.21.0, but you have torchvision 0.24.1.
numpydantic 1.9.0 has requirement numpy>=2.0.0, but you have numpy 1.26.4.
```

## 附录D：时间和空间计算公式

```text
t_update_est = median(pilot steps 2..20) × (8 / 4)
             = 4.8711112086 × 2 = 9.7422224171 s
T_compute(10000) = 10000 × t_update_est / 3600 = 27.0617 h
T_save_A = 2 × 38.1626892 = 76.3254 s
T_save_B = 10 × 38.1626892 = 381.6269 s
delta = 305.3015 s = 5.0884 min
retained_DCP = 2 × 41,017,739,148 = 82,035,478,296 bytes
retained_DCP + final_HF = 107,555,466,404 bytes
B_transient_3_DCP = 123,053,217,444 bytes
```

保存耗时、不同累积批次的速度、长训稳定性和缓存增长均可能变化；这些公式明确区分已测输入与外推结果。GB为10^9 bytes，GiB为2^30 bytes。

机器审计时间UTC：`2026-10-16T04:17:19.654363+00:00`。审计时无训练进程：`True`。本报告生成不启动训练。
