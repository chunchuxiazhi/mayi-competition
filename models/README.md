# 推理权重与下载校验

本项目保留10,000步和20,000步两个模型的完整 HF推理导出。两者均从指定原始预训练模型 `robbyant/lingbot-vla-v2-6b` 出发，在50任务 clean数据上联合微调，未使用官方 RoboTwin后训练模型作为起点。

**本次提交采用20,000步权重**，与 clean＋seen 和 randomized＋unseen结果对应。10,000步权重用于历史对照。每份模型约25.52GB，含六个 safetensors、索引、模型配置、tokenizer和 processor；不含优化器等完整续训状态。

## 分发方式

普通 Git仓库只保存清单和小型元数据，大张量通过对应 GitHub Release分片发布。20,000步 Release已完成上传校验；10,000步 Release在全部资产上传并校验后公开，以页面可见状态为准。

每个分片最多1,500,000,000字节。一个原始 safetensors可能分成多个 part，须按 manifest规定顺序合并；part文件不能直接用于模型加载。

## 下载并恢复 HF目录

在仓库根目录运行：

```bash
python tools/download_model.py --step 20000 --output model-downloads/step_20000
# 下载历史对照模型，需对应Release已公开
python tools/download_model.py --step 10000 --output model-downloads/step_10000
```

脚本逐个校验分片、合并原文件，并再次核对完整文件 SHA256。已校验的完整文件会跳过；已下载并通过校验的 part可继续使用。下载中断后可重复相同命令。临时文件位于指定 output目录，不修改服务器原始模型。

每份模型至少预留约28GB磁盘，两份至少约55GB，并保留额外余量。该流程同时保留当前合并文件及当前分片，不需要再保存一整套分片副本。

只检查清单、不访问模型下载：

```bash
python tools/download_model.py --step 20000 --check-only
```

`step_10000/manifest.json`、`step_20000/manifest.json`记录原始文件、分片偏移、尺寸、SHA256和 Release标签。Release还提供 `hf-metadata.zip`、`model-manifest.json`及 `SHA256SUMS.txt`。

## 使用范围

HF目录可用于推理部署。它不含 Adam、数据加载器游标或随机数恢复状态，不能用于完整训练状态恢复。10,000步 DCP已按此前保留策略清理；20,000步 DCP保留在原服务器，不属于本次公开模型材料。
