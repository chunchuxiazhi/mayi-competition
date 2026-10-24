# 10000步与20000步推理权重

两份模型分别来自50任务clean联合微调的10000步和20000步HF导出，每份约25.52GB，均包含完整推理张量、tokenizer和processor。它们不含优化器等续训状态。

普通Git仓库只保存清单、配置、模型索引和少量元数据；大张量放在对应GitHub Release中。每个资产最多1,500,000,000B，低于2GiB限制。同一个原始safetensors文件可能拆成多个part；分片必须按manifest中的顺序合并，不能直接把part当成完整safetensors加载。

当对应Release正式发布后，从仓库根目录运行：

```bash
python tools/download_model.py --step 20000 --output model-downloads/step_20000
# 或
python tools/download_model.py --step 10000 --output model-downloads/step_10000
```

恢复后的模型文件直接位于output目录。下载脚本逐个分片校验、合并并再次核对整文件SHA256；已验证的完整文件会跳过，已下载并验证的完整part可恢复使用。临时文件位于指定output目录内，不修改服务器原始模型。

每份模型至少预留约28GB磁盘；两份至少约55GB，实际应留更多余量。脚本只保留当前下载part和当前合并文件，不要求再存一套完整的模型分片副本。断网后可重复同一命令。

仅检查本地清单，不下载：

```bash
python tools/download_model.py --step 20000 --check-only
```

`step_10000/manifest.json`与`step_20000/manifest.json`记录原始文件、每个分片的偏移、尺寸、SHA256及Release标签。Release还含hf-metadata.zip、model-manifest.json和SHA256SUMS.txt。

来源：`robbyant/lingbot-vla-v2-6b`，没有使用官方RoboTwin后训练模型。训练与推理设置见根README和复现说明。10000步完整续训checkpoint已按先前保留策略清理，现存推理导出不能冒称具有完整恢复状态。
