# TDI 训练记录导出

当 200 个神经基线任务和结果审计完成后，可导出 TDI 的训练记录，用来检查训练曲线、
早停和最佳模型选择。MCC 为 0 时，先检查概率分布与这些记录，再判断原因。

在 AutoDL 的分发仓库运行：

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && git pull --ff-only origin main && .runtime/frozen-experiment/.venv/bin/python scripts/export_tdi_diagnostics.py
```

输出为仓库目录下的 `CYP_TDI_diagnostics.zip`；同名文件已存在时，自动使用带时间戳的新文件名。
将终端打印的 ZIP 下载后提供给审查者。导出使用标准库，无需 GPU，也不会启动训练。

导出包含 75 个成功 TDI 任务的完成清单、训练日志、预测文件、训练/验证/测试 CSV、
实际生成的配置/指标曲线文件和 TensorBoard 事件文件，以及冻结配置、划分和预处理清单。
只收集完成清单指向的成功尝试。对清单中记录了哈希的文件核验 SHA-256，
并为导出的所有文件写入 `TDI_EXPORT_MANIFEST.json`。

模型权重继续保留在完整实验备份中；这个诊断包不能替代完整备份。仅凭摘要包或本诊断包，
不能独立重验未提供的模型权重。导出不会更改冻结实验、结果、标签或 0.5 的分类阈值。

如果后续需要调整损失、早停或阈值规则，应登记为新协议，使用训练/内部验证数据作选择，
保留原版本结果。外层测试标签不能用于挑选新阈值，也不能将改进后的分数写回原冻结结果。
