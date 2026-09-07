# TDI PRC 选模方向修正

2026-09-07 的诊断包确认：75 个 TDI 任务全部按最小验证集 PRC 保存模型，
没有一个保存了已观察到的最大 PRC；39 个选中了 epoch 0。训练损失在 75 个任务中均下降。
原版零 MCC 描述的是错误选模输出，不能据此判断正确训练/选模的神经模型能力。

## 已定位的原因

Chemprop 2.3.1 的 `BinaryAUPRC` 继承 TorchMetrics 的 `BinaryPrecisionRecallCurve`，
后者的 `higher_is_better` 是 `None`。Chemprop CLI 将这个值当作假值，
同时为最佳模型保存与早停选择 `mode="min"`。PRC 面积应当最大化。

代码依据：

- [Chemprop 2.3.1 指标实现](https://github.com/chemprop/chemprop/blob/v2.3.1/chemprop/nn/metrics.py)
- [Chemprop 2.3.1 训练 CLI](https://github.com/chemprop/chemprop/blob/v2.3.1/chemprop/cli/train.py)
- [TorchMetrics 1.9.0 PR 曲线元数据](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/src/torchmetrics/classification/precision_recall_curve.py)

例如，单任务 CYP3A4、fold 0、seed 20260829 的曲线在 epoch 3 达到 PRC 0.420981，
实际恢复的却是 epoch 0、PRC 0.259299 的 checkpoint。这里比较的是训练过程中
已经记录的内部验证分数，不是用外层测试标签重新挑选阈值。

诊断同时核验了导出包 576 个文件的哈希、75 份完成记录和它们已提供的输入/输出工件，
以及 15 个 TDI 数据规格的 45 个分区：有标签样本、原始标签、内部验证/外层测试归属、
分区互斥和训练/验证两类均存在。摘要与诊断包均未包含原模型权重，权重复核留在运行机器执行。

此前的 PASS 检查覆盖工件完整性和指标重算，未覆盖选模方向，不能作为这一行为正确的证据。
这是审计覆盖缺口；新流程补上实际回调测试与逐任务模型选择验证。

## 本次修改范围

- 显式将 PRC 的 `higher_is_better` 设为 `True`，使保存模型和早停均最大化验证 PRC。
  修正在单个 Python 训练进程中应用，记录脚本、原包源文件哈希和依赖版本。
- 原 BCE、掩码、均匀采样、网络、100 epoch 上限、patience 15、5 个种子、
  数据划分与 0.5 阈值保持原定义。PRC 仍为原库的面积计算；多任务中仍汇总所有有标签项。
- 重跑 75 个 TDI 任务。125 个回归任务先检查原工件及最小 MAE 选模，然后复用。
- 每个新任务保留 best/last checkpoint 和 best.pt，额外生成验证集预测，
  从保存模型的概率独立重算 PRC；不再丢弃用于选模检查的 checkpoint。
- 新结果使用 `.runtime/frozen-experiment/.runtime/neural-prc-max-v3/`。
  原冻结源码、配置、v2 任务和 v2 汇总不被覆盖。
- 本次是查看过 v2 外层结果后的实现纠错，记录为 v3。它不是新增算法贡献；
  原 TDI 缺陷运行保留为历史记录，修正后的分数需实际运行后报告。

## 直接运行

这条入口针对已完成并上传诊断包的冻结 v2 运行，复用已有 Python 环境。使用原 RTX 4090：

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && git pull --ff-only origin main && bash scripts/start_tdi_prc_repair_4090.sh
```

进程在后台运行，日志使用独立文件：

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && tail -n 40 -F .runtime/tdi-prc-repair.log
```

状态位于 `.runtime/tdi-prc-repair-status.txt`。显示 PID 只表示启动；
需等待 75 个修正任务和独立审计完成后的 PASS。重复启动受共享实验锁保护。
同版本、同输入续跑会验证并复用已完成的新任务；失败尝试保留。

程序先检查原 200 个任务、分区和回归选模，再运行真实 Torch/Lightning 自检：
对已知分数序列 `[0.2, 0.8, 0.4, 0.3]`，旧方向必须复现选 epoch 0，
修正后必须选 epoch 1，且早停时点相符。自检失败会在正式复跑前停止。

训练结束后逐任务检查最高 PRC、早停、checkpoint 的 epoch 和回调方向，
核对 best.pt 与所选 checkpoint 的权重完全一致，并从验证预测独立复算 PRC。
全部通过后，合并 125 个回归任务的原始预测与 75 个修正 TDI 任务的预测，
再次核对 15,340 行覆盖、420 行指标与 6 行种子汇总；逐任务标明来源。

最终自动生成仓库目录下的 `CYP_neural_v3_review.zip`；若同名文件已存在，会创建带时间戳的新包。
上传该包供结果复核。完整模型与日志保留在上面的新结果目录；同时保留 v2 完整备份。

## 当前验证状态与下一阶段

本地 7 项回归测试通过；新选模审计在实际上传的全部 75 条训练曲线上均拒绝错误的最小 PRC 选择，
45 个 TDI 分区的独立只读审计通过。当前审查环境未安装 Torch，也没有用户的 4090，
所以真实回调自检、验证集重新推断及 75 任务训练由上述入口在 AutoDL 执行，结果尚未产生。

正确基线完成后，再进入 assay-structured 主模型最小原型及核心消融。
当前回归结果保留为强基线；TDI 也继续与已完成的传统树模型比较。
不能预先承诺纠错后 MCC 提高多少，也不能将基线纠错写成主模型创新或外部盲测成功。
