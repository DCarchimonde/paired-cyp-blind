# Paired CYP Blind Validation

公开数据驱动的 CYP 抑制预测研究。当前已实现并审查的是 **200 个神经网络基线任务及其独立审计**。
论文主模型、消融实验、外部盲测提交和完整论文仍未完成；不承诺新颖性、顶刊录用或零错误。

## 直接运行：两个独立命令

在 AutoDL 的 Linux 终端执行。使用一张 RTX 4090，数据盘首次安装前至少有 30 GiB 空间，
并能连接 GitHub、PyPI、Python 下载源和 Hugging Face。使用独立目录 `paired-cyp-blind-github`。

第一条：首次克隆，已有目录则拉取更新。训练期间不要更新代码。

```bash
cd /root/autodl-tmp && if [ -d paired-cyp-blind-github/.git ]; then git -C paired-cyp-blind-github pull --ff-only origin main; else git clone https://github.com/DCarchimonde/paired-cyp-blind.git paired-cyp-blind-github; fi
```

第二条：校验冻结版本，在后台执行全部已实现的基线任务和审计。

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && bash scripts/start_reviewed_neural_baselines_4090.sh
```

脚本自动恢复冻结的实验代码、安装隔离环境、核对数据和 GPU、运行测试、检查划分，
再完成 200 个训练任务、汇总和独立指标审计。终端断开后可继续运行；重复启动不会并发开两套任务。
失败后重新执行第二条会复用验证通过的完整任务；被中断的单个任务会从头重跑。
机器关机、实例被释放或磁盘丢失会中断任务。

## 进度与结果

- 日志：`.runtime/neural-run.log`
- 退出状态：`.runtime/run-status.txt`
- 结果入口：`reports/neural_v2/`

这些入口链接到 `.runtime/frozen-experiment/` 内的实际实验文件。
只有全部 200 个任务通过验证，且 `neural_result_audit.json` 的 `overall_pass` 为 `true`，
才能认定基线阶段完成。出现 PID 或启动成功提示不代表实验已经完成。

释放实例前，保存实际目录 `.runtime/frozen-experiment/reports/neural_v2/` 的完整内容，
包括模型、日志和清单；不能只复制符号链接。结果不会自动推送到 GitHub。

## 已完成的验证

冻结实验版本的 46 项回归测试、17 项真实数据输入审计、合成数据 smoke test 和真实数据
三轮训练检查均已通过。历史 Day 1–2 审计为 43 PASS / 1 WARN，经典基线 13/13、
split-gap 11/11 通过。RTX 4090 全量实验、总耗时与租赁成本尚未实测。

审查修复了缺失标签类别平衡采样、续跑清单哈希漂移、旧结果混入、
不完整文件校验和 NaN 指标漏检问题。当前 TDI 比较统一使用不重采样、屏蔽缺失标签的 BCE；
架构、划分、种子和阈值保持冻结。保留所有不利结果，不按结果更改协议。

## 版本与复现

GitHub 插件生成的是分发提交。实际执行的原始冻结提交为：
`bc00641390c559ec696506d6fd07edb77c514c13`，
标签为 `neural-protocol-hardened-20260906`。
完整旧历史和标签保存在 `releases/paired-cyp-blind-reviewed-20260906.bundle`；
启动入口会校验 SHA-256、提交、标签和工作区状态，再运行该冻结版本。
源代码和配置在本仓库中也完整保留，便于查看。

- [详细审查记录](docs/PRE_PUSH_REVIEW_20260906.md)
- [4090 运行说明](docs/NEURAL_4090_RUNBOOK.md)
- [GitHub 分发与冻结执行的对应关系](docs/GITHUB_DELIVERY.md)
- [研究方案及边界](docs/PROJECT_BLUEPRINT.md)

本项目与 ProteinMPNN、RACER-C 使用独立代码、环境和结果。
TDI 指挑战数据中按实验定义的 IC50 位移，不等同于共价机制抑制或临床药物相互作用。
