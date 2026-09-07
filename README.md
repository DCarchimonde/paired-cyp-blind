# Paired CYP Blind Validation

公开数据驱动的 CYP 抑制预测研究。当前已实现并审查的是 **200 个神经网络基线任务及其独立审计**。
论文主模型、消融实验、外部盲测提交和完整论文仍未完成；不承诺新颖性、顶刊录用或零错误。

## 已完成 v2：修正 TDI 选模方向

2026-09-07 的训练记录复核确认，原 75 个 TDI 任务错误地最小化验证集 PRC，
其零 MCC 属于有实现缺陷的历史运行。修正入口会先验证并复用 125 个回归任务，
再以 PRC 最大化重跑 75 个分类任务，独立审计后导出新的结果包。
详见[原因、验证状态与修正说明](docs/TDI_PRC_SELECTION_REPAIR.md)。

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && git pull --ff-only origin main && bash scripts/start_tdi_prc_repair_4090.sh
```

新日志：`.runtime/tdi-prc-repair.log`。完成后上传 `CYP_neural_v3_review.zip`。
修正结果尚需实际复跑；原 v2 文件保留，不能把原 TDI 零分当作已正确选模的基线结果。
若第一项修正任务报 `Last checkpoint differs from the final training epoch`，
拉取更新后使用同一入口续跑：新版本核对 Lightning 2.6.5 的实际保存行为，
验证已有模型及重新推断的一致性后恢复该任务。详细恢复规则见上方说明。

## 历史 v2 准备与复现：两个独立命令

以下为原 v2 运行说明，包含上述已确认的 TDI 选模缺陷。已经完成 v2 的用户使用上方修正入口。

在 AutoDL 的 Linux 终端执行。使用一张 RTX 4090，数据盘首次安装前至少有 30 GiB 空间，
并能连接 GitHub、PyPI 和 Python 下载源。冻结的公开数据已随仓库提供，
原始文件逐一核验 SHA256 后使用，无需从运行机器再次访问 Hugging Face。
使用独立目录 `paired-cyp-blind-github`。

第一条：首次克隆，已有目录则拉取更新。训练期间不要更新代码。

```bash
cd /root/autodl-tmp && if [ -d paired-cyp-blind-github/.git ]; then git -C paired-cyp-blind-github pull --ff-only origin main; else git clone https://github.com/DCarchimonde/paired-cyp-blind.git paired-cyp-blind-github; fi
```

第二条：校验冻结版本，在后台执行全部已实现的基线任务和审计。

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && bash scripts/start_reviewed_neural_baselines_4090.sh
```

脚本自动恢复冻结的实验代码与公开数据、安装隔离环境、核对数据和 GPU、运行测试、检查划分，
再完成 200 个训练任务、汇总和独立指标审计。终端断开后可继续运行；重复启动不会并发开两套任务。
失败后重新执行第二条会复用验证通过的完整任务；被中断的单个任务会从头重跑。
机器关机、实例被释放或磁盘丢失会中断任务。

## 进度与结果

如果停在 `[1/200]`，且任务 `train.log` 中出现 `AF_UNIX path too long`，
按[通信路径修复说明](docs/SOCKET_PATH_RECOVERY.md)恢复。修复入口核对出错任务后停止其进程，
通过短路径重新调用原冻结脚本；已有环境和历史任务保留。正常启动入口也会先检查真实通信，
环境已有 PyTorch 时会验证四个数据加载进程的张量传输。

无卡模式可准备数据，正式训练须使用 RTX 4090。
如果环境已安装完成，但在 `FETCH cyp-challenge-TEST-BLINDED.csv` 处连接失败，
按[数据准备与开机模式说明](docs/DATA_PREPARATION.md)操作，已安装环境可继续使用。

如果首次安装长时间停留在 `Downloading`，可使用
[下载恢复说明](docs/DOWNLOAD_RECOVERY.md)中的两条命令。
恢复入口会检查并停止本项目的初始安装进程，复用可用的完整缓存，
用镜像下载同版本、同哈希的依赖，再自动接回原冻结实验流程。
每个镜像有 15 分钟总时限；已经进入训练的任务不会被该入口中断。

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
split-gap 11/11 通过。已有 RTX 4090 的 200 任务记录；复核发现 TDI 选模方向缺陷，
正在通过 v3 纠错复跑。上述原审计 PASS 不覆盖该选模行为，总租赁成本未核定。

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
