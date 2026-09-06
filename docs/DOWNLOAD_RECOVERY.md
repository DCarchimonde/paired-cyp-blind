# 首次依赖下载恢复（2026-09-06）

用户的 AutoDL 日志显示：初始 `uv sync --frozen --extra dev --extra neural`
已运行约 4 小时 54 分钟，缓存目录 1.4 GiB，虚拟环境只有 72 KiB。
这表明安装尚未完成；缓存体积包含下载与解压产物，不能当作下载完成百分比。
仓库锁文件包含原始 PyPI 文件地址。修改系统 pip 源不会修改这些冻结地址。

## 两条命令

第一条，在 AutoDL 上更新外层分发仓库。冻结实验位于独立的内部 Git 仓库，
本次更新只增加安装恢复工具、测试与说明。

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && git pull --ff-only origin main
```

第二条，发起后台恢复并立即显示实时日志。

```bash
cd /root/autodl-tmp/paired-cyp-blind-github && bash scripts/recover_downloads_4090.sh && tail -n 40 -F .runtime/neural-run.log
```

Ctrl+C 只退出上述日志查看。后台恢复和后续任务可在终端断开后继续；实例关机则会中断。
不要手工删除环境、清空缓存或修改冻结的 `uv.lock`。

## 恢复入口执行的操作

1. 校验原冻结提交、标签、工作区和 Python 3.12.13。
2. 如果运行锁仍被占用，只识别同一目录下、由记录中的原工作流直接启动的
   `uv sync --frozen --extra dev --extra neural`。使用 Linux PID handle 再次核验后终止该安装子进程。
   若工作流已经进入数据处理、测试或训练，拒绝中断；也不会操作其他项目的进程。
3. 等待原工作流释放锁。保留完整缓存并尝试离线安装其中可用的依赖。
   未完成的大包可能需要重新下载；不承诺未完成文件能按字节续传。
4. 从原锁文件导出精确版本与 SHA256 清单。依次尝试清华和阿里云 PyPI 镜像，
   使用 `uv pip sync --require-hashes --only-binary :all:` 安装。
   每次镜像安装最多 15 分钟，每 30 秒打印存活提示；超时终止该次安装并保留完整缓存。
   提示是进程状态，不是下载百分比或完成承诺。
5. 执行原 `uv sync --frozen`，随后要求其能在离线模式下成功运行，并检查依赖完整性。
   再次核对冻结源代码和锁文件未变。通过后恢复原始 200 个基线任务和全部既有审计。
   仅 uv 的依赖管理进入离线模式，原始 Python 数据下载步骤仍可联网。

恢复日志追加到原 `.runtime/neural-run.log`；旧日志保留。
状态写入 `.runtime/run-status.txt`。安装恢复证据与每个离线缓存检查的日志位于
`.runtime/download-recovery/<时间与进程编号>/`。这些实际路径都相对于冻结实验目录。
中间可能看到旧安装进程的 `FAILED (exit 143)`，之后应出现恢复阶段日志；
旧进程退出信息不能单独当作恢复失败的结论。

`DEPENDENCY RECOVERY PASS` 只代表依赖恢复通过。基线实验仍须原始 200 个任务和
`neural_result_audit.json` 的 `overall_pass: true` 全部满足，才能认定完成。

## 验证与范围

- 恢复与原始 bundle 导入相关测试：15 通过；3 项原生 Linux 进程交接测试因当前
  验证环境的进程编号与 `/proc` 不对应而跳过。这些测试保留，原生 Linux 环境可以运行。
- 使用锁定的 `idna==3.19` 实际从阿里云镜像安装，要求匹配原 SHA256；
  随后清空测试缓存位置，用原 PyPI 锁文件进行离线同步成功，锁文件未变。
- 实测原始 `uv sync --frozen` 生成的缓存可能没有包索引元数据，普通的离线
  `uv pip install` 无法找到它。恢复入口因此先用原锁文件进行部分离线同步，
  该缓存复用路径已实测通过；错误 SHA256 的镜像安装请求也已确认被拒绝。
- 进程命令、父进程、工作目录不匹配时拒绝操作；安装子进程的总超时与退出回收均有测试。
- 本地测试不能证明用户 AutoDL 到镜像的实时速度，也不能替代全量 GPU 实验。

本次不修改 bundle、实验配置、数据划分、模型、阈值、种子或指标规则。

参考：[AutoDL 软件源说明](https://www.autodl.com/docs/source/)、
[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、
[uv 锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)。
