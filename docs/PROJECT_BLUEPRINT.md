# AIDD 代表作选题审计与执行蓝图

**公开检索截止日：2026-08-28**  
**项目状态：条件式 GO；先过 14 天 kill test，再决定是否作为代表作投入**

## 0. 先给结论

建议唯一主攻方向是：

> **面向 OpenADMET CYP 抑制盲测，建立一个由实验设计约束的概率模型：把 −NADPH 条件下的直接抑制视为母体效应，把 +NADPH 相对 −NADPH 的变化视为代谢依赖性潜在位移；由同一个潜变量模型同时产生直接抑制 pIC50、正向 TDI 概率与不确定性，并接受按化学类似物家族划分的真实前瞻盲测。**

暂定论文题目（先不用容易撞名的缩写）：

> **Mechanism-consistent learning of direct and time-dependent CYP inhibition under prospective analog-series validation**

一句话科学问题：

> **比起把 CYP 直接抑制和 TDI 当作两个互不相关的预测任务，显式建模 ±NADPH 成对实验、检测下限和曲线拟合不确定性，能否在真实类似物扩展盲测中获得更可靠的外推？**

这不是“再训练一个 GNN”。真正的主张是三件事同时成立：

1. **实验结构进入模型**：TDI 标签由潜在 pIC50 位移的正向尾部导出，而不是另接一个没有物理关系的分类头。
2. **实验不确定性进入似然和评价**：使用数据中可获得的 DRC 置信区间/标准差，并正确处理 pIC50 检测下限与区间删失。
3. **验证结构匹配真实用途**：内部验证模拟“高活性母核 → 最近类似物”扩展，最终使用 750 个从未公开标签的化合物做前瞻盲证。

### 必须诚实说明的边界

- **无法证明全世界“没有任何人正在做”**：竞赛参与者、未公开手稿和企业内部项目不可检索。能建立的是“截至 2026-08-28，在论文、预印本、公开代码、数据卡和可检索专利中，未发现相同数据、相同潜变量定义和相同前瞻验证组合”的**可辩护新颖性**。
- **不能保证 Nature 子刊**。普通模型加排行榜结果，合理目标是本领域高水平期刊；只有在盲测表现、统计证据和可迁移方法论三项都非常强时，才值得尝试 Nature Machine Intelligence 或 Nature Communications。
- **“代表作”资格由数据决定**。14 天 kill test 不过，就停止包装，不用半年时间维护一个无效故事。

## 1. 为什么现在值得做

### 1.1 数据刚出现，且自带真正的未知测试集

[OpenADMET CYP challenge 数据集](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)以 Apache-2.0 发布，当前包括：

| 数据块 | 规模 | 可用信号 | 在本项目中的角色 |
|---|---:|---|---|
| direct inhibition | 4,905 个化合物 | 四个 CYP 的直接抑制 pIC50、95% CI、标准差；标签稀疏 | 主回归任务与噪声建模 |
| blinded test | 750 个化合物 | 只有结构，四个 CYP 标签全盲 | 最终前瞻证据 |
| TDI | 6,145 个化合物 | CYP3A4/2D6 TDI 标签、+NADPH pIC50、配对直接 pIC50 | 潜变量位移与 TDI 主任务 |
| single concentration | 17,504 条测量 | 4,376 个化合物 × 四个 CYP 的单浓度信号 | 低成本辅助任务 |
| Emax | 6,146 个化合物 | 直接/TDI Emax 与置信区间 | 辅助任务与曲线质量信息 |

数据卡报告总计约 35,450 行、6.67 MB，完全适合单张 RTX 4090；不需要分子动力学或大规模 3D 预计算。

### 1.2 测试集不是随便随机切出来的

[挑战设计说明](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)显示：训练集来自 Enamine DDS10 与 FDA 批准药的初筛和 DRC；测试集由 CYP1A2、2C9、3A4 的高活性命中物出发，每个命中物采购最相似的 10 个 Enamine 化合物，形成 750 个类似物扩展集合，并对四个 CYP 全部测量。此设计比随机划分更接近药化中的 lead expansion。

### 1.3 TDI 是真实监管问题，不是为了造论文的标签

[ICH M12/FDA DDI 指南](https://www.fda.gov/media/161199/download)明确要求评估主要 CYP 的可逆和时间依赖性抑制。OpenADMET 采用 ±NADPH 预孵育的 IC50-shift 设计；大于 2 倍位移对应 \(\log_{10}2=0.301\) 的 pIC50 增量。挑战对直接抑制使用 MA-ST-RAE，对 CYP3A4/2D6 的 TDI 使用 MCC，截止日为 2026-11-03。[FDA 对 NAMs 的说明](https://www.fda.gov/science-research/science-and-research-special-topics/new-approach-methodologies-nams)也明确把 in silico modeling 纳入新方法学范围。

这只说明研究有现实价值，**不等于模型能够替代法规要求的实验**。

## 2. 新颖性审计：最接近的公开工作

### 2.1 直接近邻

| 公开工作 | 已经做了什么 | 与本方案的边界 | 结论 |
|---|---|---|---|
| [Faramarzi et al., Frontiers in Pharmacology 2025](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2024.1451164/full) | 10,129 个化合物；分别建立 CYP3A4 TDI 和 3A4/2C9/2C19/2D6 可逆抑制 QSAR，并整理 MBI alerts | 不是同批 ±NADPH 成对 DRC 的潜变量模型；不是当前盲测；未统一建模单浓度、Emax、CI 与删失 | **近邻，必须引用；不构成同题** |
| [Fluetsch et al., Chemical Research in Toxicology 2024](https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00305) | CYP3A4 TDI 深度学习与实验变异比较 | 单一 CYP3A4、企业数据；没有四 CYP 多保真结构与当前类似物盲测 | **近邻，不能宣称首次 TDI 深度学习** |
| [Xu et al., Molecular Pharmaceutics 2022](https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.2c00571) | 潜在 CYP3A4 TDI 分类模型 | 单任务分类；没有成对连续潜变量和盲测 | **近邻，边界清晰** |
| [Ivanov et al., Pharmacophore 2026](https://doi.org/10.51847/7FS6HA9tSy) | 提出共享图编码器联合抑制、TDI 与代谢软位点的概念框架 | 文中明确是概念性“would/could”，不是当前数据的实证；但抢占了“统一多任务 CYP 模型”这个宽泛说法 | **因此不得把“多任务”当核心创新** |
| [OpenADMET 官方教程](https://github.com/OpenADMET/CYP-Challenge-Tutorial) | LightGBM 等起始流程、官方任务和指标 | 直接抑制和 TDI 按常规任务处理；没有本方案的成对观测模型 | **官方基线** |
| [公开参赛者派生数据与 EDA](https://huggingface.co/datasets/xX-its-amit-Xx/cyp-challenge-derived) | 外部 PubChem/ChEMBL 整理、缺失率/相关性/噪声底、基础多任务实验 | 已公开指出 CYP2D6 与其他 CYP 相关性弱、必须 mask 缺失；未公开成对潜变量/删失似然方案 | **必须引用其数据发现；不能把“mask”和“分开 2D6”说成创新** |

### 2.2 当前可守住的新颖性句子

投稿前仍需更新检索；若未发现新碰撞，可使用较窄的表述：

> To our knowledge, this is the first prospectively blinded evaluation of an assay-structured probabilistic model that represents paired −NADPH/+NADPH CYP inhibition measurements through a shared direct-inhibition state and a signed metabolism-dependent potency shift with an explicit positive TDI component, while propagating available curve-fit uncertainty and censoring into both regression and TDI decisions.

这句话的每个限定词都不能随意删掉。尤其不能写：

- “first multitask CYP model”；
- “first AI model for TDI”；
- “predicts mechanism-based inhibition”——当前 IC50 shift 不能区分更强的可逆代谢物、准不可逆或共价 MBI；
- “predicts clinical DDI”；
- “replaces CYP experiments”或“regulatory-ready”；
- “generalizes to all CYPs/chemical space”。

### 2.3 可检索专利检查的含义

以“time-dependent CYP inhibition / NADPH / IC50 shift / machine learning / latent model”等组合检索 Google Patents 与公开网页，未发现直接覆盖本方案的结果。该检查只能作为论文选题审计，**不是法律上的 freedom-to-operate 意见**。

## 3. 核心模型：从实验图出发，而不是从网络名字出发

对化合物 \(i\)、CYP 亚型 \(e\)：

1. 分子编码器 \(h_i=f_\theta(x_i)\) 使用 2D 分子图；主实现选 Chemprop 风格 D-MPNN，并保留 ECFP-MLP 作为结构简单的对照。
2. 潜在直接抑制状态：

   \[
   d_{ie}=g_e(h_i)
   \]

3. 代谢依赖性位移采用三态 hurdle 变量：

   \[
   z_{ie}\sim\mathrm{Categorical}(\pi^0,\pi^+,\pi^-),\qquad
   \Delta_{ie}=0\;\text{if }z=0;\quad
   \Delta_{ie}>0\;\text{if }z=+;\quad
   \Delta_{ie}<0\;\text{if }z=-
   \]

   正向位移对应潜在 TDI，负向位移允许母体被代谢成较弱抑制物、基质/探针效应或系统偏差；不能把所有负向观测强行塞进噪声。正/负幅度可用 half-normal、Gamma 或 softplus-normal；只允许预先选一种主参数化，其他作敏感性分析。

4. +NADPH 条件的潜在 pIC50：

   \[
   t_{ie}=d_{ie}+\Delta_{ie}
   \]

5. 观测模型用 Student-t 或稳健正态似然：

   \[
   y^D_{ie}\sim p(d_{ie},\sigma^D_{assay},\sigma^D_{model}),\qquad
   y^T_{ie}\sim p(t_{ie},\sigma^T_{assay},\sigma^T_{model})
   \]

   已报告的 DRC 标准差/置信区间进入 \(\sigma_{assay}\)；某个条件没有逐曲线不确定性时，以训练折内的亚型/条件层级方差估计，不能偷看验证折。模型异方差头只学习剩余不确定性，避免把实验噪声重复计算。

6. TDI 概率不是独立分类头，而是从潜变量抽样后按官方规则计算：

   \[
   P(\mathrm{TDI})=P(\Delta_{ie}>0.301\mid x_i,\text{data})
   \]

   同时严格复现 pIC50<4 时的 inferred positive 和 assigned negative 规则。由于观测可受噪声影响，实际实现应从 \((d,t)\) 的后验预测分布中蒙特卡洛计算标签概率，而不是直接阈值化均值。

7. 单浓度和 Emax 采用**辅助观测头**，先不强行拼成 Hill 曲线。只有拿到足够的浓度级原始数据、斜率和平台信息后，才预注册“完整曲线层级模型”扩展。

### CYP 间共享方式

- 共享底层分子编码器；
- CYP3A4/2C9/1A2 使用低秩共享加亚型特异头；
- CYP2D6 使用单独专家或门控分支，因为其 readout 是 Echo-MS，而其他三者主要为荧光读出；
- “2D6 单独分支”必须作为预注册消融，而不是先验宣称必然更好。公开 EDA 报告 2D6 与其他亚型相关性接近零，这只能作为设计动机。

### 为什么不加 docking/MD/蛋白语言模型

当前测试考察的是类似物系列的配体外推，CYP 口袋柔性大、探针/读出依赖明显。未经验证的 docking 分数或单个静态结构很容易成为“看起来机制、实际添噪声”的装饰。主模型先保持 2D；只有在盲测后，结构信息对预先定义的失败亚组有可复现增益，才进入补充实验。

## 4. 数据与切分：防止最常见的假提升

### 4.1 数据冻结

- 下载原始 CSV 后只读保存；记录 URL、下载时间、SHA256、行列数和 Apache-2.0 许可快照。
- RDKit 标准化脚本固定版本：去盐/保留最大有机片段、规范化电荷、标准互变异构体策略；同时保存原始 SMILES 与标准化 SMILES。
- 用 full InChIKey、connectivity block、canonical SMILES 三种键查重；立体异构体既报告合并审计，也保留独立建模版本。
- 同一化合物的 direct、TDI、Emax、single-dose 所有记录必须进入同一 fold。
- 缺失标签只做 masked loss，绝不均值填充或把缺失当阴性。

### 4.2 三层内部验证

| 层级 | 用途 | 切分规则 |
|---|---|---|
| L0 随机分子 split | 仅检查代码和上限 | 不允许作为论文主结果 |
| L1 scaffold/ECFP cluster split | 与常见文献比较 | Bemis–Murcko scaffold 与 ECFP cluster 同时报告 |
| L2 challenge-mimetic family split | 主内部证据 | 从高活性 anchor 出发，将其 top-k 类似物/聚类家族整体留出；所有成员只在一个 fold |

L2 采用 5 个预先生成并冻结的重复 split。任何超参数选择都只在训练折内部进行；外层折只做一次评估。相似度阈值、top-k 和 anchor 选择规则在看结果前写入 YAML 并提交 Git。

### 4.3 外部数据的原则

公开参赛者已报告 PubChem/ChEMBL 聚合面板与盲测 test 的精确 InChIKey 重叠为零、与训练集仅约 2.9%。因此：

- 外部 CYP 数据只作为**预训练或稳健性实验**，不把协议不一致的 pIC50 与主标签直接合并；
- IC50 与 Ki 不混为同一数值任务；不同探针、酶系统、孵育条件和 readout 建立 source/assay token；
- 主论文的 primary model 先限定为 OpenADMET 数据，外部数据增益必须通过独立消融证明；
- 公开的 [Octant CYP3A4/2J2 well-level 数据](https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release)可用于测试原始信号/QC方法，但其 +NADPH 联合效应与本挑战的 direct arm 含义不同，不能无条件混训。

## 5. 基线与消融：审稿人能够一眼判断贡献来自哪里

### 5.1 必须完成的基线

1. 训练集均值/中位数与按亚型均值；
2. ECFP4 kNN/相似物加权 read-across；
3. ECFP4 + Random Forest；
4. ECFP4 + LightGBM；
5. 单任务 Chemprop D-MPNN；
6. 普通 masked multitask D-MPNN；
7. direct 回归 + 独立 TDI classifier；
8. 同容量、同 encoder、但没有 \(t=d+\Delta\) 约束的多头模型；
9. 可获取时加入 CheMeleon 或同等级公开预训练分子编码器，但不能只比弱基线。

所有神经模型相同的参数预算区间、调参预算、5 个 seed 和早停规则。不能让新模型获得更多搜索次数。

### 5.2 决定论文是否成立的消融

| 消融 | 检验的问题 |
|---|---|
| 去掉潜变量位移，改独立 TDI 头 | 成对实验约束是否真的有贡献 |
| 去掉 CI/std，只做普通 MSE | 实验噪声建模是否改善 ST-RAE/校准 |
| 去掉单浓度辅助任务 | 多保真信号是否有效 |
| 去掉 Emax 辅助任务 | 曲线平台信息是否有效 |
| 四 CYP 全共享 vs 2D6 独立分支 | assay/亚型异质性是否需要结构化共享 |
| 随机/scaffold vs family split | 常规验证是否高估真实盲测能力 |
| 去掉检测下限/区间删失处理 | 低活性数据处理是否是关键 |
| 加/不加外部公共 CYP 数据 | 协议异质数据究竟帮助还是伤害 |

## 6. 评价与统计计划

### 6.1 预先固定的 primary endpoints

- 直接抑制：官方 **macro-averaged ST-RAE**，四 CYP 等权；
- TDI：CYP3A4/2D6 官方 **MCC**；
- primary comparison：完整潜变量模型 vs “同 encoder、同预算的 direct + 独立 TDI 模型”。

其他 MAE、R²、Spearman、Kendall、AUROC、AUPRC、Brier、ECE、coverage 和 risk–coverage 都是 secondary。

### 6.2 统计单位必须是化学家族

- 按 analog family 做 10,000 次 paired bootstrap，不能把 750 个高度相似分子当作 750 个独立样本；
- 报告效应量与 95% CI；primary comparison 只有一个，不做挑最好结果；
- secondary endpoints/亚组检验使用 Benjamini–Hochberg；
- 神经模型报告 5 seeds 的完整分布，不只报最佳 seed；
- 对每个 CYP 同时报告样本数、阳性数、独立家族数、置信区间，不用宏平均掩盖 CYP2D6 失败。

### 6.3 盲测纪律

- 在第一次提交前冻结数据清单、split、主模型、seed、预测文件和论文 primary hypotheses；
- 建议做带时间戳的 embargoed OSF registration，并公开冻结归档的 SHA256；挑战结束后再公开完整归档。若规则或平台不支持，则保留私有 Git commit、签名 tag 和第三方可信时间戳；
- 2026-09-25 中期榜单揭示后，**不得修改 primary 模型叙事**。任何修改都标作 post-leaderboard secondary analysis；
- 只由一个团队账号提交，遵守主办方“一团队一次参赛实体”的规则；
- 不从测试结构做伪标签、分布拟合或 transductive tuning，除非比赛规则明确允许且在论文中单列；为让论文最干净，primary model 不使用这些技巧。

## 7. 14 天 kill test：不过就停

### Day 1–2：审计与单元测试

- 冻结原始文件与环境；
- 完成 SMILES/InChIKey/重复/缺失/检测下限/CI 审计；
- 用官方教程逐条复现 TDI 标签，目标 **100% 一致**；
- 对 MA-ST-RAE 与 MCC 写边界单元测试；
- 统计每个 TDI 端点的阳性化合物数和独立 analog family 数。

### Day 3–5：强基线

- 跑 median、kNN、RF、LightGBM、单任务/普通多任务 D-MPNN；
- 冻结 5 个 challenge-mimetic outer folds；
- 比较随机、scaffold、family split 的性能落差；
- 估算可达到的噪声下限，排除泄漏导致的“不可能好成绩”。

### Day 6–9：最小潜变量原型

- 只实现 direct \(d\)、三态 hurdle \(\Delta\)、TDI 概率和 CI-aware likelihood；
- 暂不加入 3D、解释性、外部数据或复杂 foundation model；
- 首先确认梯度、数值稳定性、校准和阈值规则无误。

### Day 10–12：关键消融

- 独立 TDI 头 vs 潜变量；
- MSE vs CI-aware；
- 四任务全共享 vs CYP2D6 分支；
- 至少 5 seeds，family-level paired bootstrap。

### Day 13–14：GO/STOP 评审

同时满足以下条件才升级为代表作：

1. 标签和官方指标实现全部单元测试通过，TDI 标签复现 100%；
2. CYP3A4、2D6 各自至少有足以估计 MCC 的阳性结构家族；暂定下限为 **≥100 个阳性且 ≥20 个独立家族**。若不足，必须扩大 CI 并降低主张，不能靠 molecule-level p 值冒充证据；
3. 在 5 个 family splits 中至少 4 个方向一致；相对最佳强基线的 direct MA-ST-RAE 改善建议达到 **≥5%**，且 pooled family-bootstrap 95% CI 不跨 0；
4. TDI MCC 相对同容量独立分类头的绝对改善建议达到 **≥0.05**，且 95% CI 不跨 0；
5. 改善在去除近重复、低/高相似度分层和单独 CYP 报告中仍存在；
6. “潜变量”和“CI/删失”两个核心部件中至少两个预注册贡献成立；若只有更大 encoder 有效，项目不成立。

判定规则：

- **双任务均通过**：GO，按完整代表作推进；
- **direct 通过、TDI 不通过**：不得继续讲机制一致 TDI；可把竞赛作为工程项目，但不作为该代表作；
- **TDI 通过、direct 不通过**：将论文收窄到 TDI assay observation model，并重新做一次新颖性审计；
- **都不通过**：STOP，不用漂亮图掩盖失败。

## 8. 盲测后的论文成败门槛

### 够投领域强刊

- 锁定模型在 fully blinded analog families 上显著优于同预算强基线；
- family-bootstrap CI 清楚，结论不依赖一两个家族；
- 潜变量、CI/删失和验证设计三项至少两项有稳定贡献；
- 代码、环境、预测文件、负结果可完全复现。

### 才值得冲 Nature 子刊

以下不是保证，而是最低入场条件：

- 盲测位于明显领先区间，且优势超过实验/seed 变异，不只是排行榜差 0.001；
- 方法可以在另一个成对干预/多保真药物测定数据上复现，不是 CYP 专用小技巧；
- 能证明常用随机/scaffold 验证与真实 blind analog-series 表现存在系统性偏差，并给出可推广的修正；
- 最好获得主办方释放的更细粒度曲线或 well-level 数据，或与数据生成者合作完成独立分析；
- 生物学表述严格：预测的是 assay-defined TDI shift，不冒充临床 DDI 或共价机制。

若只有普通竞争成绩，优先考虑 Journal of Cheminformatics、Journal of Chemical Information and Modeling、Digital Discovery、Patterns 等。分区会变化，投稿当年再核验；不要为了“一区”牺牲期刊与故事匹配。

## 9. 预设论文图表

1. **Figure 1 — Assay-to-model graph**：−NADPH direct、+NADPH TDI arm、潜在 \(d\) 与 \(\Delta\)、最终两个 challenge outputs。
2. **Figure 2 — Data and split audit**：标签稀疏、CYP/assay 差异、analog-family 切分、检测下限和 CI 分布。
3. **Figure 3 — Main prospective result**：内部 family CV 与盲测的 ST-RAE/MCC，family bootstrap 95% CI。
4. **Figure 4 — Ablation matrix**：潜变量、CI/删失、辅助任务、2D6 分支的效应量。
5. **Figure 5 — Calibration and decision utility**：TDI reliability、coverage、risk–coverage，以及不同阈值下的漏报/复测负担。
6. **Figure 6 — Analog-series case studies**：预注册规则挑选成功与失败家族；只做结构—误差解释，不事后挑“最漂亮”的例子。

补充材料必须包含所有 seed、所有模型、完整数据审计、阈值敏感性、失败亚型、近重复移除结果、外部数据协议表和 license/provenance。

## 10. 六个最可能毁稿的漏洞与封堵方式

| 漏洞 | 审稿人会怎么打 | 必须封堵 |
|---|---|---|
| 把 TDI 当 MBI | IC50 shift 不能识别具体代谢物或共价机制 | 全文使用 assay-defined TDI；机制只作为可能解释 |
| 分子当独立样本 | 750 test 是相似物家族，普通 bootstrap 会虚假缩小 CI | family split + family bootstrap |
| 忽略检测下限 | pIC50<4 不是精确连续值，直接 MSE 会学习伪标签 | 区间删失/弱化似然；精确复现官方规则 |
| CYP2D6 混同 | 2D6 的 Echo-MS 与三种荧光 assay 不同 | assay token、独立分支和逐 CYP 结果 |
| 榜单泄漏 | 9 月 25 日一次全 test 揭示会诱发调参 | 揭榜前冻结 primary model；之后只做标注清楚的 secondary |
| 创新说太宽 | 多任务 CYP、TDI QSAR 已有大量先例 | 主张只锁定“paired observation model + uncertainty/censoring + prospective family blind test” |

额外红线：不把 docking 图当验证；不把注意力权重当机制解释；不只报最佳 seed；不隐藏阴性消融；不使用未经许可的专有数据；不把竞赛排名等同临床价值。

## 11. 时间表（按当前日期倒排）

| 日期 | 冻结交付物 |
|---|---|
| 2026-08-28 至 08-31 | 数据 manifest、novelty ledger、标签/指标测试、5 个 family splits |
| 2026-09-01 至 09-05 | 全部强基线和噪声/泄漏审计 |
| 2026-09-06 至 09-12 | 最小潜变量模型与数值稳定性测试 |
| 2026-09-13 至 09-17 | 核心消融、5 seeds、family bootstrap |
| 2026-09-18 至 09-21 | 模型卡、预注册、Git release、冻结预测 |
| 2026-09-22 至 09-24 | 中期提交；只检查格式，不临时换模型 |
| 2026-09-25 | 保存揭榜结果；primary 分析保持冻结 |
| 2026-09-26 至 10-20 | 外部数据/原始信号稳健性、论文初稿、图 1–5 |
| 2026-10-21 至 11-03 | 最终规则核对、锁定提交和可复现包 |
| 2026-11-04 起 | 盲结果、预注册统计、失败分析、是否冲高刊的 go/no-go |

### 计算预算

- 单张 RTX 4090 足够；数据规模不支持“大模型越大越好”。
- 先给每类模型等额预算；总计约 50–100 GPU 小时足以完成主基线、消融和 5 seeds，实际以 Day 3 profiling 为准。
- 超参搜索使用小而固定的搜索空间；不允许新模型独占数倍算力。

## 12. 建议仓库结构与一键复现契约

```text
project/
├── configs/              # 数据、split、模型、主分析 YAML
├── data/
│   ├── raw/              # 只读，不提交大文件
│   ├── manifests/        # URL、SHA256、license、schema
│   └── splits/           # 冻结 family split IDs
├── src/
│   ├── data/
│   ├── metrics/
│   ├── models/
│   ├── uncertainty/
│   └── evaluation/
├── tests/                # label、metric、leakage、censoring 单测
├── scripts/
├── reports/              # 自动生成 CSV/JSON/图
├── paper/
└── environment.lock
```

一键入口在第一周就固定：

```bash
make setup
make audit
make baselines
make train-primary
make reproduce-paper
```

`make reproduce-paper` 必须从冻结预测生成全部表图，不能依赖手工复制 Excel 数值。

## 13. 持续防撞机制

从现在到投稿，每周固定检索并保存日期、检索式和结果：

- `("time-dependent inhibition" OR TDI) CYP machine learning NADPH`
- `paired direct time-dependent CYP inhibition model`
- `OpenADMET CYP challenge solution`
- `IC50 shift probabilistic model CYP`
- `CYP inhibition uncertainty censoring deep learning`
- arXiv、bioRxiv、ChemRxiv、PubMed、Crossref、Google Scholar、GitHub、Hugging Face 和 Google Patents。

若出现同数据同方法的预印本：

1. 先比较其最早公开时间戳；
2. 不继续争“first”，改为独立盲证、统计/观测模型或验证设计的差异；
3. 若核心三件套均重合，立即触发 STOP/PIVOT，不靠换模型名维持伪创新。

## 14. 最终决策

**建议做，但只能以“14 天可证伪试验 + 盲测冻结证据”的方式做。**

目前它优于环肽渗透、普通 DILI、LINCS 逆转、Cell Painting 跨模态、常规 CYP 多任务等候选，原因不是文献少，而是同时具备：新公开且一致的数据、成对实验结构、真实盲测试验、明确监管价值、纯计算可完成，以及尚未被公开工作占据的窄而硬的方法学空位。

真正能成为代表作的不是题目听起来新，而是最后可以拿出一条很难被反驳的证据链：

> **预先写下假设 → 冻结模型 → 类似物家族盲测 → 家族级统计 → 完整失败披露。**

如果这条证据链成立，文章有冲击力；如果不成立，按 kill rule 及时停手，避免把职业时间押在一个只是“看起来像代表作”的项目上。
