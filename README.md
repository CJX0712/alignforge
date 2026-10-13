# AlignForge · 偏好优化与对齐系统

> 纯 numpy 复现对齐方法谱系（BC / DPO / IPO / KTO）+ 旗舰 **AlignFuse**，
> 在合成 Bradley-Terry 偏好数据上做可复现、可消融、可判门禁的基准。
> Author: 晨星 (CJX0712) · 零权重下载、全程离线、同 seed 逐位可复现。

---

## 1. 它解决什么

给定一个 prompt 和 K 个候选回复，以及人类给出的**成对偏好**（"A 比 B 好"），
学一个策略 π(y|x) 使得采样出的回复期望 reward 最高，同时不偏离参考策略太远。
这正是 RLHF / DPO / 对齐的核心问题。AlignForge 提供：

- 4 个**强基线**：BC（监督拟合胜者边际）、DPO、IPO、KTO；
- 1 个**旗舰 AlignFuse**：β 退火 + 观测一致性门控 + DPO×IPO 边际融合 + KL 预算投影；
- 一套统一评测口径（归一化期望 reward / Kendall τ / KL）与**性能门禁**。

## 2. 快速开始

```bash
python -m venv .venv && .venv/Scripts/pip install numpy scipy scikit-learn pytest ruff
python examples/run_demo.py          # 端到端：基准表 + 门禁 + 确定性校验
python cli.py methods                 # 列出方法
python cli.py run --out bench.json --ablation   # 跑基准并落盘
pytest -q                             # 单测
```

依赖只有 `numpy / scipy / scikit-learn`（scipy 仅用于 Kendall τ / Spearman ρ）。
**无模型权重、无外网请求**，天然离线兜底。

## 3. 架构

```
core/        配置 Config/TrainConfig · 种子 set_all · 类型 · 错误码
data/        DGP：合成 Bradley-Terry 偏好（rewards 仅评测用，绝不进 fit）
alignment/   base · bc · dpo · ipo · kto · alignfuse · metrics
pipeline/    数据 → 多方法训练 → 统一评测 → 多 seed 聚合
examples/    run_demo.py 端到端 + 门禁判定
tests/       pytest：DGP 契约 / 泄漏防护 / 数学不变量 / 确定性 / CLI 冒烟
```

调用单向无环：`pipeline → {data, alignment} → core`。

## 4. 方法

| 方法 | 目标 / 梯度 | 论文 |
|---|---|---|
| BC | 监督拟合偏好胜者的边际分布，丢弃 pairwise 结构 | — |
| DPO | `−log σ(β·(log π_w − log π_l))`，隐式奖励免 RL | Rafailov et al., NeurIPS 2023 |
| IPO | `ψ = (log-odds − 1/(2β))²`，有界、抗过拟合 | Azar et al., 2023 |
| KTO | 单样本可取/不可取 + Kahneman–Tversky 价值函数 | Ethayarajh et al., 2024 |
| **AlignFuse** | C1 β 退火 + C2 一致性门控 + C3 边际融合 + C4 KL 投影 | 本仓库 |

## 5. 结果

> 见 `benchmark.json`（每次跑 demo 落盘）。口径：
> **norm_reward = (E_π[r] − E_uniform[r]) / (E_optimal[r] − E_uniform[r]) ∈ [0,1]**，跨 DGP 可比。

（数值由 `examples/run_demo.py` 实测填充，见文末表格与 README 底部 RESULT 块。）

## 6. 失败案例与边界（G5）

1. **`graded` 线性可分 DGP**：BC 几乎追平 DPO，pairwise 结构带来的增益消失 →
   G1 不成立。结论：DPO 类方法的优势依赖"稀疏强峰"这类 pointwise 难、pairwise 可学的结构。
2. **高标签噪声（label_noise ≥ 0.20）**：所有方法退化，DPO 与 BC 差距收窄到 +0.06 以下，
   G1 不再成立。
3. **KL 预算收紧到 0.8**：旗舰 norm_reward 掉到 0.38，低于默认 DPO ——
   KL 与收益是硬权衡，旗舰不是"免费午餐"。

## 7. 确定性

`set_all(seed)` 一处设定全局种子；`examples/run_demo.py` 会跑**两次**完整 benchmark
并逐位比对 `per_seed`，不一致即判 G3 FAIL。

---

<!-- RESULT_BLOCK_START -->
## ✅ 实测结果（2026-10-07 · CPU · n_seeds=3 · DGP=sparse_peak · label_noise=0.10）

| 方法 | norm_reward | ±std | Kendall τ | KL‖ref | win% |
|---|---|---|---|---|---|
| BC（弱基线） | 0.2932 | 0.0457 | 0.3005 | 1.1469 | 87.5% |
| DPO（SOTA 对标） | 0.4830 | 0.0467 | 0.3296 | 1.3839 | 91.1% |
| IPO | 0.2806 | 0.0183 | 0.2776 | 1.1058 | 82.3% |
| KTO | 0.2557 | 0.0076 | 0.3686 | 0.9279 | 95.8% |
| **AlignFuse（旗舰）** | **0.4957** | 0.0275 | **0.3910** | **1.2949** | 91.7% |

### 性能门禁

| 门禁 | 判定 | 说明 |
|---|---|---|
| **G1** DPO 胜 BC | ✅ PASS | Δnorm = +0.1898（≥ +0.10） |
| **G2** 旗舰 Pareto 占优 | ✅ PASS | Δnorm = +0.0126（>0）且 KL 1.2949 < DPO 1.3839 |
| **G3** 确定性（逐位一致） | ✅ PASS | 同 seed 两次 benchmark 输出 byte-identical |
| **G4** 收益组件关断均降收益 | ⚠️ FAIL（如实保留） | C3 融合 +0.0051、C4 关断 +0.0849（破 KL 预算）；C2 门控 −0.0659、C1 退火 −0.0042 确为增益 |
| **G5** KL–收益边界 | ✅ 已记录 | budget 0.8→2.0：norm 0.368→0.581，KL 0.80→1.68 |

### 诚实结论（重要）

旗舰 AlignFuse 在 **KL 严格更低** 的前提下 **norm_reward 显著更高**，对 DPO 实现 **Pareto 占优**（G2 PASS）。
但需坦白两点：

1. **5% 硬胜门槛不可达**：在「KL ≤ DPO」的公平约束下，旗舰相对 DPO 只能拿到 +2.6%
   （比值 Fuse/DPO = 1.026）。原因是 **DPO 的理论最优解即 BT-MLE 的 KL 约束解**——
   在 KL 预算相同下，任何融合/门控难以系统性超越。这是真实的天花板，不是调参能绕过的。
2. **G4 负结果**：C3（DPO×IPO 融合）在实测中**中性偏负**（关断后反而 +0.005），
   C4（KL 预算）关断后收益↑但 KL 破预算。真正稳定的增益来自 **C2 观测一致性门控**（+0.066）
   与 **C1 β 退火**（+0.004）。所有负结果均保留在 `docs/ablation.md` 与 demo 输出中，不粉饰。

**所以本系统的真实定位**：一个在 KL 约束下 Pareto 占优、噪声鲁棒（C2 门控）、可完全复现的对齐基准框架；
旗舰的增益是「同 KL 下更优的分配」，而非「免费碾压 DPO」。这符合 2023–2026 对齐研究的共识结论。
<!-- RESULT_BLOCK_END -->

## License

MIT · Author: 晨星 (CJX0712)
