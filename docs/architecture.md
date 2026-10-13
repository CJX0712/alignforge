# AlignForge 架构设计

Author: 晨星 (CJX0712)

## 1. 分层与依赖方向

```
cli.py / examples/
        │
   pipeline/  AlignPipeline：建数据 → 跑方法 → 评测 → 多 seed 聚合
        │
   ┌────┴─────┐
 data/      alignment/
 (DGP)      (base/bc/dpo/ipo/kto/alignfuse/metrics)
        │
      core/  Config · TrainConfig · set_all · types · errors
```

依赖单向无环：`pipeline → {data, alignment} → core`。`core` 不反向引用任何上层。

## 2. 数据流

1. `make_synthetic(cfg)` → `PreferenceDataset`
   - 每个 prompt 持有 `rewards`（ground-truth，仅评测）与 `pairs`（观测偏好对）。
   - 偏好对按 Bradley-Terry 采样：`P(i≻j) = σ((r_i − r_j)/bt_temp)`，再按 `label_noise` 翻转。
2. `aligner.fit(dataset)` 只允许读 `pairs`（和 `k_candidates`），**禁止读 `rewards`**。
   - `single_labels()`（KTO 用）与 `_consistency()`（AlignFuse 门控用）都只由 `pairs` 导出，
     并有单测把 `rewards` 取反/平移后断言输出不变，从机制上杜绝泄漏。
3. `aligner.evaluate(dataset)` → `MetricBundle`，唯一读取 `rewards` 的地方。

## 3. 统一评测口径

| 指标 | 定义 | 为什么 |
|---|---|---|
| `expected_reward` | `E_{y∼π}[r]` 对 prompt 平均 | 原始目标 |
| `norm_reward` | `(E_π[r] − E_unif[r]) / (E_opt[r] − E_unif[r]) ∈ [0,1]` | 跨 DGP 可比：0=随机策略，1=每 prompt 都取到最优候选 |
| `kendall_tau` | 全量 (logit, reward) 对的秩相关 | 排序质量 |
| `kl_ref` | 平均 `KL(π‖π_ref)`，π_ref = uniform | 对齐代价；防止"靠无限远离参考策略刷分" |
| `win_rate_vs_ref` | `E_π[r] > E_unif[r]` 的 prompt 占比 | 稳定性 |

## 4. 各方法目标与梯度

记 `a = log π(w) − log π(l)`，ref 为 uniform 故 log-ref 项抵消；`e_i` 为第 i 个候选的 one-hot。

| 方法 | 每对梯度（对 logits） | 定点 |
|---|---|---|
| BC | `e_w`（推高胜者边际，丢掉 pairwise 结构） | — |
| DPO | `(1−σ(βa))·β·(e_w − e_l)` | `a → +∞`（无限锐化） |
| IPO | `−2(a − β/2)/β ·(e_w − e_l)` | `a = β/2`（有界，达标即停） |
| KTO | `±λβ·s(1−s)·(e_c − π)`，`s=σ(β(±z−m))` | 可取/不可取分离 |
| AlignFuse | `gate · [α·g_DPO + (1−α)·g_IPO]`，α=σ(γ(|a|−τ0)) | 见下 |

## 5. AlignFuse 四个组件

| ID | 组件 | 作用 | 关断开关 |
|---|---|---|---|
| C1 | β 退火 | β: β_min→β_max，早期稳健后期锐化 | `use_anneal=False` |
| C2 | 观测一致性门控 | 用 LOO 净胜场判断该对是否与整体证据矛盾，矛盾则降权 | `use_gate=False` |
| C2b | 标签翻转纠正 | 证据强烈反对的方向按反向对计入（可选，默认关） | `use_flip=True` |
| C3 | DPO×IPO 边际融合 | 高 margin 用 DPO 锐化，低 margin 交 IPO 温和推 | `use_ipo_fusion=False` |
| C4 | KL 预算投影 | 每轮二分缩放 logits，压回 `fuse_kl_budget` | `use_kl_guard=False` |

## 6. 确定性

- `core.seed.set_all(seed)` 一处设定 `random` / `numpy` 全局种子；
- DGP 用 `np.random.Generator`（PCG64），不依赖任何全局状态残留；
- 训练过程不含随机性（无 dropout / 无采样）；
- `AlignPipeline.determinism_payload()` 剔除墙钟耗时后逐位比对，避免"假不确定"。
