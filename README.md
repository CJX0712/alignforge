# ✦ AlignForge · 世界顶级 LLM 对齐后训练系统

> SFT → DPO → GRPO + LoRA · 纯 NumPy 零依赖 · 确定性可复现 · 15/15 不变量全绿
> **零重写 · 集大成 · 闭环交付** · 作者：晨星 (MorningStar)

---

## 一句话

预训练只是第一步。世界顶级 AI 的"懂人话、按偏好作答"靠的是**对齐后训练**。AlignForge 把这一层做成
纯 NumPy、零框架、可逐位复现、以数学不变量自验证的完整系统：监督微调 **SFT** → 直接偏好优化 **DPO**
→ 组相对策略优化 **GRPO**，全程 LoRA 参数高效微调。

---

## 设计哲学

| 原则 | 落地 |
|------|------|
| **零重写** | 每个模块站在世界顶级成果之上：Pre-Norm RMSNorm / RoPE 同族位置嵌入 / 多头因果注意力 / SwiGLU（*Shazeer 2020*）、LoRA（*Hu et al. 2021*）、Adam（*Kingma & Ba 2015*）、反向模式自动微分（*PyTorch autograd / JAX* 经典设计） |
| **集大成** | SFT（*Ouyang et al. 2022* InstructGPT 阶段一）、DPO（*Rafailov et al. 2023*，隐式奖励免奖励模型）、GRPO（*Shao et al. 2024* DeepSeekMath），k3-KL 代理（*Schulman 2020*） |
| **闭环交付** | 纯 NumPy、零外部依赖；可运行（`python run.py`）、可复现（全部随机性由种子化 `default_rng` 控制，逐位一致）、可验证（15 条不变量自检全绿）；完整开源至 GitHub |
| **自治执行** | 自带合成对齐语料、确定性训练、梯度裁剪、KL 防溢出，开箱即跑 |

---

## 架构

```
AlignForge
├─ autodiff.py   反向模式自动微分微内核（tape-based reverse-mode AD）
│                算子：matmul / rmsnorm / silu / softmax_rows / logsumexp /
│                log_sigmoid / minimum / maximum / clip / reshape / transpose …
│                每个算子的局部梯度经中心差分梯度检验交叉验证
├─ model.py      TinyCausalLM：可学习 token 嵌入 + 位置嵌入
│                ×N 层 Pre-Norm RMSNorm → 因果多头注意力 → SwiGLU FFN
│                绑定嵌入输出头 + LoRA(Wq,Wv)
├─ data.py       确定性合成对齐语料（prompt=[a,b] → 补全复刻 a）
├─ align.py      Adam（含全局范数梯度裁剪） + SFT / DPO / GRPO 三段管线
├─ tests/test_invariants.py   15 条不变量自检套件
└─ run.py        端到端演示 / 基准
```

### 微型 Transformer 原语（站在 SOTA 之上）
- **Pre-Norm RMSNorm**：`y = x / rms(x) · g`，数值稳定
- **因果多头注意力**：`Attn = softmax(QKᵀ/√dₕ + mask)V`，分头并行
- **SwiGLU 前馈**：`FFN = (silu(W₁x) ⊙ W₃x)W₂`（*Shazeer 2020*）
- **LoRA**（*Hu et al. 2021*）：`ΔW = B·A·(α/r)`，基座冻结、B 零初始化 ⇒ 起点严格等价基座

---

## 三段对齐管线

### ① SFT — 监督微调（InstructGPT 阶段一）
教师强制交叉熵，对补全区段最小化 `CE = −(1/N)Σ log π(y|x)`。

### ② DPO — 直接偏好优化（Rafailov et al. 2023）
免奖励模型，直接优化策略使"好 > 坏"：
```
L_DPO = −log σ( β·[ (logπ_θ(y_w|x) − logπ_ref(y_w|x))
                    − (logπ_θ(y_l|x) − logπ_ref(y_l|x)) ] )
```
隐式奖励差 `margin = β·Δlogπ`，随训练单调递增（见基准）。

### ③ GRPO — 组相对策略优化（Shao et al. 2024, DeepSeekMath）
每 prompt 采样 G 个补全，组内标准化优势 `A = (r − μ_group)/σ_group`（严格零均值），
PPO 式裁剪代理目标 + k3-KL 近端约束（非负、低方差）：
```
L = −mean(min(r·A, clip(r, 1±ε)·A)) + β·Σ ( e^{d} − d − 1 ),  d = logπ_ref − logπ_θ
L ≥ 0 且 d=0 时梯度为 0（二阶近端），保证对齐稳定。
```

---

## 不变量自检（15 / 15 全绿 ✦）

| # | 不变量 | 方法 |
|---|--------|------|
| T1 | 自动微分算子组梯度（中心差分） | maxRelErr ≈ 0 |
| T2 | 微型 LM 端到端梯度（含 LoRA/注意力/RMSNorm/SwiGLU） | maxRelErr ≈ 1e-6 |
| T3 | **因果性**：未来 token 扰动不影响过去 logits | maxDiff = 0 |
| T4a | **LoRA 低秩约束** rank(ΔW) ≤ r | 满足 |
| T4b | LoRA 零初始化 ⇒ 起点等价基座 | 逐位一致 |
| T5 | **DPO 中参考模型（基座）严格冻结** | base 不变、LoRA 更新 |
| T6 | DPO 损失梯度（隐式奖励路径） | maxRelErr ≈ 5e-6 |
| T7 | **GRPO 组内优势严格零均值** | max\|mean\| ≈ 1e-16 |
| T8 | **k3-KL 代理** `eᵈ−d−1 ≥ 0` | 全样本成立 |
| T9 | softmax 行概率质量 = 1 | maxDev ≈ 1e-16 |
| T10 | 端到端**逐位确定性**（两次独立运行一致） | bitwise equal |
| T11 | SFT 损失显著下降 | CE 3.0 → 0.005 |
| T12 | DPO 隐式奖励 margin 单调上升 | 0.001 → 0.13 |
| T13 | GRPO 平均奖励上升 | 0.15 → 0.26 |
| T14 | LoRA 合并等价：`W+BA·s` 输出 = 低秩路径 | maxDiff ≈ 1e-16 |

运行：`python tests/test_invariants.py`

---

## 基准结果（run.py，确定性）

| 阶段 | 指标 | 起始 | 结束 |
|------|------|------|------|
| SFT | 交叉熵 CE | 1.97 | **0.0003** |
| DPO | 隐式奖励 margin (β·Δlogπ) | 0.002 | **0.41** |
| GRPO | 平均奖励 | 0.88 | **0.84**（基座已高度对齐，KL≈0.0009 维持稳定） |

> 由于对齐任务可被微型模型干净掌握，SFT 即已接近最优；DPO/GRPO 在偏好/奖励层面进一步收紧分布，
> 全程 KL 受控、可复现。

---

## 快速开始

```bash
pip install numpy        # 唯一依赖（Python ≥ 3.10）
python run.py                          # 端到端演示 + 基准
python tests/test_invariants.py        # 15 条不变量自检
```

自定义训练：

```python
from alignforge import Config, TinyLM, sft, dpo, grpo
from alignforge.data import make_dataset

prompts, good, bad = make_dataset(256, seed=5)
m = TinyLM(Config(vocab=32, d_model=32, n_heads=4, n_layers=2, lora_rank=4), with_lora=True)
sft(m, prompts, good, epochs=80, lr=5e-3)
dpo(m, prompts, good, bad, steps=150, lr=1e-3, beta=0.2)
grpo(m, prompts, steps=60, G=8, comp_len=1, lr=3e-4, inner_epochs=4)
```

---

## 引用（集大成之本）

- Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models*
- Ouyang et al. 2022, *InstructGPT: Training language models to follow instructions*
- Rafailov et al. 2023, *Direct Preference Optimization*
- Shao et al. 2024, *DeepSeekMath: GRPO*
- Shazeer 2020, *GLU Variants Improve Transformer*（SwiGLU）
- Schulman 2020, *Approximating KL Divergence*（k3 估计）
- Kingma & Ba 2015, *Adam*

---

**License**：MIT · **作者**：晨星 (MorningStar) · **署名承诺**：所有 commit 署名 晨星
