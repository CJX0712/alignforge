# -*- coding: utf-8 -*-
"""AlignFuse 旗舰：门控融合偏好优化。Author: 晨星 (CJX0712)

四个可独立消融的组件（G4）：

C1 β 退火 (use_anneal)
    β_t = β_min + (β_max − β_min)·(t/T)^p。
    早期小 β → 对低置信/矛盾对稳健；后期大 β → 锐化排序。
    固定点不变（终点 β = β_max），只改变优化轨迹。

C2 观测一致性门控 (use_gate) —— 噪声鲁棒性核心
    只用「已观测的成对反馈」构造每个候选的净胜场 s_i（LOO：剔除当前对自身
    贡献，避免自证）。若当前对的胜负与整体证据一致 → 门控→1；若矛盾（极可能
    是标签翻转）→ 门控→0，其梯度被压掉。绝不读取 rewards。

C3 DPO × IPO 边际融合 (use_ipo_fusion)
    按模型当前 logit 差 |a| 自适应加权 α = σ(γ(|a| − τ0))：
      · |a| 大（置信、已拉开）→ α→1 取 DPO 继续锐化；
      · |a| 小（未拟合/矛盾）→ α→0 交回 IPO，ψ 有界、步长温和，避免噪声对上猛推。
    注：反向（高 margin 交 IPO）经实测更差 —— IPO 定点 a=β_i/2 会提前停推，
    模型塌回参考策略、KL 预算用不满（见 docs/ablation.md）。

C4 KL 预算投影 (use_kl_guard)
    每轮结束若平均 KL(π‖π_ref) 超过 kl_budget，用二分缩放全部 logits 压回预算内。
    对应 KL 约束 RLHF 的原始目标，保证「不靠无限远离参考策略刷分」。
"""

from __future__ import annotations

import numpy as np

from core.config import TrainConfig

from .base import BaseAligner


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return float(e / (1.0 + e))


class AlignFuseAligner(BaseAligner):
    name = "AlignFuse"

    def __init__(
        self,
        tc: TrainConfig | None = None,
        use_anneal: bool = True,
        use_gate: bool = True,
        use_ipo_fusion: bool = True,
        use_kl_guard: bool = True,
        use_flip: bool = False,
    ) -> None:
        super().__init__(tc)
        self.use_anneal = use_anneal
        self.use_gate = use_gate
        self.use_ipo_fusion = use_ipo_fusion
        self.use_kl_guard = use_kl_guard
        self.use_flip = use_flip
        self.diagnostics: dict = {}

    # ---------- 观测一致性统计（只依赖 pairs，绝不读 rewards） ----------
    @staticmethod
    def _consistency(pairs, K: int, kappa: float, bias: float) -> list:
        """返回每对的门控权重 ∈(0,1)。LOO 净胜场差 → sigmoid。"""
        score = np.zeros(K, dtype=np.float64)
        for w, l in pairs:
            score[w] += 1.0
            score[l] -= 1.0
        gates = []
        for w, l in pairs:
            # 剔除当前对自身贡献，防止自证放大
            d = (score[w] - 1.0) - (score[l] + 1.0)
            gates.append(_sigmoid(kappa * (d + bias)))
        return gates

    # ---------- KL 预算投影 ----------
    @staticmethod
    def _mean_kl(logits: dict, K: int) -> float:
        """平均 KL(π‖π_ref)，π_ref = uniform(1/K)。向量化，O(P·K)。"""
        if not logits:
            return 0.0
        M = np.asarray(list(logits.values()), dtype=np.float64)  # (P,K)
        z = M - M.max(axis=1, keepdims=True)
        e = np.exp(z)
        pi = e / e.sum(axis=1, keepdims=True)
        kl = np.sum(pi * (np.log(pi + 1e-12) + np.log(K)), axis=1)
        return float(np.mean(kl))

    def _project_kl(self, logits: dict, K: int, budget: float) -> None:
        """若超预算，二分缩放全部 logits 压回预算内（KL 关于缩放因子单调）。"""
        if self._mean_kl(logits, K) <= budget:
            return
        keys = list(logits.keys())
        base = np.asarray([logits[k] for k in keys], dtype=np.float64)
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            trial = {k: mid * base[i] for i, k in enumerate(keys)}
            if self._mean_kl(trial, K) <= budget:
                lo = mid
            else:
                hi = mid
        for i, k in enumerate(keys):
            logits[k] = lo * base[i]

    # ---------- 主训练 ----------
    def fit(self, dataset):
        self._init_logits(dataset)
        tc = self.tc
        lr = tc.lr
        K = dataset.k_candidates
        eye = np.eye(K)
        T = max(tc.n_epochs, 1)

        b_min = tc.fuse_beta_min
        b_max = tc.fuse_beta_max
        anneal_p = tc.fuse_anneal_power
        gamma = tc.fuse_gamma
        tau0 = tc.fuse_tau0
        kappa = tc.fuse_kappa
        bias = tc.fuse_consistency_bias
        budget = tc.fuse_kl_budget

        gate_stats = []
        flip_stats = []
        for epoch in range(T):
            if self.use_anneal:
                frac = (epoch + 1) / T
                beta_d = b_min + (b_max - b_min) * (frac**anneal_p)
            else:
                beta_d = tc.beta_dpo
            beta_i = tc.beta_ipo

            for p in dataset.prompts:
                pid = p.prompt_id
                lg = self.logits[pid]
                gates = (
                    self._consistency(p.pairs, K, kappa, bias)
                    if self.use_gate
                    else [1.0] * len(p.pairs)
                )
                grad = np.zeros(K)
                n_flip = 0
                for (w, l), g in zip(p.pairs, gates):
                    # C2b 标签纠正：证据强烈反对当前对的方向时，按「反向对」计入
                    if self.use_flip and g < 0.5:
                        w, l = l, w
                        g = 1.0 - g
                        n_flip += 1
                    a = lg[w] - lg[l]
                    delta = _sigmoid(beta_d * a)
                    # d/dθ of (log π_w − log π_l) = e_w − e_l
                    g_dpo = (1.0 - delta) * beta_d * (eye[w] - eye[l])
                    # IPO: ψ = (a − 1/(2β))² 的梯度（按 β 归一化，步长与 DPO 同量级）
                    g_ipo = -2.0 * (a - 0.5 * beta_i) / beta_i * (eye[w] - eye[l])

                    if self.use_ipo_fusion:
                        # |a| 大 → α→1 取 DPO 锐化；|a| 小（未拟合/矛盾）→ α→0 交回 IPO
                        alpha = _sigmoid(gamma * (abs(a) - tau0))
                        gstep = alpha * g_dpo + (1.0 - alpha) * g_ipo
                    else:
                        gstep = g_dpo

                    grad += g * gstep
                    if epoch == T - 1:
                        gate_stats.append(g)
                if epoch == T - 1:
                    flip_stats.append(n_flip / max(len(p.pairs), 1))

                self.logits[pid] = lg + np.clip(lr * grad, -tc.grad_clip, tc.grad_clip)

            if self.use_kl_guard:
                self._project_kl(self.logits, K, budget)

        self.diagnostics = {
            "gate_mean": float(np.mean(gate_stats)) if gate_stats else 1.0,
            "gate_min": float(np.min(gate_stats)) if gate_stats else 1.0,
            "gate_frac_below_0.5": (
                float(np.mean(np.asarray(gate_stats) < 0.5)) if gate_stats else 0.0
            ),
            "final_kl": self._mean_kl(self.logits, K),
            "flip_frac": float(np.mean(flip_stats)) if flip_stats else 0.0,
        }
        self._fitted = True
        return self
