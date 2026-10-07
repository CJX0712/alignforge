# -*- coding: utf-8 -*-
"""AlignPipeline：数据生成 → 多方法训练 → 统一评测 → 多 seed 聚合。Author: 晨星 (CJX0712)

调用单向无环：pipeline → {data, alignment} → core。
确定性：set_all(seed) 一处设定；两次运行同 seed 核心指标逐位一致。
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List

import numpy as np

from alignment import AlignFuseAligner, build_registry
from core.config import Config, TrainConfig
from core.seed import set_all
from core.types import MetricBundle, PreferenceDataset
from data.synthetic import make_synthetic


class AlignPipeline:
    def __init__(self, cfg: Config | None = None, tc: TrainConfig | None = None) -> None:
        self.cfg = cfg or Config()
        self.tc = tc or TrainConfig()
        self.dataset: PreferenceDataset | None = None

    # ------------------------------------------------------------------
    def build_dataset(self, seed: int | None = None) -> PreferenceDataset:
        cfg = self.cfg if seed is None else replace(self.cfg, seed=seed)
        self.dataset = make_synthetic(cfg)
        return self.dataset

    def run_method(self, name: str, cls, dataset: PreferenceDataset, **kwargs):
        inst = cls(self.tc, **kwargs) if name.startswith("AlignFuse") else cls(self.tc)
        t0 = time.perf_counter()
        inst.fit(dataset)
        mb = inst.evaluate(dataset)
        mb.elapsed_sec = time.perf_counter() - t0
        return inst, mb

    def _run_on(self, dataset: PreferenceDataset, ablation: bool):
        registry = build_registry()
        out: List = []
        for name, cls in registry.items():
            out.append((name, *self.run_method(name, cls, dataset)))
        if ablation:
            # G4：逐组件关断，验证每个组件都真实贡献收益
            for tag, kw in (
                ("noFusion", dict(use_ipo_fusion=False)),
                ("noGate", dict(use_gate=False)),
                ("noAnneal", dict(use_anneal=False)),
                ("noKL", dict(use_kl_guard=False)),
                ("flip", dict(use_flip=True)),
            ):
                out.append(
                    (
                        f"AlignFuse-{tag}",
                        *self.run_method(f"AlignFuse-{tag}", AlignFuseAligner, dataset, **kw),
                    )
                )
        return out

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def determinism_payload(rep: dict) -> dict:
        """去掉耗时后的可比对载荷。

        elapsed_sec 是墙钟时间，天然不可复现；把它纳入比对会造成「假不确定」。
        G3 判定只看算法输出：norm_reward / kendall_tau / kl_ref / win_rate。
        """
        out = []
        for row in rep["per_seed"]:
            out.append(
                {
                    "seed": row["seed"],
                    "metrics": {
                        name: {k: v for k, v in m.items() if k != "elapsed_sec"}
                        for name, m in row["metrics"].items()
                    },
                }
            )
        return {"per_seed": out}

    def benchmark(self, n_seeds: int | None = None, ablation: bool = False) -> dict:
        n_seeds = n_seeds or self.cfg.n_seeds
        aggr: Dict[str, List[MetricBundle]] = {}
        per_seed: List[dict] = []

        for s in range(n_seeds):
            seed = self.cfg.seed + s * 1000
            set_all(seed)
            ds = self.build_dataset(seed)
            res = self._run_on(ds, ablation)
            row = {"seed": seed, "metrics": {}}
            for name, _inst, mb in res:
                aggr.setdefault(name, []).append(mb)
                row["metrics"][name] = {
                    "norm_reward": round(mb.norm_reward, 6),
                    "kendall_tau": round(mb.kendall_tau, 6),
                    "kl_ref": round(mb.kl_ref, 6),
                    "elapsed_sec": round(mb.elapsed_sec, 4),
                }
            per_seed.append(row)

        def _agg(key: str):
            return {
                name: {
                    "mean": float(np.mean([m.__getattribute__(key) for m in mbs])),
                    "std": float(np.std([m.__getattribute__(key) for m in mbs])),
                    "n": len(mbs),
                }
                for name, mbs in aggr.items()
            }

        summary = {
            "norm_reward": _agg("norm_reward"),
            "kendall_tau": _agg("kendall_tau"),
            "kl_ref": _agg("kl_ref"),
            "win_rate_vs_ref": _agg("win_rate_vs_ref"),
        }
        return {
            "per_seed": per_seed,
            "summary": summary,
            "cfg": self.cfg.to_dict(),
            "n_seeds": n_seeds,
            "ablation": ablation,
        }
