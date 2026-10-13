# -*- coding: utf-8 -*-
"""AlignForge 单元测试。Author: 晨星 (CJX0712)

覆盖：DGP 契约 / 泄漏防护 / 各 aligner 数学不变量 / 确定性 / CLI 冒烟 / 离线兜底。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from alignment import build_registry  # noqa: E402
from alignment.metrics import compute_metrics  # noqa: E402
from core.config import Config, TrainConfig  # noqa: E402
from core.errors import NotFittedError  # noqa: E402
from core.seed import set_all  # noqa: E402
from data.synthetic import make_synthetic  # noqa: E402
from pipeline.pipeline import AlignPipeline  # noqa: E402


# ----------------------------------------------------------------- DGP
def test_dgp_shape_and_finite():
    ds = make_synthetic(Config(n_prompts=8, k_candidates=6, n_pairs=12, seed=1))
    assert len(ds.prompts) == 8
    assert ds.k_candidates == 6
    for p in ds.prompts:
        assert p.rewards.shape == (6,)
        assert np.all(np.isfinite(p.rewards))
        assert len(p.pairs) == 12
        for w, l in p.pairs:
            assert w != l and 0 <= w < 6 and 0 <= l < 6


def test_dgp_determinism_same_seed():
    a = make_synthetic(Config(n_prompts=8, k_candidates=6, n_pairs=12, seed=7))
    b = make_synthetic(Config(n_prompts=8, k_candidates=6, n_pairs=12, seed=7))
    for pa, pb in zip(a.prompts, b.prompts):
        assert np.array_equal(pa.rewards, pb.rewards)
        assert pa.pairs == pb.pairs


def test_dgp_seed_changes_data():
    a = make_synthetic(Config(n_prompts=8, k_candidates=6, n_pairs=12, seed=1))
    b = make_synthetic(Config(n_prompts=8, k_candidates=6, n_pairs=12, seed=2))
    assert not np.array_equal(a.prompts[0].rewards, b.prompts[0].rewards)


def test_dgp_rejects_bad_config():
    with pytest.raises(ValueError):
        make_synthetic(Config(k_candidates=1))
    with pytest.raises(ValueError):
        make_synthetic(Config(reward_scheme="nope"))


def test_single_labels_never_read_rewards():
    """防泄漏：single_labels 只由 pairs 导出，与 rewards 排序无关。"""
    ds = make_synthetic(Config(n_prompts=6, k_candidates=8, n_pairs=20, seed=3))
    labels = ds.single_labels()
    assert len(labels) > 0
    # 构造反例：把 rewards 整体取反（真值完全翻转），标签集合必须一字不变
    ds2 = make_synthetic(Config(n_prompts=6, k_candidates=8, n_pairs=20, seed=3))
    for p in ds2.prompts:
        p.rewards = -p.rewards
    assert ds2.single_labels() == labels


# ------------------------------------------------------------- metrics
def test_norm_reward_bounds():
    ds = make_synthetic(Config(n_prompts=6, k_candidates=6, n_pairs=12, seed=5))
    for name in ["uniform", "greedy", "anti"]:
        logits = {}
        for p in ds.prompts:
            if name == "uniform":
                logits[p.prompt_id] = np.zeros(6)
            elif name == "greedy":
                lg = np.zeros(6)
                lg[int(np.argmax(p.rewards))] = 20.0
                logits[p.prompt_id] = lg
            else:
                logits[p.prompt_id] = -20.0 * p.rewards
        mb = compute_metrics(name, logits, ds, 5)
        assert np.isfinite(mb.norm_reward) and 0.0 <= mb.norm_reward <= 1.0
        if name == "uniform":
            assert mb.norm_reward < 1e-6
        if name == "greedy":
            assert mb.norm_reward > 0.99


# -------------------------------------------------------------- aligners
@pytest.mark.parametrize("name", list(build_registry().keys()))
def test_aligner_fit_finite_and_improves(name):
    """不变量：fit 后 logits 有限；在甜点 DGP 上 norm_reward > 0（确实学到）。"""
    cfg = Config(n_prompts=24, k_candidates=8, n_pairs=20, bt_temp=0.25, seed=11)
    ds = make_synthetic(cfg)
    set_all(11)
    inst = build_registry()[name](TrainConfig())
    inst.fit(ds)
    mb = inst.evaluate(ds)
    assert np.isfinite(mb.norm_reward)
    assert np.isfinite(mb.kl_ref) and mb.kl_ref >= -1e-9
    assert np.all(np.isfinite(np.asarray(list(inst.logits.values()))))
    assert mb.norm_reward > 0.0


@pytest.mark.parametrize("name", list(build_registry().keys()))
def test_aligner_not_fitted_raises(name):
    inst = build_registry()[name](TrainConfig())
    with pytest.raises(NotFittedError):
        inst.predict_logits(0)


def test_dpo_solves_single_pair_analytic():
    """不变量：单条偏好对、只含两个候选时，DPO 单调拉开 winner−loser logit 差。"""
    cfg = Config(n_prompts=1, k_candidates=2, n_pairs=1, bt_temp=0.25, seed=13)
    ds = make_synthetic(cfg)
    p = ds.prompts[0]
    w, l = p.pairs[0]
    inst = build_registry()["DPO"](TrainConfig(n_epochs=50))
    inst.fit(ds)
    lg = inst.logits[0]
    assert lg[w] > lg[l]


def test_ipo_bounded_no_explosion():
    """不变量：IPO 的 ψ 有界，长训练下 logits 不发散（有限）。"""
    cfg = Config(n_prompts=8, k_candidates=8, n_pairs=20, seed=17)
    ds = make_synthetic(cfg)
    inst = build_registry()["IPO"](TrainConfig(n_epochs=1000, lr=1.0))
    inst.fit(ds)
    arr = np.asarray(list(inst.logits.values()))
    assert np.all(np.isfinite(arr))
    assert np.max(np.abs(arr)) < 1e6


def test_alignfuse_gate_uses_only_pairs():
    """不变量：一致性门控与 rewards 无关（翻转真值 → 门控不变）。"""
    from alignment.alignfuse import AlignFuseAligner

    cfg = Config(n_prompts=4, k_candidates=8, n_pairs=20, seed=23)
    ds = make_synthetic(cfg)
    ds2 = make_synthetic(cfg)
    for p in ds2.prompts:
        p.rewards = -p.rewards + 100.0  # 真值彻底改变
    g1 = AlignFuseAligner._consistency(ds.prompts[0].pairs, 8, 1.0, 2.0)
    g2 = AlignFuseAligner._consistency(ds2.prompts[0].pairs, 8, 1.0, 2.0)
    assert g1 == g2


def test_alignfuse_kl_guard_respected():
    """不变量：开启 KL 预算投影后，平均 KL 不超过预算 + 容差。"""
    from alignment.alignfuse import AlignFuseAligner

    cfg = Config(n_prompts=8, k_candidates=8, n_pairs=20, seed=29)
    ds = make_synthetic(cfg)
    budget = 0.35
    tc = TrainConfig(fuse_kl_budget=budget)
    inst = AlignFuseAligner(tc, use_kl_guard=True)
    inst.fit(ds)
    assert inst._mean_kl(inst.logits, 8) <= budget + 1e-6
    inst2 = AlignFuseAligner(TrainConfig(fuse_kl_budget=budget), use_kl_guard=False)
    inst2.fit(ds)
    # 关断守卫后 KL 应严格更大（说明守卫真的在起作用）
    assert inst2._mean_kl(inst2.logits, 8) > inst._mean_kl(inst.logits, 8)


# ----------------------------------------------------------- determinism
def test_pipeline_bitwise_determinism():
    """G3：同 seed 两次完整 benchmark 逐位一致。"""
    cfg = Config(n_prompts=8, k_candidates=6, n_pairs=12, seed=31, n_seeds=2)
    r1 = AlignPipeline(cfg, TrainConfig()).benchmark(n_seeds=2)
    r2 = AlignPipeline(cfg, TrainConfig()).benchmark(n_seeds=2)
    p1 = AlignPipeline.determinism_payload(r1)
    p2 = AlignPipeline.determinism_payload(r2)
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


def test_ablation_runs_all_variants():
    cfg = Config(n_prompts=8, k_candidates=6, n_pairs=12, seed=37, n_seeds=1)
    rep = AlignPipeline(cfg, TrainConfig()).benchmark(n_seeds=1, ablation=True)
    for tag in ["noFusion", "noGate", "noAnneal", "noKL"]:
        assert f"AlignFuse-{tag}" in rep["summary"]["norm_reward"]


# -------------------------------------------------------------- CLI 冒烟
def test_cli_smoke(tmp_path):
    env = dict(os.environ)
    env.update(
        {
            "ALIGNFORGE_N_PROMPTS": "8",
            "ALIGNFORGE_K_CANDIDATES": "6",
            "ALIGNFORGE_N_PAIRS": "12",
            "ALIGNFORGE_N_SEEDS": "1",
            "PYTHONPATH": ROOT,
        }
    )
    out = tmp_path / "bench.json"
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "cli.py"), "run", "--out", str(out)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in data and "DPO" in data["summary"]["norm_reward"]


def test_offline_no_network_required():
    """离线兜底：全程只读本地 numpy/scipy，无下载、无权重、无外网。"""
    import core.config as cm

    src = open(cm.__file__, encoding="utf-8").read()
    for bad in ["requests.", "urllib.request", "http://", "https://"]:
        assert bad not in src
