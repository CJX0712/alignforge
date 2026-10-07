# -*- coding: utf-8 -*-
"""AlignForge CLI 入口。Author: 晨星 (CJX0712)

子命令:
  benchmark          运行多 seed 基准，打印汇总 + 写 benchmark.json
  methods            列出已注册对齐方法
  run-demo          运行端到端演示（表 + 门禁 + 确定性校验）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alignment import build_registry
from core.config import Config, TrainConfig
from pipeline.pipeline import AlignPipeline


def _cmd_benchmark(args) -> int:
    cfg = Config.from_env()
    pipe = AlignPipeline(cfg, TrainConfig())
    rep = pipe.benchmark(n_seeds=args.seeds, ablation=args.ablation)
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"benchmark.json 已写入（n_seeds={rep['n_seeds']}）")
    return 0


def _cmd_run(args) -> int:
    """运行基准并写盘（--out 指定路径，默认 ./benchmark.json）。"""
    cfg = Config.from_env()
    pipe = AlignPipeline(cfg, TrainConfig())
    rep = pipe.benchmark(n_seeds=args.seeds, ablation=args.ablation)
    out = args.out or "benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
    print(f"benchmark.json 已写入 {os.path.abspath(out)}（n_seeds={rep['n_seeds']}）")
    return 0


def _cmd_methods(_args) -> int:
    for name in build_registry():
        print(name)
    return 0


def _cmd_run_demo(_args) -> int:
    from examples.run_demo import main as demo_main

    demo_main()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="alignforge", description="AlignForge 偏好优化与对齐系统")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("benchmark", help="运行基准")
    b.add_argument("--seeds", type=int, default=None)
    b.add_argument("--ablation", action="store_true", help="含 AlignFuse 消融变体")
    sub.add_parser("methods", help="列出方法")
    r = sub.add_parser("run", help="运行基准并写 benchmark.json")
    r.add_argument("--out", type=str, default=None)
    r.add_argument("--seeds", type=int, default=None)
    r.add_argument("--ablation", action="store_true")
    sub.add_parser("run-demo", help="端到端演示")
    args = ap.parse_args(argv)
    if args.cmd == "benchmark":
        return _cmd_benchmark(args)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "methods":
        return _cmd_methods(args)
    if args.cmd == "run-demo":
        return _cmd_run_demo(args)
    # 默认跑 demo
    return _cmd_run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
