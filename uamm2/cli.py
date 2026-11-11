import argparse
import json
from dataclasses import replace

from .ablation import format_table, run_ablation
from .config import Config
from .simulator import Simulator


def main(argv=None):
    p = argparse.ArgumentParser(prog="uamm2", description="UAMM-2 reference simulator (synthetic workload)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="run one configuration")
    s.add_argument("--tokens", type=int, default=1024)
    s.add_argument("--policy", choices=["utility", "threshold", "static"], default="utility")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--block-size", type=int, default=16)
    s.add_argument("--attn-blocks", type=int, default=24)
    s.add_argument("--allow-drop", action="store_true")
    s.add_argument("--no-entropy", action="store_true")
    s.add_argument("--codecs", default="fp16,int8,kivi4,kivi2")

    a = sub.add_parser("ablate", help="UAMM-1 vs UAMM-2 variants")
    a.add_argument("--tokens", type=int, default=1536)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--json", action="store_true")

    h = sub.add_parser("hwvectors", help="write RTL golden test vectors")
    h.add_argument("--out", default="hw/vectors")

    args = p.parse_args(argv)
    if args.cmd == "simulate":
        cfg = replace(Config(), policy=args.policy, block_size=args.block_size,
                      attn_block_budget=args.attn_blocks, allow_drop=args.allow_drop,
                      use_entropy_coding=not args.no_entropy,
                      codecs=tuple(c for c in args.codecs.split(",") if c))
        print(json.dumps(Simulator(cfg, args.seed).run(args.tokens), indent=2))
    elif args.cmd == "ablate":
        res = run_ablation(args.tokens, seed=args.seed)
        print(json.dumps(res, indent=2) if args.json else format_table(res))
    else:
        from .hwmodel import write_vectors
        write_vectors(args.out)
        print(f"wrote vectors to {args.out}")


if __name__ == "__main__":
    main()
