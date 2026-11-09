from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import CORE_STAGES, OPTIONAL_STAGES, STAGES, run_pipeline, run_stage
from .visualization import PLOT_STAGES, plot_all, plot_stage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hydrogeo-insar")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="Run the publication core workflow")
    p.add_argument("config")
    p.add_argument("--from", dest="start", choices=list(CORE_STAGES))
    p.add_argument("--to", dest="stop", choices=list(CORE_STAGES))
    p.add_argument("--only", choices=list(CORE_STAGES))

    p = sub.add_parser("stage", help="Run one core or optional stage")
    p.add_argument("config")
    p.add_argument("stage", choices=list(STAGES))

    p = sub.add_parser("plot", help="Generate result-check figures")
    p.add_argument("config")
    p.add_argument("--stage", choices=["all", *PLOT_STAGES], default="all")

    sub.add_parser("list-stages")
    args = parser.parse_args(argv)

    if args.command == "list-stages":
        print("Core stages:")
        print("\n".join(f"  {x}" for x in CORE_STAGES))
        print("\nOptional/legacy stages:")
        print("\n".join(f"  {x}" for x in OPTIONAL_STAGES))
        return 0

    cfg = load_config(args.config)
    try:
        if args.command == "plot":
            result = plot_all(cfg) if args.stage == "all" else plot_stage(cfg, args.stage)
        elif args.command == "stage":
            result = run_stage(cfg, args.stage)
        else:
            result = run_pipeline(
                cfg,
                start=args.start,
                stop=args.stop,
                only=args.only,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        print(json.dumps(
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
            indent=2,
            ensure_ascii=False,
        ))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
