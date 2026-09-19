from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import STAGES, run_pipeline, run_stage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hydrogeo-insar")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="Run all or part of the workflow")
    p.add_argument("config")
    p.add_argument("--from", dest="start", choices=list(STAGES))
    p.add_argument("--to", dest="stop", choices=list(STAGES))
    p.add_argument("--only", choices=list(STAGES))
    p = sub.add_parser("stage", help="Run one stage")
    p.add_argument("config")
    p.add_argument("stage", choices=list(STAGES))
    sub.add_parser("list-stages")
    args = parser.parse_args(argv)
    if args.command == "list-stages":
        print("\n".join(STAGES)); return 0
    cfg = load_config(args.config)
    try:
        result = run_stage(cfg, args.stage) if args.command == "stage" else run_pipeline(cfg, start=args.start, stop=args.stop, only=args.only)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
