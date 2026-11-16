"""llmscope serve | bench | trace <id>"""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="llmscope")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    sub.add_parser("bench")
    t = sub.add_parser("trace")
    t.add_argument("trace_id")
    t.add_argument("--db", default="llmscope.db")
    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("llmscope.api:app", host=args.host, port=args.port)
    elif args.cmd == "bench":
        from .bench import to_markdown

        print(to_markdown())
    elif args.cmd == "trace":
        from .store import Store
        from .tracing import render_tree

        print(render_tree(Store(args.db).spans(args.trace_id)) or "no such trace")


if __name__ == "__main__":
    main()
