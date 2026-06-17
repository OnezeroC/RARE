#!/usr/bin/env python3

from __future__ import annotations

from src.cli.pipeline_cli import build_parser
from src.pipelines.main import dispatch


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch(args)


if __name__ == "__main__":
    main()
